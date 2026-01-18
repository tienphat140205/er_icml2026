"""
Router-Guided Multi-Dataset Preprocessing.
Uses the router model to predict the best tag for each prompt,
then injects that tag into the prompt for generation.

Supports: gsm8k, MATH, csqa, BBH, SVAMP, openbookqa, AIME2025

Usage:
    python preprocess_all.py --router_path tp140205/arm-router-step100
    python preprocess_all.py --router_path ./checkpoints/global_step_100/router --datasets gsm8k MATH csqa
"""

import os
import argparse
import datasets
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm

# Tag mapping: class index -> tag token
TAG_MAPPING = {
    0: "<ANSWER>\n",  # ANSWER
    1: "<COT>\n",      # COT
    2: "<LONG_COT>\n", # LONG_COT
    3: "<CODE>\n",     # CODE
}

LABEL_NAMES = ["ANSWER", "COT", "LONG_COT", "CODE"]

# Dataset configurations
DATASET_CONFIGS = {
    "gsm8k": {
        "origin_path": "gsm8k",
        "data_source": "my_data/gsm8k",
        "ability": "math",
        "test_file": "gsm8k_test.jsonl",
        "train_file": "gsm8k_train.jsonl",
    },
    "MATH": {
        "origin_path": "MATH",
        "data_source": "my_data/MATH",
        "ability": "math",
        "test_file": "MATH_test.jsonl",
        "train_file": "MATH_train.jsonl",
    },
    "csqa": {
        "origin_path": "csqa",
        "data_source": "my_data/csqa",
        "ability": "commonsense",
        "test_file": "csqa_test.jsonl",
        "train_file": "csqa_train.jsonl",
    },
    "BBH": {
        "origin_path": "BBH",
        "data_source": "my_data/BBH",
        "ability": "reasoning",
        "test_file": "BBH_test.jsonl",
        "train_file": None,
    },
    "SVAMP": {
        "origin_path": "SVAMP",
        "data_source": "my_data/SVAMP",
        "ability": "math",
        "test_file": "SVAMP_test.jsonl",
        "train_file": None,
    },
    "openbookqa": {
        "origin_path": "openbookqa",
        "data_source": "my_data/openbookqa",
        "ability": "commonsense",
        "test_file": "openbookqa_test.jsonl",
        "train_file": None,
    },
    "AIME2025": {
        "origin_path": "AIME2025",
        "data_source": "my_data/AIME2025",
        "ability": "math",
        "test_file": "AIME2025_test.jsonl",
        "train_file": None,
    },
}


def load_router(router_path: str, device: str = "cuda"):
    """Load the router model and tokenizer."""
    print(f"Loading router from: {router_path}")
    tokenizer = AutoTokenizer.from_pretrained(router_path, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        router_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).to(device)
    model.eval()
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer


def predict_best_tag(model, tokenizer, questions: list, batch_size: int = 32, device: str = "cuda"):
    """Predict the best tag for each question using the router."""
    predictions = []
    
    for i in tqdm(range(0, len(questions), batch_size), desc="Router prediction"):
        batch_questions = questions[i:i + batch_size]
        
        inputs = tokenizer(
            batch_questions,
            padding=True,
            truncation=True,
            max_length=2048,
            return_tensors="pt"
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            pred_classes = outputs.logits.argmax(dim=-1).cpu().tolist()
        
        predictions.extend(pred_classes)
    
    return predictions


def process_dataset(
    dataset_name: str,
    model,
    tokenizer,
    origin_data_dir: str,
    output_dir: str,
    batch_size: int,
    device: str,
    split: str = "test"
):
    """Process a single dataset with router-guided tags."""
    config = DATASET_CONFIGS[dataset_name]
    
    # Determine file to load
    if split == "test":
        data_file = config["test_file"]
    else:
        data_file = config.get("train_file")
        if data_file is None:
            print(f"⚠️  {dataset_name} does not have a {split} split, skipping...")
            return None
    
    data_path = os.path.join(origin_data_dir, config["origin_path"], data_file)
    
    if not os.path.exists(data_path):
        print(f"⚠️  File not found: {data_path}, skipping...")
        return None
    
    print(f"\n{'='*60}")
    print(f"Processing {dataset_name} ({split})")
    print(f"{'='*60}")
    
    # Load dataset
    dataset = datasets.load_dataset('json', data_files={split: data_path})
    data = dataset[split]
    
    # Extract questions
    questions = [ex['question'] for ex in data]
    
    # Predict best tags
    print(f"Predicting tags for {len(questions)} samples...")
    predicted_classes = predict_best_tag(model, tokenizer, questions, batch_size, device)
    
    # Print tag distribution
    print("\n📊 Tag Distribution:")
    for i, name in enumerate(LABEL_NAMES):
        count = predicted_classes.count(i)
        pct = count / len(predicted_classes) * 100 if predicted_classes else 0
        print(f"  {name}: {count} ({pct:.1f}%)")

    # Create processed dataset
    def make_map_fn(split_name, predicted_tags, cfg):
        def process_fn(example, idx):
            question = example['question']
            answer = example.get('answer', '')
            
            tag_idx = predicted_tags[idx]
            tag_token = TAG_MAPPING[tag_idx]
            tag_name = LABEL_NAMES[tag_idx]
            
            prompt = [{
                "role": "user",
                "content": question,
            }, {
                "role": "assistant",
                "content": tag_token,
            }]

            return {
                "data_source": cfg["data_source"],
                "prompt": prompt,
                "ability": cfg["ability"],
                "reward_model": {
                    "style": "rule",
                    "ground_truth": answer
                },
                "extra_info": {
                    'split': split_name,
                    'index': idx,
                    'answer': answer,
                    "question": question,
                    "predicted_tag": tag_name,
                    "predicted_tag_idx": tag_idx,
                }
            }
        return process_fn

    processed = data.map(
        function=make_map_fn(split, predicted_classes, config),
        with_indices=True
    )

    # Save
    output_filename = f"router_guided_{dataset_name}_{split}.parquet"
    output_path = os.path.join(output_dir, output_filename)
    processed.to_parquet(output_path)
    
    print(f"✅ Saved: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Router-guided multi-dataset preprocessing")
    parser.add_argument('--origin_data', default='../../../data/jsonl',
                        help='Base directory containing dataset jsonl folders')
    parser.add_argument('--local_dir', default='../../../data/parquet/router_guided',
                        help='Output directory for processed parquet files')
    parser.add_argument('--router_path', type=str, required=True,
                        help='Path to router model (HuggingFace or local)')
    parser.add_argument('--datasets', nargs='+', 
                        default=['gsm8k', 'MATH', 'csqa'],
                        choices=list(DATASET_CONFIGS.keys()),
                        help='Datasets to process')
    parser.add_argument('--splits', nargs='+', default=['test'],
                        choices=['train', 'test'],
                        help='Splits to process')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for router prediction')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    # Load router
    model, tokenizer = load_router(args.router_path, args.device)
    
    # Create output directory
    os.makedirs(args.local_dir, exist_ok=True)
    
    # Process each dataset
    processed_files = []
    for dataset_name in args.datasets:
        for split in args.splits:
            result = process_dataset(
                dataset_name=dataset_name,
                model=model,
                tokenizer=tokenizer,
                origin_data_dir=args.origin_data,
                output_dir=args.local_dir,
                batch_size=args.batch_size,
                device=args.device,
                split=split
            )
            if result:
                processed_files.append(result)
    
    # Summary
    print(f"\n{'='*60}")
    print("📦 Processing Complete!")
    print(f"{'='*60}")
    print(f"Processed {len(processed_files)} files:")
    for f in processed_files:
        print(f"  - {os.path.basename(f)}")
    
    # Generate test_files string for run script
    print(f"\n📝 Copy this for your run script:")
    parquet_list = ", ".join([f"'{f}'" for f in processed_files])
    print(f"test_files=\"[{parquet_list}]\"")


if __name__ == '__main__':
    main()
