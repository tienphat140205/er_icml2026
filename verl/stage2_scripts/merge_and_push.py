#!/usr/bin/env python3
"""
Merge FSDP sharded checkpoints and push to HuggingFace Hub.
Usage:
    python merge_and_push.py --checkpoint_dir checkpoints/ER/test_arm --repo_id username/model-name
    
This will automatically find the latest global_step checkpoint.
"""
import os
import argparse
import re
import torch
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer, AutoConfig


def find_latest_checkpoint(base_dir: str) -> str:
    """Find the latest global_step checkpoint in a directory."""
    if not os.path.exists(base_dir):
        raise ValueError(f"Directory not found: {base_dir}")
    
    # Look for global_step_* folders
    step_dirs = []
    for name in os.listdir(base_dir):
        match = re.match(r'global_step_(\d+)', name)
        if match:
            step = int(match.group(1))
            step_dirs.append((step, os.path.join(base_dir, name)))
    
    if not step_dirs:
        raise ValueError(f"No global_step_* folders found in {base_dir}")
    
    # Return the one with highest step
    step_dirs.sort(key=lambda x: x[0], reverse=True)
    latest_step, latest_dir = step_dirs[0]
    print(f"Found latest checkpoint: global_step_{latest_step}")
    return latest_dir


def merge_fsdp_checkpoint(checkpoint_dir: str, output_dir: str, model_type: str = "causal_lm"):
    """Merge FSDP sharded checkpoints into a single HuggingFace model."""
    
    # Find all shard files
    shard_files = sorted([f for f in os.listdir(checkpoint_dir) if f.startswith("model_world_size")])
    
    if not shard_files:
        raise ValueError(f"No shard files found in {checkpoint_dir}")
    
    print(f"Found {len(shard_files)} shard files")
    
    # Load and merge shards
    merged_state_dict = {}
    for shard_file in shard_files:
        shard_path = os.path.join(checkpoint_dir, shard_file)
        print(f"Loading {shard_file}...")
        shard = torch.load(shard_path, map_location="cpu")
        
        for key, value in shard.items():
            if key not in merged_state_dict:
                merged_state_dict[key] = value
    
    # Load config and tokenizer from huggingface subfolder
    hf_dir = os.path.join(checkpoint_dir, "huggingface")
    if not os.path.exists(hf_dir):
        raise ValueError(f"huggingface folder not found in {checkpoint_dir}")
    
    config = AutoConfig.from_pretrained(hf_dir)
    tokenizer = AutoTokenizer.from_pretrained(hf_dir)
    
    # Create model and load merged weights
    print("Creating model...")
    if model_type == "causal_lm":
        model = AutoModelForCausalLM.from_config(config)
    else:
        model = AutoModelForSequenceClassification.from_config(config)
    
    model.load_state_dict(merged_state_dict, strict=False)
    
    # Save merged model
    print(f"Saving merged model to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print(f"✓ Merged model saved to {output_dir}")
    return output_dir


def push_to_hub(local_dir: str, repo_id: str, token: str, private: bool = False):
    """Push merged model to HuggingFace Hub."""
    from huggingface_hub import HfApi
    
    api = HfApi(token=token)
    
    print(f"Creating repo {repo_id}...")
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    
    print(f"Uploading to {repo_id}...")
    api.upload_folder(
        folder_path=local_dir,
        repo_id=repo_id,
        repo_type="model",
        commit_message="Upload merged model",
    )
    print(f"✓ Pushed to https://huggingface.co/{repo_id}")


def main():
    parser = argparse.ArgumentParser(description="Merge FSDP checkpoints and push to HuggingFace")
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                        help="Path to checkpoint base directory (e.g., checkpoints/ER/test_arm)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for merged model (default: auto)")
    parser.add_argument("--repo_id", type=str, required=True,
                        help="HuggingFace repo ID (e.g., username/model-name)")
    parser.add_argument("--token", type=str, default=None,
                        help="HuggingFace token (or set HF_TOKEN env var)")
    parser.add_argument("--private", action="store_true",
                        help="Make repo private")
    parser.add_argument("--model_type", type=str, default="causal_lm", choices=["causal_lm", "sequence_classification"],
                        help="Model type: causal_lm (actor) or sequence_classification (router)")
    parser.add_argument("--model_name", type=str, default="actor", choices=["actor", "router"],
                        help="Which model to push: actor or router")
    parser.add_argument("--skip_merge", action="store_true",
                        help="Skip merge step, assume merged model already exists")
    args = parser.parse_args()
    
    token = args.token or os.environ.get("HF_TOKEN")
    if not token:
        print("Error: Please provide --token or set HF_TOKEN env var")
        exit(1)
    
    # Find latest checkpoint
    latest_checkpoint = find_latest_checkpoint(args.checkpoint_dir)
    
    # Get the specific model folder (actor or router)
    model_checkpoint = os.path.join(latest_checkpoint, args.model_name)
    if not os.path.exists(model_checkpoint):
        print(f"Error: {args.model_name} folder not found in {latest_checkpoint}")
        exit(1)
    
    print(f"Using checkpoint: {model_checkpoint}")
    
    output_dir = args.output_dir or os.path.join(latest_checkpoint, f"merged_{args.model_name}")
    
    if not args.skip_merge:
        merge_fsdp_checkpoint(model_checkpoint, output_dir, args.model_type)
    
    push_to_hub(output_dir, args.repo_id, token, args.private)


if __name__ == "__main__":
    main()
