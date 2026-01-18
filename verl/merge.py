"""
Merge LoRA adapter into base model and save for vLLM.
"""
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, default="./merged_model")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--adapter_path", type=str, default="tonyshelby/qwen2.5_3b_checkpoints")
    parser.add_argument("--adapter_subfolder", type=str, default="checkpoint-3654")
    args = parser.parse_args()
    
    print("=" * 60)
    print("Merging LoRA Adapter into Base Model")
    print("=" * 60)
    
    print(f"\n📦 Base model: {args.base_model}")
    print(f"📦 Adapter: {args.adapter_path}/{args.adapter_subfolder}")
    print(f"📦 Output: {args.output_path}")
    
    # Load tokenizer
    print("\n🔧 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.adapter_path,
        subfolder=args.adapter_subfolder,
        trust_remote_code=True
    )
    
    # Load base model
    print("\n🔧 Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Load and merge adapter
    print("\n🔧 Loading LoRA adapter...")
    model = PeftModel.from_pretrained(
        base_model,
        args.adapter_path,
        subfolder=args.adapter_subfolder,
    )
    
    print("\n🔧 Merging weights...")
    model = model.merge_and_unload()
    
    # Save merged model
    print(f"\n💾 Saving merged model to {args.output_path}...")
    os.makedirs(args.output_path, exist_ok=True)
    model.save_pretrained(args.output_path)
    tokenizer.save_pretrained(args.output_path)
    
    print("\n✅ Done! Merged model saved to:", args.output_path)
    print("=" * 60)


if __name__ == "__main__":
    main()
