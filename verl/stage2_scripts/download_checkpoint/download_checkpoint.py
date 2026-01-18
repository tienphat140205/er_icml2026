"""
Download raw checkpoint from HuggingFace Hub.
Usage:
    python download_checkpoint.py --step 25
    python download_checkpoint.py --step 50 --repo tp140205/arm-checkpoint
    python download_checkpoint.py --step 25 --output ./checkpoints
"""
import argparse
import os
from huggingface_hub import snapshot_download


def main():
    parser = argparse.ArgumentParser(description="Download raw checkpoint from HuggingFace Hub")
    parser.add_argument("--step", type=int, required=True,
                        help="Checkpoint step number (e.g., 25)")
    parser.add_argument("--repo", type=str, default="tp140205/arm-checkpoint",
                        help="HuggingFace repo base name (default: tp140205/arm-checkpoint)")
    parser.add_argument("--output", type=str, default="./checkpoints",
                        help="Local directory to save checkpoint")
    parser.add_argument("--token", type=str, default=None,
                        help="HuggingFace token (optional, uses cached token if not provided)")
    args = parser.parse_args()
    
    repo_full = f"{args.repo}-step{args.step}"
    checkpoint_dir = os.path.join(args.output, f"global_step_{args.step}")
    
    print("=" * 60)
    print("Downloading Checkpoint from HuggingFace Hub")
    print("=" * 60)
    print(f"\n📦 Repo: {repo_full}")
    print(f"📁 Output: {checkpoint_dir}")
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    print("\n⬇️  Downloading...")
    local_dir = snapshot_download(
        repo_id=repo_full,
        local_dir=checkpoint_dir,
        token=args.token,
        repo_type="model",
    )
    
    print(f"\n✅ Downloaded to: {local_dir}")
    print("=" * 60)
    
    # List downloaded files
    print("\n📋 Downloaded files:")
    for root, dirs, files in os.walk(local_dir):
        level = root.replace(local_dir, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f'{indent}{os.path.basename(root)}/')
        subindent = ' ' * 2 * (level + 1)
        for file in files[:10]:  # Limit to first 10 files per dir
            print(f'{subindent}{file}')
        if len(files) > 10:
            print(f'{subindent}... and {len(files) - 10} more files')
    
    print("\n" + "=" * 60)
    print("To resume training, run:")
    print(f"  bash run.sh trainer.resume_mode={checkpoint_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
