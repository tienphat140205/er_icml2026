#!/bin/bash
# Download checkpoint from HuggingFace Hub for resume training
# Usage: 
#   ./download_checkpoint.sh <step_number>
#   ./download_checkpoint.sh 25
#   ./download_checkpoint.sh 50 tp140205/arm-checkpoint

set -e

# Default values
STEP=${1:-25}
CHECKPOINT_REPO=${2:-"tp140205/arm-checkpoint"}
OUTPUT_DIR=${3:-"checkpoints"}

REPO_FULL="${CHECKPOINT_REPO}-step${STEP}"
CHECKPOINT_DIR="${OUTPUT_DIR}/global_step_${STEP}"

echo "============================================================"
echo "Downloading Checkpoint Step ${STEP} from HuggingFace Hub"
echo "============================================================"
echo ""
echo "📦 Repo: ${REPO_FULL}"
echo "📁 Output dir: ${CHECKPOINT_DIR}"
echo ""

# Download entire checkpoint folder
echo "⬇️  Downloading checkpoint..."
huggingface-cli download "${REPO_FULL}" --local-dir "${CHECKPOINT_DIR}" --repo-type model

echo ""
echo "============================================================"
echo "✅ Download complete!"
echo "============================================================"
echo ""
echo "📁 Checkpoint saved to: ${CHECKPOINT_DIR}"
echo ""
echo "To resume training, run:"
echo "  bash run.sh trainer.resume_mode=${CHECKPOINT_DIR}"
echo ""
echo "Folder structure:"
ls -la "${CHECKPOINT_DIR}"
