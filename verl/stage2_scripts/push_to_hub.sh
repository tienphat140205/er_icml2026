#!/bin/bash
# Merge and push models to HuggingFace after training
# Automatically finds the latest checkpoint

export HF_TOKEN='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'

# Base checkpoint directory (change if needed)
CHECKPOINT_BASE="checkpoints/ER/test_arm"

# Push Actor (LLM)
echo "=== Pushing Actor Model ==="
python stage2_scripts/merge_and_push.py \
    --checkpoint_dir "${CHECKPOINT_BASE}" \
    --repo_id tp140205/arm-actor \
    --model_name actor \
    --model_type causal_lm

# Push Router
echo "=== Pushing Router Model ==="
python stage2_scripts/merge_and_push.py \
    --checkpoint_dir "${CHECKPOINT_BASE}" \
    --repo_id tp140205/arm-router \
    --model_name router \
    --model_type sequence_classification

echo "=== Done! ==="
