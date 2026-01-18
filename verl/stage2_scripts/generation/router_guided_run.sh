#!/bin/bash
# Format Distribution Sampling Generation Script
# 
# This script uses router to get probability distribution once per batch,
# then samples n_samples formats from that distribution for generation.
#
# Each of the 8 generations will use a different sampled format tag.

export CUDA_VISIBLE_DEVICES=0,1
export N_GPUS=2
export TP_SIZE=1
export DATA_DIR=/workspace/arm/verl/data/parquet
export MODEL_PATH=/workspace/checkpoints/model
export ROUTER_PATH=/workspace/checkpoints/router
export OUTPUT_RESPONSE_PATH=/output/format_sampled
export TEMP=0.7
export N_SAMPLES=8
export VLLM_ATTENTION_BACKEND=XFORMERS

gsm8k_test_path=$DATA_DIR/gsm8k_test.parquet
csqa_test_path=$DATA_DIR/csqa_test.parquet

test_files="['$gsm8k_test_path', '$csqa_test_path']"

python3 -u -m verl.trainer.router_guided_generation \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=$N_GPUS \
    data.batch_size=1408 \
    data.path="$test_files" \
    data.prompt_key=prompt \
    data.n_samples=$N_SAMPLES \
    data.output_path=$OUTPUT_RESPONSE_PATH \
    model.path=$MODEL_PATH \
    +model.router_path=$ROUTER_PATH \
    +model.trust_remote_code=True \
    rollout.temperature=$TEMP \
    rollout.top_k=-1 \
    rollout.top_p=1.0 \
    rollout.prompt_length=2048 \
    rollout.response_length=4096 \
    rollout.tensor_model_parallel_size=$TP_SIZE \
    rollout.gpu_memory_utilization=0.8
