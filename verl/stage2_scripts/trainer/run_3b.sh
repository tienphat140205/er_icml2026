export N_GPUS=4
export BASE_MODEL='arm-team/ARM-3B'
export BASE_ROUTER='tp140205/arm-router-base'
export DATA_DIR='verl/data/parquet'
export ROLLOUT_TP_SIZE=1
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Hyperparameters (can be overridden by env vars)
export ROUTER_BETA_MAX=${ROUTER_BETA_MAX:-1.0}
export ROUTER_BETA_MIN=${ROUTER_BETA_MIN:-0.5}
export ROUTER_RHO=${ROUTER_RHO:-0.7}
export ROUTER_LR=${ROUTER_LR:-5e-5}

# Auto-generated experiment name
export EXPERIMENT_NAME="arm-3B-${N_GPUS}gpu-bmax${ROUTER_BETA_MAX}-bmin${ROUTER_BETA_MIN}-r${ROUTER_RHO}-lr${ROUTER_LR}"

export VLLM_ATTENTION_BACKEND=XFORMERS
export WANDB_API_KEY='ac6e358c9e02e44fdccee1c0c68e4a7ea095bafb'
export WANDB_MODE=offline
export WANDB_CONSOLE=wrap
export WANDB_PROJECT='ER'
export WANDB_ENTITY='ER_ver2'
export HF_TOKEN='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'
export RAY_memory_usage_threshold=0.99
gsm8k_train_path=$DATA_DIR/gsm8k_train.parquet
gsm8k_test_path=$DATA_DIR/gsm8k_test.parquet
csqa_train_path=$DATA_DIR/csqa_train.parquet
csqa_test_path=$DATA_DIR/csqa_test.parquet
MATH_train_path=$DATA_DIR/MATH_train.parquet
MATH_test_path=$DATA_DIR/MATH_test.parquet


train_files="['$gsm8k_train_path', '$MATH_train_path', '$csqa_train_path']"
val_files="['$gsm8k_test_path', '$MATH_test_path', '$csqa_test_path']"

# Build test file list (Full Benchmark)
SVAMP_test_path=$DATA_DIR/SVAMP_test.parquet
openbookqa_test_path=$DATA_DIR/openbookqa_test.parquet
AIME2025_test_path=$DATA_DIR/AIME2025_test.parquet
test_files="['$gsm8k_test_path', '$MATH_test_path', '$csqa_test_path', '$SVAMP_test_path', '$openbookqa_test_path', '$AIME2025_test_path'"
for bbh_file in $DATA_DIR/BBH/BBH_*_test.parquet; do
    if [ -f "$bbh_file" ]; then
        test_files+=", '$bbh_file'"
    fi
done
test_files+="]"


python3 -m verl.trainer.my_ppo \
    algorithm.adv_estimator=router_grpo \
    data.train_files="$train_files" \
    data.val_files="$val_files" \
    +data.test_files="$test_files" \
    data.train_batch_size=1024 \
    data.val_batch_size=1312 \
    data.max_prompt_length=2048 \
    data.max_response_length=4096 \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=180 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=30 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=k3+ \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.strategy=fsdp \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.grad_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=120 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP_SIZE \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=120 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    +algorithm.router_beta_max=$ROUTER_BETA_MAX \
    +algorithm.router_beta_min=$ROUTER_BETA_MIN \
    algorithm.router_rho=$ROUTER_RHO \
    +algorithm.format_reward_agg=mean \
    data.return_raw_chat=True \
    router.model.path=$BASE_ROUTER \
    router.model.fsdp_config.param_offload=True \
    router.model.fsdp_config.grad_offload=True \
    router.model.fsdp_config.optimizer_offload=True \
    router.optim.lr=$ROUTER_LR \
    router.forward_micro_batch_size_per_gpu=8 \
    router.router_mini_batch_size=256 \
    router.router_micro_batch_size_per_gpu=8 \
    router.grad_clip=1.0 \
    router.router_kl_loss_type=k3+ \
    +trainer.val_before_train=True \
    trainer.critic_warmup=0 \
    trainer.logger="['wandb', 'console']" \
    trainer.project_name=$WANDB_PROJECT \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=25 \
    trainer.test_freq=25 \
    trainer.total_epochs=3 \
    $@

