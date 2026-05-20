#!/bin/bash
# BrReMark - Stage 2: GRPO Reinforcement Learning
# Two phases:
#   Phase 1 (real data): r_fmt(0.4) + r_loc(0.5) + r_llm(1.0), rollout_n=6, 3x A100
#   Phase 2 (synthetic): r_fmt(0.4) + r_loc(0.5) only (r_llm masked), rollout_n=6, 4x A100
#
# Prerequisites:
#   Phase 1 requires Lingshu-32B as LLM-as-judge. Start it in a separate terminal:
#     CUDA_VISIBLE_DEVICES=3 python -m vllm.entrypoints.openai.api_server \
#         --model <path/to/Lingshu-32B> --port 8000 --trust-remote-code --dtype auto
#   Then configure .env:
#     LLM_JUDGE_MODEL=lingshu-judge-vllm
#     OPENAI_BASE_URL=http://127.0.0.1:8000/v1
#     OPENAI_API_KEY=EMPTY
#
# Usage:
#   bash scripts/train_rl.sh                    # Run both phases
#   bash scripts/train_rl.sh --phase 1          # Real data only (paper: "w/o synthesis")
#   bash scripts/train_rl.sh --phase 2          # Synthetic only (assumes phase 1 done)

set -x

source /home/lsk_22307130255/miniconda3/etc/profile.d/conda.sh
conda activate brremark

if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

export RAY_DISABLE_DOCKER_CPU_WARNING=1
export WANDB_MODE=offline
export WANDB_DIR=./wandb

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

export SAVE_CHECKPOINT_DIR=./checkpoint

# === Configure these paths ===
SFT_CHECKPOINT="<SFT_CHECKPOINT_PATH>"              # e.g., ./checkpoints/sft_brremark/global_step_3360
RL_DATA_FILE="<RL_DATA_PATH>"                       # e.g., ./data/rl_train.parquet
RL_VAL_FILE="${RL_VAL_FILE:-$RL_DATA_FILE}"
SYNTHETIC_DATA_FILE="<SYNTHETIC_DATA_PATH>"         # e.g., ./data/synthetic_brain_mri.parquet

# Parse arguments
PHASE="both"
while [[ $# -gt 0 ]]; do
    case $1 in
        --phase) PHASE="$2"; shift 2 ;;
        *) shift ;;
    esac
done

mkdir -p ./logs

# ============================================================
# Phase 1: GRPO on Real Data
# ============================================================
run_phase1() {
    echo "======================================"
    echo "Phase 1: GRPO on Real Data"
    echo "======================================"

    export CUDA_VISIBLE_DEVICES=0,1,2
    ray stop --force 2>/dev/null || true

    local PROJECT_NAME="BrReMark"
    local EXPERIMENT_NAME="rl_after_sft"
    local REF_MODEL_PATH="$SFT_CHECKPOINT"

    if [ ! -f "$RL_DATA_FILE" ]; then
        echo "ERROR: RL data not found: $RL_DATA_FILE"; exit 1
    fi
    if [ ! -d "$REF_MODEL_PATH" ]; then
        echo "ERROR: SFT checkpoint not found: $REF_MODEL_PATH"; exit 1
    fi

    PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
        +debug=False \
        +vs_debug=False \
        data.train_files=["${RL_DATA_FILE}"] \
        data.val_files=["${RL_VAL_FILE}"] \
        data.train_batch_size=96 \
        data.max_prompt_length=4096 \
        data.max_response_length=2048 \
        data.return_raw_chat=True \
        data.filter_overlong_prompts=False \
        data.reward_fn_key=data_source \
        algorithm.adv_estimator=grpo \
        algorithm.kl_ctrl.kl_coef=0.001 \
        actor_rollout_ref.model.path=${REF_MODEL_PATH} \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.optim.lr=1.5e-6 \
        actor_rollout_ref.actor.optim.warmup_style=cosine \
        actor_rollout_ref.actor.optim.total_training_steps=32082 \
        actor_rollout_ref.actor.optim.lr_warmup_steps=21 \
        actor_rollout_ref.actor.ppo_mini_batch_size=96 \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
        actor_rollout_ref.actor.use_kl_loss=True \
        actor_rollout_ref.actor.kl_loss_coef=0.001 \
        actor_rollout_ref.actor.kl_loss_type=low_var_kl \
        actor_rollout_ref.actor.entropy_coeff=0.0 \
        actor_rollout_ref.actor.checkpoint.contents=['model','hf_model','optimizer','extra'] \
        actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
        actor_rollout_ref.actor.fsdp_config.param_offload=True \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.n=6 \
        actor_rollout_ref.rollout.prompt_length=4096 \
        actor_rollout_ref.rollout.max_num_batched_tokens=262144 \
        actor_rollout_ref.rollout.max_model_len=32768 \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.80 \
        actor_rollout_ref.rollout.enforce_eager=False \
        actor_rollout_ref.rollout.free_cache_engine=False \
        actor_rollout_ref.rollout.enable_chunked_prefill=False \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
        actor_rollout_ref.ref.fsdp_config.param_offload=False \
        actor_rollout_ref.rollout.agent.activate_agent=True \
        actor_rollout_ref.rollout.agent.tool_name_key=env_name \
        actor_rollout_ref.rollout.agent.single_response_max_tokens=1024 \
        actor_rollout_ref.rollout.agent.max_turns=2 \
        actor_rollout_ref.rollout.agent.concurrent_workers=32 \
        actor_rollout_ref.rollout.agent.show_tqdm=True \
        actor_rollout_ref.rollout.agent.max_vllm_images=2 \
        actor_rollout_ref.rollout.agent.max_vllm_videos=0 \
        trainer.critic_warmup=0 \
        trainer.logger=['console','wandb','tensorboard'] \
        trainer.val_before_train=False \
        trainer.n_gpus_per_node=3 \
        trainer.nnodes=1 \
        trainer.save_freq=50 \
        trainer.test_freq=10000 \
        trainer.project_name=${PROJECT_NAME} \
        trainer.experiment_name=${EXPERIMENT_NAME} \
        trainer.default_local_dir=${SAVE_CHECKPOINT_DIR}/${PROJECT_NAME}/${EXPERIMENT_NAME} \
        +trainer.tensorboard_dir=${SAVE_CHECKPOINT_DIR}/logs/tensorboard \
        +trainer.rl_logging_board_dir=${SAVE_CHECKPOINT_DIR}/logs/rl_logging_board \
        trainer.total_epochs=2 2>&1 | tee ./logs/${EXPERIMENT_NAME}.log

    echo "Phase 1 completed!"

    # Merge FSDP checkpoint to HuggingFace format for Phase 2
    local LAST_CKPT=$(ls -d ${SAVE_CHECKPOINT_DIR}/${PROJECT_NAME}/${EXPERIMENT_NAME}/global_step_*/actor 2>/dev/null | sort -t_ -k2 -n | tail -1)
    if [ -n "$LAST_CKPT" ]; then
        echo "Merging checkpoint: $LAST_CKPT"
        python scripts/model_merger.py --local_dir "$LAST_CKPT"
    fi
}

# ============================================================
# Phase 2: GRPO on Synthetic Data (spatial targeting)
# ============================================================
run_phase2() {
    echo "======================================"
    echo "Phase 2: GRPO on Synthetic Data"
    echo "======================================"

    export CUDA_VISIBLE_DEVICES=0,1,2,3
    ray stop --force 2>/dev/null || true

    local PROJECT_NAME="BrReMark"
    local EXPERIMENT_NAME="rl_synthetic"

    # Use merged HF checkpoint from Phase 1
    local LAST_HF=$(ls -d ${SAVE_CHECKPOINT_DIR}/${PROJECT_NAME}/rl_after_sft/global_step_*/actor/huggingface 2>/dev/null | sort -t_ -k2 -n | tail -1)
    if [ -z "$LAST_HF" ]; then
        echo "ERROR: No merged HF checkpoint found from Phase 1."
        echo "Run Phase 1 first, or set REF_MODEL_PATH manually."
        exit 1
    fi
    local REF_MODEL_PATH="$LAST_HF"

    if [ ! -f "$SYNTHETIC_DATA_FILE" ]; then
        echo "ERROR: Synthetic data not found: $SYNTHETIC_DATA_FILE"; exit 1
    fi

    echo "Base Model: $REF_MODEL_PATH"
    echo "Data:       $SYNTHETIC_DATA_FILE"
    echo "Reward:     format(0.4) + IoU(0.5) only"

    PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
        +debug=False \
        +vs_debug=False \
        data.train_files=["${SYNTHETIC_DATA_FILE}"] \
        data.val_files=["${SYNTHETIC_DATA_FILE}"] \
        data.train_batch_size=64 \
        data.max_prompt_length=4096 \
        data.max_response_length=2048 \
        data.return_raw_chat=True \
        data.filter_overlong_prompts=False \
        data.reward_fn_key=data_source \
        algorithm.adv_estimator=grpo \
        algorithm.kl_ctrl.kl_coef=0.001 \
        actor_rollout_ref.model.path=${REF_MODEL_PATH} \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.optim.lr=1.5e-6 \
        actor_rollout_ref.actor.optim.warmup_style=cosine \
        actor_rollout_ref.actor.optim.total_training_steps=45 \
        actor_rollout_ref.actor.optim.lr_warmup_steps=5 \
        actor_rollout_ref.actor.ppo_mini_batch_size=64 \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
        actor_rollout_ref.actor.use_kl_loss=True \
        actor_rollout_ref.actor.kl_loss_coef=0.001 \
        actor_rollout_ref.actor.kl_loss_type=low_var_kl \
        actor_rollout_ref.actor.entropy_coeff=0.0 \
        actor_rollout_ref.actor.checkpoint.contents=['model','hf_model','optimizer','extra'] \
        actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
        actor_rollout_ref.actor.fsdp_config.param_offload=True \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.n=6 \
        actor_rollout_ref.rollout.prompt_length=4096 \
        actor_rollout_ref.rollout.max_num_batched_tokens=262144 \
        actor_rollout_ref.rollout.max_model_len=32768 \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.80 \
        actor_rollout_ref.rollout.enforce_eager=False \
        actor_rollout_ref.rollout.free_cache_engine=False \
        actor_rollout_ref.rollout.enable_chunked_prefill=False \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
        actor_rollout_ref.ref.fsdp_config.param_offload=False \
        actor_rollout_ref.rollout.agent.activate_agent=True \
        actor_rollout_ref.rollout.agent.tool_name_key=env_name \
        actor_rollout_ref.rollout.agent.single_response_max_tokens=1024 \
        actor_rollout_ref.rollout.agent.max_turns=2 \
        actor_rollout_ref.rollout.agent.concurrent_workers=32 \
        actor_rollout_ref.rollout.agent.show_tqdm=True \
        actor_rollout_ref.rollout.agent.max_vllm_images=2 \
        actor_rollout_ref.rollout.agent.max_vllm_videos=0 \
        trainer.critic_warmup=0 \
        trainer.logger=['console','wandb','tensorboard'] \
        trainer.val_before_train=False \
        trainer.n_gpus_per_node=4 \
        trainer.nnodes=1 \
        trainer.save_freq=30 \
        trainer.test_freq=10000 \
        trainer.project_name=${PROJECT_NAME} \
        trainer.experiment_name=${EXPERIMENT_NAME} \
        trainer.default_local_dir=${SAVE_CHECKPOINT_DIR}/${PROJECT_NAME}/${EXPERIMENT_NAME} \
        +trainer.tensorboard_dir=${SAVE_CHECKPOINT_DIR}/logs/tensorboard \
        +trainer.rl_logging_board_dir=${SAVE_CHECKPOINT_DIR}/logs/rl_logging_board \
        trainer.total_epochs=3 2>&1 | tee ./logs/${EXPERIMENT_NAME}.log

    echo "Phase 2 (Synthetic) completed!"
}

# ============================================================
# Main
# ============================================================
case "$PHASE" in
    1)    run_phase1 ;;
    2)    run_phase2 ;;
    both) run_phase1 && run_phase2 ;;
    *)    echo "Usage: $0 [--phase 1|2|both]"; exit 1 ;;
esac
