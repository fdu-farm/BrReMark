#!/bin/bash
# BrReMark - Stage 1: Supervised Fine-Tuning
# Train Lingshu-7B on two-turn reasoning trajectories (think→mark_bbox→rethink→answer)
#
# Hyperparameters (from paper):
#   lr=3e-6, batch=8, epochs=8, 4x A100 80GB, cosine schedule with 10% warmup

set -x

source /home/lsk_22307130255/miniconda3/etc/profile.d/conda.sh
conda activate brremark

if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

export CUDA_VISIBLE_DEVICES=0,1,2,3
export TOKENIZERS_PARALLELISM=true
export NCCL_DEBUG=WARN
export PYTHONUNBUFFERED=1

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SAVE_DIR="$PROJECT_ROOT/checkpoints/sft_brremark"

# === Configure these paths ===
MODEL_PATH="<BASE_MODEL_PATH>"          # e.g., lingshu-medical-mllm/Lingshu-7B or local path
TRAIN_FILE="<SFT_DATA_PATH>"            # e.g., ./data/sft_train.parquet
VAL_FILE="${VAL_FILE:-$TRAIN_FILE}"

mkdir -p "$SAVE_DIR" "$PROJECT_ROOT/logs"

echo "======================================"
echo "BrReMark SFT Training"
echo "======================================"
echo "Model: $MODEL_PATH"
echo "Data:  $TRAIN_FILE"
echo "Save:  $SAVE_DIR"
echo "======================================"

if [ ! -f "$TRAIN_FILE" ]; then
    echo "ERROR: Training data not found: $TRAIN_FILE"
    echo "See docs/DATA.md for data preparation instructions."
    exit 1
fi

cd "$PROJECT_ROOT"

python -m torch.distributed.run \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=4 \
    -m verl.trainer.fsdp_sft_trainer \
    --config-name sft_brremark \
    data.train_files=$TRAIN_FILE \
    data.val_files=$VAL_FILE \
    data.train_batch_size=8 \
    data.micro_batch_size_per_gpu=2 \
    optim.lr=3e-6 \
    optim.warmup_steps_ratio=0.1 \
    optim.lr_scheduler=cosine \
    trainer.default_local_dir=$SAVE_DIR \
    trainer.default_hdfs_dir=null \
    trainer.total_epochs=8 \
    model.partial_pretrain=$MODEL_PATH \
    trainer.project_name=BrReMark-sft \
    trainer.experiment_name=sft_lingshu \
    trainer.logger="['console','wandb']" \
    "$@" 2>&1 | tee "$PROJECT_ROOT/logs/sft_train.log"

echo "SFT Training completed! Checkpoints saved to: $SAVE_DIR"
