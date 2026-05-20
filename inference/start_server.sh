#!/bin/bash
# Start vLLM server for BrReMark model inference

MODEL_PATH=${1:-"path/to/your/checkpoint"}
PORT=${2:-8001}
GPU_IDS=${3:-"0"}
MAX_NUM_SEQS=16

echo "============================================"
echo "  BrReMark vLLM Server"
echo "============================================"
echo "Model: $MODEL_PATH"
echo "Port:  $PORT"
echo "GPU:   $GPU_IDS"
echo ""

CUDA_VISIBLE_DEVICES=$GPU_IDS python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --port $PORT \
    --trust-remote-code \
    --max-num-seqs $MAX_NUM_SEQS \
    --limit-mm-per-prompt image=2 \
    --gpu-memory-utilization 0.9 \
    --dtype auto
