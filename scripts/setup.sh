#!/bin/bash
# BrReMark Setup Script
# Creates conda environment and installs all dependencies

set -e

ENV_NAME="brremark"
PYTHON_VERSION="3.11"

echo "======================================"
echo "BrReMark Environment Setup"
echo "======================================"

# Create conda environment
if conda info --envs | grep -q "^${ENV_NAME} "; then
    echo "Environment '${ENV_NAME}' already exists. Activating..."
else
    echo "Creating conda environment: ${ENV_NAME} (Python ${PYTHON_VERSION})"
    conda create -n ${ENV_NAME} python=${PYTHON_VERSION} -y
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ${ENV_NAME}

# Install PyTorch (CUDA 12.4)
echo "Installing PyTorch..."
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# Install flash-attn (requires torch to be installed first)
echo "Installing flash-attn..."
pip install flash-attn --no-build-isolation

# Install core dependencies
echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

# Install verl in editable mode
echo "Installing verl (editable)..."
pip install -e .

echo ""
echo "======================================"
echo "Setup complete!"
echo "======================================"
echo ""
echo "Activate the environment:"
echo "  conda activate ${ENV_NAME}"
echo ""
echo "Next steps:"
echo "  1. Prepare data (see docs/DATA.md)"
echo "  2. Run SFT:  bash scripts/train_sft.sh"
echo "  3. Run RL:   bash scripts/train_rl.sh"
