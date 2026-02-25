#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Configuration
ENV_NAME="GR00T"
REPO_DIR="/home/ubuntu/MT_of_VLAs"

# 1. Initialize Conda for this shell session
# This allows 'conda activate' to work inside the script
CONDA_PATH=$(conda info --base)
source "$CONDA_PATH/etc/profile.d/conda.sh"
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true
echo "--- Creating Conda Environment: $ENV_NAME ---"
conda create -n "$ENV_NAME" python=3.10 -y
conda activate "$ENV_NAME"

echo "--- Installing repo ---"
cd "$REPO_DIR"
# Check if EO1 already exists before cloning
cd "Isaac-GR00T"
pip install "setuptools==80.10.2"
pip install -e .
# Flash-attn can take a long time to compile

echo "--- Fixing Numpy version for Pinocchio compatibility ---"
pip install numpy==1.24.4

echo "--- Installing ManiSkill2 real2sim ---"
cd "$REPO_DIR/ManiSkill2_real2sim"
pip install -e .

echo "--- Installing main package ---"
cd "$REPO_DIR"
pip install -e .
pip install "numpy==1.24.4"
pip install matplotlib
echo "--- Downloading Models ---"
cd "$REPO_DIR/checkpoints"
python download_model.py nvidia/GR00T-N1.6-bridge
python download_model.py nvidia/GR00T-N1.6-fractal
echo "--- Setup Complete! ---"
echo "To start using the environment, run: conda activate $ENV_NAME"
