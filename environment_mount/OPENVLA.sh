#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Configuration
ENV_NAME="OPENVLA"
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




echo "--- Installing LeRobot and EO1 ---"
pip install numpy==1.24.4
echo "--- Installing ManiSkill2 real2sim ---"
cd "$REPO_DIR/ManiSkill2_real2sim"
pip install -e .

echo "--- Installing main package ---"
cd "$REPO_DIR"
pip install -e .
apt install ffmpeg

pip install tensorflow==2.15.0
pip install -r requirements_full_install.txt
pip install tensorflow[and-cuda]==2.15.1
pip install git+https://github.com/nathanrooy/simulated-annealing
pip install torch==2.3.1 torchvision==0.18.1 timm==0.9.10 tokenizers==0.15.2 accelerate==0.32.1
pip install flash-attn==2.6.1 --no-build-isolation

echo "--- Installing transformers ---"
cd "$REPO_DIR/transformers-4.40.1"
pip install -e .

pip install "numpy==1.24.4"
pip install matplotlib
pip install openpyxl
pip install seaborn
pip install frechetdist
pip install matplotlib_venn
echo "--- Downloading models ---"
cd "$REPO_DIR/checkpoints"
python download_model.py openvla/openvla-7b
cp modeling_prismatic.py openvla-7b

echo "--- Setup Complete! ---"
echo "To start using the environment, run: conda activate $ENV_NAME"
