#!/usr/bin/env bash
# Linux / WSL venv setup with CUDA
set -e
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
