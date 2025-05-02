#!/bin/bash
set -e

# Install dependencies
pip install --upgrade pip
pip install wheel
# Pin PyTorch version compatible with pre-built flash-attn wheels and CUDA 12.x
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install datasets==3.2.0 transformers==4.47.1 trl==0.14.0 peft==0.14.0 accelerate==1.2.1 bitsandbytes==0.45.2 wandb==0.19.7
# Install pre-built flash-attn wheel
pip install flash-attn==2.7.4.post1
