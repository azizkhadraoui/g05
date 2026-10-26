#!/bin/bash -l
#SBATCH -o /export/home/kaziz/g05/logs/slurm-%x-%j.out
#SBATCH -p cpu-all
#SBATCH -c 8
#SBATCH --mem 32000MB
#SBATCH --time 03:00:00
# Env for Cosmos 3 generation with diffusers (Cosmos3OmniPipeline; cookbooks/cosmos3/generator/action/*_diffusers.ipynb).
# NVIDIA's recipe uses torch cu128/cu130 wheels, which drop V100 (sm70); the cu126 build keeps T4 + V100 kernels.
# The guardrail and the LeRobot fork are left out (safety checker disabled; the fork only feeds their plotting cells).
set -e
export TMPDIR=/export/home/kaziz/g05/tmp
source /export/home/kaziz/g05/env_g05.sh; export PATH=$HOME/.local/bin:$PATH
V=/export/home/kaziz/g05/wm/cosmos/.venv-diffusers
uv venv -q -p 3.12 --clear $V
uv pip install -q --python $V/bin/python --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cu126 \
  "torch==2.9.0" "torchvision==0.24.0" "diffusers @ git+https://github.com/huggingface/diffusers.git" transformers accelerate \
  imageio imageio-ffmpeg av pandas pyarrow scipy matplotlib loguru huggingface_hub safetensors sentencepiece 2>&1 | tail -5
$V/bin/python -c "import torch, diffusers, transformers; from diffusers import Cosmos3OmniPipeline, CosmosActionCondition; print('cosmos ok', torch.__version__, torch.version.cuda, diffusers.__version__, transformers.__version__)"
echo ALL_DONE
