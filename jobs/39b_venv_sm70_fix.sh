#!/bin/bash -l
#SBATCH -o /export/home/kaziz/g05/logs/slurm-%x-%j.out
#SBATCH -p cpu-all
#SBATCH -c 8
#SBATCH --mem 32000MB
#SBATCH --time 01:00:00
# Swap torch in .venv_sm70 for the cu126 build (has sm_70). Run outside the project dir with --no-config so the
# project's uv index pin does not force the cu128 wheel again.
export PATH=$HOME/.local/bin:$PATH
cd /tmp
uv pip install --no-config --python /export/home/kaziz/g05/GalaxeaVLA/.venv_sm70/bin/python \
  --index-url https://download.pytorch.org/whl/cu126 --reinstall-package torch --reinstall-package torchvision --reinstall-package torchaudio \
  "torch==2.7.1+cu126" "torchvision==0.22.1+cu126" "torchaudio==2.7.1+cu126" && echo installed
/export/home/kaziz/g05/GalaxeaVLA/.venv_sm70/bin/python -c "import torch; print(torch.__version__, torch.version.cuda)"
