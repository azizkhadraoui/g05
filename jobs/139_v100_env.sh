#!/bin/bash -l
#SBATCH -o /export/home/kaziz/g05/logs/slurm-%x-%j.out
#SBATCH -p gpu-all
#SBATCH --gres gpu:v100_16GB:1
#SBATCH -c 8
#SBATCH --mem 32000MB
#SBATCH --time 00:40:00
# Make the V100 twin of the G0.5 env (.venv_sm70, 39_venv_sm70.sh) match .venv: the cu126 torch reinstall pulled newer
# numpy / pillow / filelock / fsspec / typing_extensions / setuptools than .venv has. Then a smoke test on the V100:
# kernels, fp16 matmul, LIBERO + LIBERO-Mem rendering (EGL), VLM loader import (139_v100_smoke.py).
set -e
source /export/home/kaziz/g05/env_g05.sh
echo "node=$(hostname) py=$PY"; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
uv pip install --python $PROJECT_ROOT/.venv_sm70/bin/python --no-deps numpy==1.26.4 pillow==10.2.0 filelock==3.20.0 \
    fsspec==2025.3.0 typing_extensions==4.15.0 setuptools==80.9.0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd $G05_ROOT/jobs && $PY -u 139_v100_smoke.py libero_goal
source $G05_ROOT/jobs/lmem_env.sh && $PY -u 139_v100_smoke.py libero_mem
