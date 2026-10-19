#!/bin/bash -l
#SBATCH -o /export/home/kaziz/g05/logs/slurm-%x-%j.out
#SBATCH -p gpu-all
#SBATCH --gres gpu:v100nv_32GB:1
#SBATCH -c 8
#SBATCH --mem 120000MB
#SBATCH --time 08:00:00
set -e
source /export/home/kaziz/g05/env_g05.sh
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd $G05_ROOT/jobs
echo "node=$(hostname)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
PY=$PROJECT_ROOT/.venv_sm70/bin/python
$PY -u $SCRIPT $ARGS
