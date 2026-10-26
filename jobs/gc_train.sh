#!/bin/bash -l
#SBATCH -o /export/home/kaziz/g05/logs/slurm-%x-%j.out
#SBATCH -p gpu-all
#SBATCH --gres gpu:v100nv_32GB:1
#SBATCH -c 16
#SBATCH --mem 120000MB
#SBATCH --time 10:00:00
# Goal-conditioned adapter (gc_train.py), fp16 path. sbatch [--gres gpu:T4_16GB:1] --export=ALL,TRAIN_ARGS="...",[VENV70=1] gc_train.sh
set -e
source /export/home/kaziz/g05/env_g05.sh
[ -n "${VENV70:-}" ] && PY=$PROJECT_ROOT/.venv_sm70/bin/python
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONFAULTHANDLER=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "node=$(hostname) py=$PY args=$TRAIN_ARGS"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
$PY -u $G05_ROOT/jobs/gc_train.py $TRAIN_ARGS
