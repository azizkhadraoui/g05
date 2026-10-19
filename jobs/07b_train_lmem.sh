#!/bin/bash -l
#SBATCH -o /export/home/kaziz/g05/logs/slurm-%x-%j.out
#SBATCH -p gpu-all
#SBATCH -c 16
#SBATCH --mem 120000MB
#SBATCH --time 10:00:00
# fp16-capable trainer (T4 / V100 / H200). sbatch --gres gpu:<type>:1 --export=ALL,TRAIN_ARGS="...",[VENV70=1] 07b_train_lmem.sh
set -e
source /export/home/kaziz/g05/env_g05.sh
[ -n "${VENV70:-}" ] && PY=$PROJECT_ROOT/.venv_sm70/bin/python
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONFAULTHANDLER=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "node=$(hostname) py=$PY args=$TRAIN_ARGS"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
$PY -u $G05_ROOT/jobs/07b_train_lmem.py $TRAIN_ARGS
