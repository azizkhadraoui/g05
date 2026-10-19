#!/bin/bash -l
#SBATCH -J g05_diag
#SBATCH -o /export/home/kaziz/g05/logs/slurm-%x-%j.out
#SBATCH -A h200
#SBATCH --qos h200_qos
#SBATCH -p gpu-H200
#SBATCH --gres gpu:1
#SBATCH -c 16
#SBATCH --mem 128000MB
#SBATCH --time 01:00:00
source /export/home/kaziz/g05/env_g05.sh
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd $G05_ROOT/jobs && $PY -u 12_diag_mismatch.py
