#!/bin/bash -l
#SBATCH -J g05_lawm_chk
#SBATCH -o /export/home/kaziz/g05/logs/slurm-%x-%j.out
#SBATCH -A h200
#SBATCH --qos h200_qos
#SBATCH -p gpu-H200
#SBATCH --gres gpu:1
#SBATCH -c 8
#SBATCH --mem 96000MB
#SBATCH --time 02:00:00
# Phase 0 step 4 (go/no-go): pretrained LaWM vs copy baseline on LIBERO frames.
set -e
source /export/home/kaziz/g05/env_g05.sh
cd $G05_ROOT
env TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 $PY -u jobs/05_lawm_sanity.py
