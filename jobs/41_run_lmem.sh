#!/bin/bash -l
#SBATCH -o /export/home/kaziz/g05/logs/slurm-%x-%j.out
#SBATCH -p gpu-all
#SBATCH --gres gpu:T4_16GB:1
#SBATCH -c 16
#SBATCH --mem 100000MB
#SBATCH --time 02:00:00
source /export/home/kaziz/g05/env_g05.sh
source /export/home/kaziz/g05/jobs/lmem_env.sh
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONFAULTHANDLER=1
cd $G05_ROOT/jobs
$PY -u $SCRIPT $ARGS
