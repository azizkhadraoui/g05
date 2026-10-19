#!/bin/bash -l
#SBATCH -J g05_train
#SBATCH -o /export/home/kaziz/g05/logs/slurm-%x-%j.out
#SBATCH -A h200
#SBATCH --qos h200_qos
#SBATCH -p gpu-H200
#SBATCH --gres gpu:1
#SBATCH -c 24
#SBATCH --mem 200000MB
#SBATCH --time 10:00:00
# Usage: sbatch -J g05_<name> --export=ALL,TRAIN_ARGS="--mode adapter --branch current --out ..." 07_train_hybrid.sh
set -e
source /export/home/kaziz/g05/env_g05.sh
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONFAULTHANDLER=1
echo "node=$(hostname) args=$TRAIN_ARGS"
$PY -u $G05_ROOT/jobs/07_train_hybrid.py $TRAIN_ARGS
