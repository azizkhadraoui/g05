#!/bin/bash -l
#SBATCH -o /export/home/kaziz/g05/logs/slurm-%x-%j.out
#SBATCH -A h200
#SBATCH --qos h200_qos
#SBATCH -p gpu-H200
#SBATCH --gres gpu:1
#SBATCH -c 16
#SBATCH --mem 160000MB
#SBATCH --time 06:00:00
# Generic: sbatch -J name --export=ALL,SCRIPT=xx.py,ARGS="..." run_py_gpu.sh
set -e
source /export/home/kaziz/g05/env_g05.sh
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONFAULTHANDLER=1
cd $G05_ROOT/jobs
echo "node=$(hostname) script=$SCRIPT args=$ARGS"
$PY -u $SCRIPT $ARGS
