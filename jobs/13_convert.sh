#!/bin/bash -l
#SBATCH -J g05_convert
#SBATCH -o /export/home/kaziz/g05/logs/slurm-%x-%j.out
#SBATCH -A h200
#SBATCH --qos h200_qos
#SBATCH -p gpu-H200
#SBATCH --gres gpu:1
#SBATCH -c 4
#SBATCH --mem 32000MB
#SBATCH --time 00:30:00
set -e
source /export/home/kaziz/g05/env_g05.sh
$PY -u $G05_ROOT/jobs/13_convert_gripper.py
ln -sfn $G05_ROOT/data/libero_merged_g05grip $PROJECT_ROOT/data/libero/libero_merged_no_noops
ls -la $PROJECT_ROOT/data/libero/
