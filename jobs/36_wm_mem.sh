#!/bin/bash -l
#SBATCH -o /export/home/kaziz/g05/logs/slurm-%x-%j.out
#SBATCH -p gpu-all
#SBATCH --gres gpu:T4_16GB:1
#SBATCH -c 16
#SBATCH --mem 120000MB
#SBATCH --time 06:00:00
# WM-MEM: world model with episodic memory vs Markovian control (35_train_wm_mem.py). NAME, HIST from the environment.
source /export/home/kaziz/g05/env_g05.sh
cd $G05_ROOT/jobs
echo "=== $NAME hist=$HIST ($(date +%T))"
$PY -u 35_train_wm_mem.py --hist "$HIST" --out $G05_ROOT/results/wmmem_$NAME ${EXTRA:-}
