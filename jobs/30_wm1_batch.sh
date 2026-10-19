#!/bin/bash -l
#SBATCH -o /export/home/kaziz/g05/logs/slurm-%x-%j.out


#SBATCH -p gpu-all
#SBATCH --gres gpu:T4_16GB:1
#SBATCH -c 16
#SBATCH --mem 120000MB
#SBATCH --time 08:00:00
# WM-1: several world-model runs back to back on one H200. RUNS="name|args;name|args;..."
source /export/home/kaziz/g05/env_g05.sh
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd $G05_ROOT/jobs
R=$G05_ROOT/results
IFS=';' read -ra LIST <<< "$RUNS"
for item in "${LIST[@]}"; do
  n=${item%%|*}; x=${item#*|}
  echo "=== $n: $x ($(date +%T))"
  $PY -u 23_train_wm.py $x --out $R/wm_$n || echo "FAILED $n"
done
