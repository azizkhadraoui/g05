#!/bin/bash -l
#SBATCH -o /export/home/kaziz/g05/logs/slurm-%x-%j.out
#SBATCH -p gpu-all
#SBATCH --gres gpu:T4_16GB:1
#SBATCH -c 16
#SBATCH --mem 120000MB
#SBATCH --time 06:00:00
# MEM-1b: extra seeds + per-row correctness for episode-level bootstrap. RUNS="cond:seed[:eval] ..."
source /export/home/kaziz/g05/env_g05.sh
cd $G05_ROOT/jobs
for r in $RUNS; do
  IFS=: read -r cond seed ev <<< "$r"
  echo "=== $cond seed=$seed ${ev:-train} ($(date +%T))"
  $PY -u 29_mem_intent.py --cond $cond --seed $seed ${ev:+--eval_only} || echo "FAILED $r"
done
