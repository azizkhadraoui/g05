#!/bin/bash -l
#SBATCH -o /export/home/kaziz/g05/logs/slurm-%x-%j.out
#SBATCH -p gpu-all
#SBATCH --gres gpu:T4_16GB:1
#SBATCH -c 8
#SBATCH --mem 48000MB
#SBATCH --time 01:30:00
# Train the DSBC noise policy (109_dsbc_train.py) on the recorded nudge episodes. Env: EXTRA (e.g. --all)
source /export/home/kaziz/g05/env_g05.sh
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 HF_HUB_OFFLINE=1
cd $G05_ROOT/jobs && $PY -u 109_dsbc_train.py --out ${OUT_CKPT:-$G05_ROOT/results/steer_frs/dsbc/policy.pt} ${EXTRA:-}
