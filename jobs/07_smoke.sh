#!/bin/bash -l
#SBATCH -J g05_hsmoke
#SBATCH -o /export/home/kaziz/g05/logs/slurm-%x-%j.out
#SBATCH -A h200
#SBATCH --qos h200_qos
#SBATCH -p gpu-H200
#SBATCH --gres gpu:1
#SBATCH -c 24
#SBATCH --mem 200000MB
#SBATCH --time 01:30:00
# Smoke: prior (40 steps + small held-out eval), then adapter F02 using that prior (40 steps).
source /export/home/kaziz/g05/env_g05.sh
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONFAULTHANDLER=1
S=$G05_ROOT/results/hybrid_smoke; rm -rf $S
$PY -u $G05_ROOT/jobs/07_train_hybrid.py --mode prior --steps 40 --batch 32 --eval_n 128 --save_every 40 --out $S/prior --workers 12 || exit 1
$PY -u $G05_ROOT/jobs/07_train_hybrid.py --mode adapter --branch future --prior_ckpt $S/prior/hybrid_last.pt --steps 40 --batch 32 --save_every 40 --out $S/f02 --workers 12 || exit 1
echo SMOKE_OK
