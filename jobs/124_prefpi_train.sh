#!/bin/bash -l
#SBATCH -o /export/home/kaziz/g05/logs/slurm-%x-%j.out
#SBATCH -p gpu-all
#SBATCH --gres gpu:T4_16GB:1
#SBATCH -c 8
#SBATCH --mem 36000MB
#SBATCH --time 03:00:00
# One PrefPI training step (121_prefpi_train.py) on a T4: the G0.5 env's torch build has no sm70 (V100) kernels.
# Env: ARGS (passed through: --rollouts --obj --init_expert --out ...)
source /export/home/kaziz/g05/env_g05.sh
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# node-local Triton cache (bitsandbytes imports Triton): the shared ~/.triton cache holds helpers compiled on nodes
# with a newer glibc, which fail to load here
export TRITON_CACHE_DIR=/tmp/triton_${SLURM_JOB_ID}; mkdir -p $TRITON_CACHE_DIR; trap 'rm -rf $TRITON_CACHE_DIR' EXIT
echo "node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader) args=${ARGS:?}"
cd $G05_ROOT/jobs && $PY -u 121_prefpi_train.py $ARGS
