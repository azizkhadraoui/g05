#!/bin/bash -l
#SBATCH -o /export/home/kaziz/g05/logs/slurm-%x-%j.out
#SBATCH -p gpu-all
#SBATCH --gres gpu:T4_16GB:1
#SBATCH -c 8
#SBATCH --mem 36000MB
#SBATCH --time 03:00:00
# SuperMimic-style steering of frozen G0.5 (119_smsteer.py): CMD=fit (world model + critic on the recorded window
# transitions) or CMD=improve (noise policy by backpropagation through critic, world model and G0.5's sampler).
# Env: CMD (fit | improve), ARGS (passed through, e.g. --sets ... --out ...). T4: the G0.5 env's torch has no sm70 kernels.
source /export/home/kaziz/g05/env_g05.sh
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader) cmd=${CMD:?} args=${ARGS:-}"
cd $G05_ROOT/jobs && $PY -u 119_smsteer.py $CMD ${ARGS:-}
