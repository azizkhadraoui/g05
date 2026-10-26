#!/bin/bash -l
#SBATCH -o /export/home/kaziz/g05/logs/slurm-%x-%j.out
#SBATCH -p gpu-all
#SBATCH --gres gpu:T4_16GB:1
#SBATCH -c 8
#SBATCH --mem 48000MB
#SBATCH --time 04:00:00
# Cosmos 3 policy actions on the FLUX household windows (116_cosmos_action_eval.py) on one T4 (model CPU offload).
# Env: ARGS (e.g. "--shard 0/3"), MODEL (default Cosmos3-Edge-Policy-DROID), COSMOS_DTYPE (fp16)
export TMPDIR=/export/home/kaziz/g05/tmp HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_ALLOC_CONF=expandable_segments:True
echo "node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader) args=${ARGS:-}"
/export/home/kaziz/g05/wm/cosmos/.venv-diffusers/bin/python -u /export/home/kaziz/g05/jobs/116_cosmos_action_eval.py \
  --model ${MODEL:-Cosmos3-Edge-Policy-DROID} ${ARGS:-}
