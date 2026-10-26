#!/bin/bash -l
#SBATCH -o /export/home/kaziz/g05/logs/slurm-%x-%j.out
#SBATCH -p gpu-all
#SBATCH --gres gpu:T4_16GB:1
#SBATCH -c 8
#SBATCH --mem 48000MB
#SBATCH --time 06:00:00
# Cosmos 3 Edge-Policy-DROID videos on real DROID household episodes (107_cosmos_policy_viz.py).
# Env: COSMOS_DTYPE (fp16), CHUNKS (5), PER (2), MODEL (Cosmos3-Edge-Policy-DROID)
export TMPDIR=/export/home/kaziz/g05/tmp HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True COSMOS_DTYPE=${COSMOS_DTYPE:-fp16}
echo "node=$(hostname) gpu=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader) dtype=$COSMOS_DTYPE"
/export/home/kaziz/g05/wm/cosmos/.venv-diffusers/bin/python -u /export/home/kaziz/g05/jobs/107_cosmos_policy_viz.py \
  --chunks ${CHUNKS:-5} --per_category ${PER:-2} --anchor ${ANCHOR:-chain} ${EPISODES:+--episodes $EPISODES} ${EXAMPLE:+--example} --fps ${FPS:-5} --steps ${STEPS:-30} --guidance ${GUID:-1.0}
