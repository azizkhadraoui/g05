#!/bin/bash -l
#SBATCH -o /export/home/kaziz/g05/logs/slurm-%x-%j.out
#SBATCH -p gpu-all
#SBATCH --gres gpu:T4_16GB:1
#SBATCH -c 8
#SBATCH --mem 56000MB
#SBATCH --time 05:00:00
# FLUX 3 Action on the DROID household clips (112_flux_action_eval.py) on one T4. Uses the first complete BF16 package:
# the root (base recipe: 4 steps, CFG 4) if present, else the guidance-distilled, else the step-distilled one; waits
# up to 3 h for one to finish downloading. Env: SUBFOLDER (force a package), ARGS (extra arguments), RESIDENT_GB
export TMPDIR=/export/home/kaziz/g05/tmp PYTORCH_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
C=/export/home/kaziz/g05/wm/checkpoints/flux-3-action-droid
for i in $(seq 1 180); do
  if [ -n "${SUBFOLDER+x}" ]; then [ -f $C/$SUBFOLDER/model.safetensors ] && break
  else for s in "" variants/gd variants/sd; do [ -f $C/$s/model.safetensors ] && { SUBFOLDER=$s; break 2; }; done; fi
  sleep 60
done
echo "node=$(hostname) gpu=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader) package=${SUBFOLDER:-root}"
[ -f $C/${SUBFOLDER:-}/model.safetensors ] || { echo "no BF16 package downloaded"; exit 1; }
/export/home/kaziz/g05/wm/flux-action/.venv/bin/python -u /export/home/kaziz/g05/jobs/112_flux_action_eval.py \
  --subfolder "${SUBFOLDER:-}" ${ARGS:-}
