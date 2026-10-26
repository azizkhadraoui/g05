#!/bin/bash -l
#SBATCH -o /export/home/kaziz/g05/logs/slurm-%x-%j.out
#SBATCH -p gpu-all
#SBATCH --gres gpu:T4_16GB:1
#SBATCH -c 8
#SBATCH --mem 56000MB
#SBATCH --time 06:00:00
# Household world-model evaluation (100_wm_eval_robojepa.py) on one GPU. The V-JEPA 2.1 ViT-G checkpoint is 30 GB on
# disk, hence the memory request. Env: SIZES (22m,300m,1b), N_CLIPS (192), STARTS_EVERY (3)
export TMPDIR=/export/home/kaziz/g05/tmp PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1
echo "node=$(hostname) gpu=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader) sizes=${SIZES:-22m,300m,1b}"
/export/home/kaziz/g05/wm/robo_jepa/.venv/bin/python -u /export/home/kaziz/g05/jobs/100_wm_eval_robojepa.py \
  --sizes ${SIZES:-22m,300m,1b} --n_clips ${N_CLIPS:-192} --starts_every ${STARTS_EVERY:-3} --batch_size ${BATCH:-8} ${EXTRA:-}
