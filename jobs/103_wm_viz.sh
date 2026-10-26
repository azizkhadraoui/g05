#!/bin/bash -l
#SBATCH -o /export/home/kaziz/g05/logs/slurm-%x-%j.out
#SBATCH -p gpu-all
#SBATCH --gres gpu:T4_16GB:1
#SBATCH -c 8
#SBATCH --mem 56000MB
#SBATCH --time 02:00:00
# RoboJEPA imagination videos (102_wm_viz_robojepa.py) on real DROID household clips.
export TMPDIR=/export/home/kaziz/g05/tmp PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1
/export/home/kaziz/g05/wm/robo_jepa/.venv/bin/python -u /export/home/kaziz/g05/jobs/102_wm_viz_robojepa.py --per_category ${PER:-2}
