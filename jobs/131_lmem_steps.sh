#!/bin/bash -l
#SBATCH -o /export/home/kaziz/g05/logs/slurm-%x-%j.out
#SBATCH -p gpu-all
#SBATCH --gres gpu:1
#SBATCH -c 8
#SBATCH --mem 48000MB
#SBATCH --time 06:00:00
# Step labels + step episodes for the LIBERO-Mem object-identity tasks 6-9 (131_lmem_steps.py; EGL rendering on any GPU, 4 processes).
source /export/home/kaziz/g05/env_g05.sh
source /export/home/kaziz/g05/jobs/lmem_env.sh
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
cd $PROJECT_ROOT
$PY -u $G05_ROOT/jobs/131_lmem_steps.py
