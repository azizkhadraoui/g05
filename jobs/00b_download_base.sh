#!/bin/bash -l
#SBATCH -J g05_dl_base
#SBATCH -o /export/home/kaziz/g05/logs/slurm-%x-%j.out
#SBATCH -p cpu-all
#SBATCH -c 4
#SBATCH --mem 16000MB
#SBATCH --time 08:00:00
# g05-base only (00_download.sh's multi-pattern --include was parsed as filenames by hf 1.31 and skipped it).
source /export/home/kaziz/g05/env_g05.sh
export HF_XET_HIGH_PERFORMANCE=1
HF=/export/home/kaziz/motion/miniconda3/envs/ardy/bin/hf
echo "node: $(hostname) $(date +%T)"
$HF download OpenGalaxea/G05 --include "g05-base/*" --local-dir $PROJECT_ROOT/checkpoints; rc=$?
echo "rc=$rc $(date +%T)"; ls -la $PROJECT_ROOT/checkpoints/g05-base/checkpoints/ $PROJECT_ROOT/checkpoints/g05-base/
exit $rc
