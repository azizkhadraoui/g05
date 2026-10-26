#!/bin/bash -l
#SBATCH -o /export/home/kaziz/g05/logs/slurm-%x-%j.out
#SBATCH -p gpu-all
#SBATCH --gres gpu:T4_16GB:1
#SBATCH -c 8
#SBATCH --mem 64000MB
#SBATCH --time 03:00:00
# Level-2 memory modules: next-step-instruction decoders for LIBERO-Mem tasks 6-9 (46_lmem_count.py on 131's labels),
# from 8 frames over the episode + now (memory) and from the current frame only (Markovian control).
source /export/home/kaziz/g05/env_g05.sh
cd $PROJECT_ROOT
F=$G05_ROOT/results/lmem_feats
for cond in uniform8 current; do
    $PY -u $G05_ROOT/jobs/46_lmem_count.py --cond $cond --labels $F/labels_instr.npy --tasks 6,7,8,9 --n_classes 22 \
        --classes $F/labels_instr_classes.json --tag instr_$cond
done
