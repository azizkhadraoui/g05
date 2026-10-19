#!/bin/bash -l
#SBATCH -J g05_lawm_drift
#SBATCH -o /export/home/kaziz/g05/logs/slurm-%x-%j.out
#SBATCH -p cpu-all
#SBATCH -c 4
#SBATCH --mem 48000MB
#SBATCH --time 00:30:00
set -e
source /export/home/kaziz/g05/env_g05.sh
cd $G05_ROOT
$PY -u jobs/05b_lawm_drift.py
