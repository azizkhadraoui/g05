#!/bin/bash -l
#SBATCH -J g05_collect_all
#SBATCH -o /export/home/kaziz/g05/logs/slurm-%x-%j.out
#SBATCH -p cpu-all
#SBATCH -c 2
#SBATCH --mem 4000MB
#SBATCH --time 00:20:00
cd /export/home/kaziz/g05 && python3 jobs/20_collect.py && python3 jobs/21_print_steer.py
