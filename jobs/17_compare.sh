#!/bin/bash -l
#SBATCH -J g05_compare
#SBATCH -o /export/home/kaziz/g05/logs/slurm-%x-%j.out
#SBATCH -p cpu-all
#SBATCH -c 2
#SBATCH --mem 4000MB
#SBATCH --time 00:20:00
cd /export/home/kaziz/g05
python3 jobs/10_compare.py results/screen/compare.json \
  F02_screen:F01_screen F01_screen:B00_screen F02_screen:B00_screen \
  F02_current:F02_screen F02_shuffled:F02_screen F02_suppressed:F02_screen \
  J1_screen:F02_screen J2_screen:F02_screen J3_screen:F02_screen \
  J1_screen:F01_screen J2_screen:F01_screen J3_screen:F01_screen J1_screen:B00_screen J2_screen:B00_screen J3_screen:B00_screen
for n in B00_screen F01_screen F02_screen F02_current F02_shuffled F02_suppressed J1_screen J2_screen J3_screen; do
  python3 -c "import json;s=json.load(open('results/screen/$n/screen_summary.json'));print('$n',s['successes'],'/',s['total'],round(s['rate'],3))" 2>/dev/null || echo "$n missing"
done
