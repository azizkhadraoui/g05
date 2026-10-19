#!/bin/bash -l
#SBATCH -o /export/home/kaziz/g05/logs/slurm-%x-%j.out
#SBATCH -p cpu-all
#SBATCH -c 8
#SBATCH --mem 32000MB
#SBATCH --time 02:00:00
source /export/home/kaziz/g05/env_g05.sh
mkdir -p $G05_ROOT/models
for m in BAAI/RoboBrain2.0-3B BAAI/RoboBrain2.5-4B; do
  $PY -c "from huggingface_hub import snapshot_download; p=snapshot_download('$m', local_dir='$G05_ROOT/models/$(basename $m)'); print('done', p)"
done
du -sh $G05_ROOT/models/*; cat $G05_ROOT/models/*/config.json | grep -E '"architectures"|"model_type"' -A1 | head -12
