#!/bin/bash -l
#SBATCH -o /export/home/kaziz/g05/logs/slurm-%x-%j.out
#SBATCH -p cpu-all
#SBATCH -c 4
#SBATCH --mem 16000MB
#SBATCH --time 04:00:00
# Official RoboCasa365 checkpoints (robocasa/robocasa365_checkpoints) for the "high-compute reference" track: evaluated
# with our harness next to the equal-budget fine-tunes. pi0.5 multitask (300 pretraining tasks, 75k steps x batch 64):
# params + norm stats only (12.4 GB; the 32 GB optimizer state is skipped).
set -e
G=/export/home/kaziz/g05
$G/baselines/Awomo-0.5/.venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download
p = "pi05_pretrain_human300/multitask_learning/75000"
out = snapshot_download("robocasa/robocasa365_checkpoints", allow_patterns=[f"{p}/params/*", f"{p}/params/**", f"{p}/assets/*",
                        f"{p}/_CHECKPOINT_METADATA"], local_dir="/export/home/kaziz/g05/models/robocasa365_official")
print(out)
PY
du -sh $G/models/robocasa365_official/*/*/*
echo ALL_DONE
