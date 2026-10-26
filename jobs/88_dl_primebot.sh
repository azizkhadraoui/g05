#!/bin/bash -l
#SBATCH -o /export/home/kaziz/g05/logs/slurm-%x-%j.out
#SBATCH -p cpu-all
#SBATCH -c 8
#SBATCH --mem 16000MB
#SBATCH --time 12:00:00
# Real household data for the world-model evaluation: PrimeBot Household Bimanual Manipulation Challenge
# (huggingface.co/datasets/challenge-2026/challenge_data, CC-BY-SA-4.0; X2W mobile bimanual robot, LeRobot v2.1, 30 Hz,
# head + 2 wrist cameras at 1280x720). Tasks: washer door open/close, load / unload clothes, basket, folding.
# Stage 1 (default): meta of every robot_data batch + partial_task_batch3_train in full (116 GB: door open, unload, basket).
# PATTERNS="..." downloads further allow-patterns (e.g. chosen chunks of batch4 / fold batch2) into the same tree.
set -e
export TMPDIR=/export/home/kaziz/g05/tmp   # node-local /tmp can be full (here-documents need it)
G=/export/home/kaziz/g05
export HF_TOKEN=$(cat $HOME/.cache/huggingface/token) HF_HUB_ENABLE_HF_TRANSFER=0
$G/baselines/Awomo-0.5/.venv/bin/python - <<PY
import os
from huggingface_hub import snapshot_download
B = "official_data/robot_data"
pats = os.environ.get("PATTERNS", "").split() or [f"{B}/*/meta/*", f"{B}/*/meta/**", f"{B}/partial_task_batch3_train/**",
                                                  "README.md", "robot_description/**"]
print(pats)
p = snapshot_download("challenge-2026/challenge_data", repo_type="dataset", allow_patterns=pats,
                      local_dir="$G/data/primebot", max_workers=8)
print(p)
PY
du -sh $G/data/primebot/official_data/robot_data/* ; echo ALL_DONE
