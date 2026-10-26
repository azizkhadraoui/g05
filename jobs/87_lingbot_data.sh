#!/bin/bash -l
#SBATCH -o /export/home/kaziz/g05/logs/slurm-%x-%j.out
#SBATCH -p cpu-all
#SBATCH -c 16
#SBATCH --mem 64000MB
#SBATCH --time 24:00:00
# LingBot-VLA trains from one LeRobot v3.0 directory; RoboCasa365 ships 50 LeRobot v2.1 task datasets. Convert each task to
# v3.0 (lerobot 0.4.2 converter, LingBot env) and aggregate the 50 into data/rc365_v30/target50.
# The converter works in place on <root>/<repo_id>: it is pointed at symlinks, so the shared v2.1 datasets (used by GR00T,
# pi0.5 and G0.5) are never modified; the old link is renamed *_old and the converted copy takes its name.
set -eu
G=/export/home/kaziz/g05; P=$G/baselines/lingbot-vla/.venv/bin/python
L=$G/baselines/lingbot-vla/.venv/lib/python3.10/site-packages/lerobot/datasets/v30/convert_dataset_v21_to_v30.py
DS=$G/robocasa365/robocasa/datasets/v1.0/target; W=$G/data/rc365_v30; mkdir -p $W/src
for d in $DS/*/*/*/lerobot; do t=$(basename $(dirname $(dirname $d))); [ -e $W/src/$t ] || ln -s $d $W/src/$t; done
echo "tasks: $(ls $W/src | grep -v _old | wc -l)"
export HF_HUB_OFFLINE=1
ls $W/src | grep -v "_old$" | xargs -P ${PAR:-8} -I{} bash -c \
  "[ -f $W/src/{}/meta/episodes/chunk-000/file-000.parquet ] && echo 'skip {}' || \
   ($P $L --repo-id={} --root=$W/src --push-to-hub=false > $W/convert_{}.log 2>&1 && echo 'converted {}' || echo 'FAILED {}')"
$P - <<PY
import glob, json
from pathlib import Path
import pandas as pd
from lerobot.datasets.aggregate import aggregate_datasets
W = Path("$W"); names = sorted(p.name for p in (W / "src").iterdir() if not p.name.endswith("_old") and not p.is_symlink())
print(len(names), "converted tasks")
# 18 of the 50 tasks carry subtask annotations the others lack; aggregation needs identical features, and LingBot does
# not use them, so drop them from our converted copies (data parquet, info.json, stats.json, per-episode stats).
EXTRA = ("annotation.human.subtask", "annotation.human.subtask_name", "annotation.human.subtask_stage", "subtask_idx")
for n in names:
    root = W / "src" / n; info = json.loads((root / "meta/info.json").read_text())
    if not any(k in info["features"] for k in EXTRA):
        continue
    for k in EXTRA:
        info["features"].pop(k, None)
    (root / "meta/info.json").write_text(json.dumps(info, indent=4))
    st = json.loads((root / "meta/stats.json").read_text()); [st.pop(k, None) for k in EXTRA]
    (root / "meta/stats.json").write_text(json.dumps(st, indent=4))
    for f in glob.glob(str(root / "data/**/*.parquet"), recursive=True) + glob.glob(str(root / "meta/episodes/**/*.parquet"), recursive=True):
        df = pd.read_parquet(f); drop = [c for c in df.columns if c in EXTRA or any(c.startswith(f"stats/{k}/") for k in EXTRA)]
        if drop:
            df.drop(columns=drop).to_parquet(f, index=False)
    print("dropped subtask features from", n)
aggregate_datasets(repo_ids=names, aggr_repo_id="rc365_target50", roots=[W / "src" / n for n in names], aggr_root=W / "target50")
print("aggregated ->", W / "target50")
PY
du -sh $W/target50; echo ALL_DONE
