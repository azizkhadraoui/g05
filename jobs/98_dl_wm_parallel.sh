#!/bin/bash -l
#SBATCH -o /export/home/kaziz/g05/logs/slurm-%x-%j.out
#SBATCH -p cpu-all
#SBATCH -c 8
#SBATCH --mem 16000MB
#SBATCH --time 12:00:00
# World-model checkpoints in parallel (97_dl_parallel.py for the RoboJEPA / V-JEPA files, huggingface_hub for Cosmos 3).
set -u
export TMPDIR=/export/home/kaziz/g05/tmp
W=/export/home/kaziz/g05/wm/checkpoints; R=$W/robojepa; F=https://dl.fbaipublicfiles.com; P=/export/home/kaziz/g05/wm/robo_jepa/.venv/bin/python
for s in 22m 300m 1b 8b; do mkdir -p $R/wm-$s; cp -n /export/home/kaziz/g05/wm/robo_jepa/configs/released/wm-$s/config.json $R/wm-$s/; ln -sfn ../vjepa2_1_vitG_384.pt $R/wm-$s/encoder.pth.tar; done
mkdir -p $R/wm-droid-8b; cp -n /export/home/kaziz/g05/wm/robo_jepa/configs/released/wm-droid-8b-3views-720p/config.json $R/wm-droid-8b/; ln -sfn ../vjepa2_1_vitG_384.pt $R/wm-droid-8b/encoder.pth.tar
export HF_TOKEN=$(cat $HOME/.cache/huggingface/token)
/export/home/kaziz/g05/baselines/Awomo-0.5/.venv/bin/python - <<'PY' &
from huggingface_hub import snapshot_download
W = "/export/home/kaziz/g05/wm/checkpoints"
for r in ["nvidia/Cosmos3-Edge-Policy-DROID", "nvidia/Cosmos3-Edge", "nvidia/Cosmos-0.1-Tokenizer-CV4x8x8"]:
    try:
        print(r, snapshot_download(r, local_dir=f"{W}/{r.split('/')[1]}", max_workers=8), flush=True)
    except Exception as e:
        print("FAILED", r, type(e).__name__, str(e)[:200], flush=True)
PY
HFPID=$!
$P /export/home/kaziz/g05/jobs/97_dl_parallel.py \
  $F/vjepa2/vjepa2_1_vitG_384.pt=$R/vjepa2_1_vitG_384.pt \
  $F/robojepa/wm/wm-22m/cooldown_22m_150k.pth.tar=$R/wm-22m/cooldown_22m_150k.pth.tar \
  $F/robojepa/wm/wm-300m/cooldown_300m_300k.pth.tar=$R/wm-300m/cooldown_300m_300k.pth.tar \
  $F/robojepa/wm/wm-1b/cooldown_1b_300k.pth.tar=$R/wm-1b/cooldown_1b_300k.pth.tar \
  $F/robojepa/wm/wm-8b/cooldown_8b_300k.pth.tar=$R/wm-8b/cooldown_8b_300k.pth.tar \
  $F/robojepa/wm/finetuning/wm-droid-8b-3views-720p_300k/world_model.pth.tar=$R/wm-droid-8b/world_model.pth.tar
wait $HFPID
du -sh $W/* $R/*; echo ALL_DONE
