#!/bin/bash -l
#SBATCH -o /export/home/kaziz/g05/logs/slurm-%x-%j.out
#SBATCH -p cpu-all
#SBATCH -c 8
#SBATCH --mem 16000MB
#SBATCH --time 12:00:00
# Checkpoints for the household world-model evaluation (wm/):
#  RoboJEPA (facebookresearch/robo_jepa, CC BY-NC-SA 4.0): predictors 22M / 300M / 1B / 8B at the paper's checkpoints,
#    8B DROID 720p 3-view, the shared frozen V-JEPA 2.1 ViT-G/384 encoder, and the Cosmos tokenizer (pixel decoder).
#  Cosmos 3 (NVIDIA, OpenMDW 1.1): Cosmos3-Edge (4B) and Cosmos3-Edge-Policy-DROID.
#  FLUX 3 Action (FLUX Kommunity License, gated): base + DROID policy, once the licence is accepted (FLUX=1).
set -u
W=/export/home/kaziz/g05/wm/checkpoints; mkdir -p $W/robojepa; cd $W/robojepa
F=https://dl.fbaipublicfiles.com
get() { [ -s "$2" ] || wget -q -O "$2.part" "$1" && mv "$2.part" "$2" 2>/dev/null; ls -la "$2" | awk '{print $5, $9}'; }
get $F/vjepa2/vjepa2_1_vitG_384.pt vjepa2_1_vitG_384.pt
for s in "22m 150k" "300m 300k" "1b 300k" "8b 300k"; do set -- $s; mkdir -p wm-$1
  cp -n /export/home/kaziz/g05/wm/robo_jepa/configs/released/wm-$1/config.json wm-$1/ 2>/dev/null
  get $F/robojepa/wm/wm-$1/cooldown_$1_$2.pth.tar wm-$1/cooldown_$1_$2.pth.tar; ln -sfn ../vjepa2_1_vitG_384.pt wm-$1/encoder.pth.tar; done
mkdir -p wm-droid-8b; cp -n /export/home/kaziz/g05/wm/robo_jepa/configs/released/wm-droid-8b-3views-720p/config.json wm-droid-8b/
get $F/robojepa/wm/finetuning/wm-droid-8b-3views-720p_300k/world_model.pth.tar wm-droid-8b/world_model.pth.tar
ln -sfn ../vjepa2_1_vitG_384.pt wm-droid-8b/encoder.pth.tar
export HF_TOKEN=$(cat $HOME/.cache/huggingface/token)
/export/home/kaziz/g05/baselines/Awomo-0.5/.venv/bin/python - <<'PY'
import os
from huggingface_hub import snapshot_download
W = "/export/home/kaziz/g05/wm/checkpoints"
repos = ["nvidia/Cosmos3-Edge", "nvidia/Cosmos3-Edge-Policy-DROID", "nvidia/Cosmos-0.1-Tokenizer-CV4x8x8"]
if os.environ.get("FLUX") == "1":
    repos += ["black-forest-labs/flux-3-action-base", "black-forest-labs/flux-3-action-droid"]
for r in repos:
    try:
        print(r, snapshot_download(r, local_dir=f"{W}/{r.split('/')[1]}", max_workers=8), flush=True)
    except Exception as e:
        print("FAILED", r, type(e).__name__, str(e)[:200], flush=True)
PY
du -sh $W/* $W/robojepa/*; echo ALL_DONE
