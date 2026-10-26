#!/bin/bash -l
#SBATCH -o /export/home/kaziz/g05/logs/slurm-%x-%j.out
#SBATCH -p cpu-all
#SBATCH -c 8
#SBATCH --mem 32000MB
#SBATCH --time 03:00:00
# Env for FLUX 3 Action (wm/flux-action, docs/setup.md): the locked uv env (torch 2.10 cu128, which keeps T4 sm75
# kernels; the nodes' 570 driver runs CUDA 12.8) + the NATTEN wheel built for it, and the shared frozen encoders
# (Qwen3-VL-4B text encoder, video VAE) at the base-repository revision the DROID package pins.
set -e
export TMPDIR=/export/home/kaziz/g05/tmp
source /export/home/kaziz/g05/env_g05.sh; export PATH=$HOME/.local/bin:$PATH
F=/export/home/kaziz/g05/wm/flux-action
cd $F
uv sync -q --locked --extra encoders --extra data 2>&1 | tail -5
uv pip install -q --python .venv/bin/python 'natten==0.21.6+torch2100cu128' -f https://whl.natten.org/ 2>&1 | tail -3
uv pip install -q --python .venv/bin/python pandas 2>&1 | tail -2  # 112_flux_action_eval.py reads the DROID parquet rows
.venv/bin/python -c "import torch, natten, transformers, av, flux_action; print('flux env ok', torch.__version__, torch.version.cuda, natten.__version__, transformers.__version__)"
HF_TOKEN=$(cat $HOME/.cache/huggingface/token) HF_HUB_DISABLE_XET=1 .venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download
p = snapshot_download("black-forest-labs/flux-3-action-base", revision="62878e2925e59b7a89ec14463ce89932624c490d",
                      allow_patterns=["video_vae.safetensors", "text_encoder/*"],
                      local_dir="/export/home/kaziz/g05/wm/checkpoints/flux-3-action-base", max_workers=8)
print("base encoders", p, flush=True)
PY
du -sh /export/home/kaziz/g05/wm/checkpoints/flux-3-action-base
echo ALL_DONE
