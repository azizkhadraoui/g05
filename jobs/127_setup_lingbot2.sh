#!/bin/bash -l
#SBATCH -o /export/home/kaziz/g05/logs/slurm-%x-%j.out
#SBATCH -p cpu-all
#SBATCH -c 8
#SBATCH --mem 32000MB
#SBATCH --time 04:00:00
# LingBot-VLA 2.0 (Robbyant, Apache-2.0 code): repo, env (Python 3.12, torch 2.8) and the 6B checkpoint, for the low-level
# policy bake-off (RoboCasa365 through Awomo's harness, then OpenArm). No flash-attn: the prebuilt 2.8.3 wheel needs glibc 2.32
# (T4 nodes have 2.31) and FlashAttention 2 has no sm75 kernels anyway, so the model runs with PyTorch SDPA.
set -e
G=/export/home/kaziz/g05; B=$G/baselines; M=$G/models
export TMPDIR=$G/tmp HF_TOKEN=$(cat $HOME/.cache/huggingface/token) HF_HUB_DISABLE_XET=1 UV_HTTP_TIMEOUT=600 UV_CONCURRENT_DOWNLOADS=4
source $G/env_g05.sh; export PATH=$HOME/.local/bin:$PATH
cd $B
[ -d lingbot-vla-v2 ] || git clone -q https://github.com/robbyant/lingbot-vla-v2.git
cd lingbot-vla-v2; git log -1 --format="%H %cd"
ls; ls configs 2>/dev/null | head; find . -name "*.yaml" -path "*robotwin*" | head -5
uv venv -q -p 3.12 --clear .venv
uv pip install -q --python .venv/bin/python --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cu128 \
  "torch==2.8.0" "torchvision==0.23.0" 2>&1 | tail -3
if [ -f requirements.txt ]; then grep -v -i -e "flash.attn" -e "^torch==" -e "^torchvision==" requirements.txt > $TMPDIR/lb2_req.txt
  uv pip install -q --python .venv/bin/python --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cu128 \
    "torch==2.8.0" "torchvision==0.23.0" -r $TMPDIR/lb2_req.txt 2>&1 | tail -5; fi
if [ -f pyproject.toml ]; then uv pip install -q --python .venv/bin/python -e . --no-deps 2>&1 | tail -3; fi
.venv/bin/python -c "import torch, transformers; print('lingbot2 env ok', torch.__version__, torch.version.cuda, transformers.__version__)"
$B/Awomo-0.5/.venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download
for r in ("robbyant/lingbot-vla-v2-6b", "lerobot/lingbot_vla_v2_base"):
    try:
        print(r, snapshot_download(r, local_dir=f"/export/home/kaziz/g05/models/{r.split('/')[1]}", max_workers=8), flush=True)
        break
    except Exception as e:
        print("FAILED", r, type(e).__name__, str(e)[:200], flush=True)
PY
du -sh $M/lingbot* 2>/dev/null
echo ALL_DONE
