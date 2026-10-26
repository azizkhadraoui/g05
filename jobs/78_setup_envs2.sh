#!/bin/bash -l
#SBATCH -p cpu-all -c 8 --mem 32G -t 06:00:00 -o /export/home/kaziz/g05/logs/slurm-%x-%j.out
# Python envs for LingBot-VLA-4B and Awomo-0.5I (RoboCasa365 benchmark). torch from the cu126 index so T4 (sm75) and V100 (sm70)
# kernels are present; flash-attn is skipped (no FA2 on T4/V100, the models fall back to SDPA).
set -e
source /export/home/kaziz/g05/env_g05.sh; export PATH=$HOME/.local/bin:$PATH
B=/export/home/kaziz/g05/baselines
IDX="--extra-index-url https://download.pytorch.org/whl/cu126 --index-strategy unsafe-best-match"
WHICH=${WHICH:-lingbot awomo}
for w in $WHICH; do
  if [ "$w" = lingbot ]; then
    cd $B/lingbot-vla
    git submodule update --init --recursive || true
    ls lingbotvla/models/vla/vision_models/lingbot-depth lingbotvla/models/vla/vision_models/MoGe | head -4
    uv venv -q -p 3.10 --clear .venv
    # The official install.sh runs `pip install lerobot` then `pip install -e .`, so LingBot's own pins win over lerobot's
    # (torch 2.8 / torchvision 0.23 / torchcodec 0.6 / datasets 3.6 / numpy 1.26.4 / transformers 4.51.3). Reproduce that
    # environment with overrides; the resolved lock is ~/.lingbot_res/lock.txt (uv pip compile, same flags).
    printf "torch==2.8.0\ntorchvision==0.23.0\ntorchcodec==0.6.0\ndatasets==3.6.0\nnumpy==1.26.4\ntransformers==4.51.3\nhuggingface-hub>=0.30.0,<1.0\n" > $HOME/.lingbot_overrides.txt
    uv pip install -q --python .venv/bin/python $IDX --override $HOME/.lingbot_overrides.txt "torch==2.8.0" "torchvision==0.23.0" \
      "lerobot @ https://github.com/huggingface/lerobot/archive/refs/tags/v0.4.2.tar.gz" -r requirements.txt -e .
    uv pip install -q --python .venv/bin/python --no-deps -e ./lingbotvla/models/vla/vision_models/lingbot-depth/
    uv pip install -q --python .venv/bin/python $IDX --override $HOME/.lingbot_overrides.txt -e ./lingbotvla/models/vla/vision_models/MoGe/
    .venv/bin/python -c "import torch, transformers, lerobot, lingbotvla; print('lingbot ok', torch.__version__, torch.cuda.get_arch_list() if torch.cuda.is_available() else torch.version.cuda, transformers.__version__)"
  fi
  if [ "$w" = awomo ]; then
    cd $B/Awomo-0.5
    uv venv -q -p 3.11 --clear .venv
    uv pip install -q --python .venv/bin/python $IDX -r requirements-05i.txt
    PYTHONPATH=$PWD .venv/bin/python -c "import torch, transformers, diffusers, awomo; print('awomo ok', torch.__version__, torch.version.cuda, transformers.__version__)"
  fi
done
echo ALL_DONE
