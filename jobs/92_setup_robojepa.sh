#!/bin/bash -l
#SBATCH -o /export/home/kaziz/g05/logs/slurm-%x-%j.out
#SBATCH -p cpu-all
#SBATCH -c 8
#SBATCH --mem 32000MB
#SBATCH --time 04:00:00
# uv env for RoboJEPA (wm/robo_jepa): Python 3.11 + the pinned pip list of its environment.yml (torch 2.7.1 cu126, which
# has T4/V100 kernels). The conda-only evdev pin is skipped (it is for RoboCasa teleop; it fails to build here), and the
# Cosmos tokenizer is installed without deps as its README says. Also used for V-JEPA 2.1 features in the WM harness.
set -e
export TMPDIR=/export/home/kaziz/g05/tmp
source /export/home/kaziz/g05/env_g05.sh; export PATH=$HOME/.local/bin:$PATH
cd /export/home/kaziz/g05/wm/robo_jepa
awk '/- pip:/{p=1; next} p && /^ *- /{sub(/^ *- /, ""); print}' environment.yml | grep -v "^evdev\|^pynput" > /export/home/kaziz/g05/tmp/robojepa_req.txt
wc -l /export/home/kaziz/g05/tmp/robojepa_req.txt; grep -n "^torch\|index-url" /export/home/kaziz/g05/tmp/robojepa_req.txt | head
uv venv -q -p 3.11 --clear .venv
uv pip install -q --python .venv/bin/python --index-strategy unsafe-best-match -r /export/home/kaziz/g05/tmp/robojepa_req.txt 2>&1 | tail -5
uv pip install -q --python .venv/bin/python --no-deps "git+https://github.com/NVIDIA/Cosmos-Tokenizer.git@3584ae752ce8ebdbe06a420bf60d7513c0e878cc" 2>&1 | tail -3
PYTHONPATH=$PWD .venv/bin/python -c "import torch; from app.robo_jepa.world_model import WorldModel; print('robojepa ok', torch.__version__, torch.version.cuda)"
echo ALL_DONE
