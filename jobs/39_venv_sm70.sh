#!/bin/bash -l
#SBATCH -o /export/home/kaziz/g05/logs/slurm-%x-%j.out
#SBATCH -p cpu-all
#SBATCH -c 8
#SBATCH --mem 32000MB
#SBATCH --time 01:30:00
# Second G0.5 venv whose torch build includes Volta (sm_70) kernels, for the V100 nodes. Same torch version (2.7.1), cu126.
source /export/home/kaziz/g05/env_g05.sh
cd /export/home/kaziz/g05/GalaxeaVLA
rm -rf .venv_sm70 && cp -a .venv .venv_sm70 && echo copied
sed -i "s#/GalaxeaVLA/.venv#/GalaxeaVLA/.venv_sm70#g" .venv_sm70/bin/activate .venv_sm70/pyvenv.cfg 2>/dev/null
which uv || export PATH=$HOME/.local/bin:$HOME/.cargo/bin:$PATH
uv pip install --python .venv_sm70/bin/python --index-url https://download.pytorch.org/whl/cu126 --reinstall \
  torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 && echo installed
.venv_sm70/bin/python -c "import torch; print(torch.__version__, torch.cuda.get_arch_list() if torch.cuda.is_available() else 'no-gpu')"
.venv_sm70/bin/python -c "import torch, torch.utils.cpp_extension as c; print('cuda', torch.version.cuda)"
