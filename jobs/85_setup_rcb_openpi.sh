#!/bin/bash -l
#SBATCH -o /export/home/kaziz/g05/logs/slurm-%x-%j.out
#SBATCH -p cpu-all
#SBATCH -c 8
#SBATCH --mem 32000MB
#SBATCH --time 04:00:00
# Env for pi0.5 training on RoboCasa365 with the official RoboCasa openpi fork (robocasa-benchmark/openpi): its own uv
# env (jax 0.5.3 / torch 2.7.1 / lerobot 0.3.3) plus robosuite + robocasa from our robocasa365 checkout (no-deps; the
# fork imports robocasa's dataset registry, and robocasa's default dataset path is our downloaded datasets/v1.0).
set -e
source /export/home/kaziz/g05/env_g05.sh; export PATH=$HOME/.local/bin:$PATH
G=/export/home/kaziz/g05; cd $G/baselines/rcb-openpi
# The fork's lock pins av 14.4.0, which builds from source here (needs ffmpeg dev libs); upstream openpi resolves av 17.0.0,
# which has a prebuilt wheel -> add it to the fork's existing uv overrides and re-lock.
grep -q '"av==17.0.0"' pyproject.toml || sed -i 's/^override-dependencies = \["ml-dtypes==0.4.1", "tensorstore==0.1.74"\]/override-dependencies = ["ml-dtypes==0.4.1", "tensorstore==0.1.74", "av==17.0.0"]/' pyproject.toml
grep -n "override-dependencies" pyproject.toml
GIT_LFS_SKIP_SMUDGE=1 uv sync -q --python 3.11 2>&1 | tail -5
# robosuite / robocasa on the path via a .pth file: an editable install makes setuptools walk the robocasa tree, which
# now holds the 97 GB of datasets and 33 GB of assets (it ran for an hour without finishing)
SP=$(.venv/bin/python -c "import site; print(site.getsitepackages()[0])")
printf "%s\n%s\n" $G/robocasa365/robosuite $G/robocasa365/robocasa > $SP/robocasa365_src.pth
uv pip install -q --python .venv/bin/python "mujoco==3.3.1" numba scipy termcolor h5py lxml hidapi gymnasium opencv-python-headless pyyaml tqdm imageio
unset MUJOCO_GL PYOPENGL_PLATFORM
.venv/bin/python -c "
from robocasa.utils.dataset_registry import DATASET_SOUP_REGISTRY, get_ds_meta
import openpi.training.config as C
print('soups', [k for k in DATASET_SOUP_REGISTRY if 'target' in k][:6])
print('target50 dirs', len(DATASET_SOUP_REGISTRY['target50']))
print('pi05 configs', [c.name for c in C._CONFIGS if 'pi05' in c.name and 'robocasa' in c.name or c.name.startswith('pi05_pretrain')])
"
echo ALL_DONE
