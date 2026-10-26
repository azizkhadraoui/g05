#!/bin/bash -l
#SBATCH -o /export/home/kaziz/g05/logs/slurm-%x-%j.out
#SBATCH -p cpu-all
#SBATCH -c 8
#SBATCH --mem 32000MB
#SBATCH --time 02:00:00
# Complete the LingBot-VLA 2.0 venv with the training-side steps of the repo's tools/create_train_env.sh that
# 127_setup_lingbot2.sh skipped: numpydantic, lerobot v0.4.2 (no deps), the depth requirements, the local MoGe /
# lingbot-depth packages and utils3d path, huggingface_hub 0.34.0; the torch 2.8 / cu128 pins are restored at the end.
# (flash-attn is installed by 135_lingbot_rc365.sh on the A100 node; its wheel needs glibc >= 2.32.)
set -e
G=/export/home/kaziz/g05; L=$G/baselines/lingbot-vla-v2; P=$L/.venv/bin/python
export PATH=$HOME/.local/bin:$PATH UV_HTTP_TIMEOUT=600 TMPDIR=$G/tmp MUJOCO_GL=disable
cd $L
PIN=(--index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cu128 torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0)
uv pip install --python $P numpydantic==1.9.0 --no-deps
uv pip install --python $P --no-deps "lerobot @ https://github.com/huggingface/lerobot/archive/refs/tags/v0.4.2.tar.gz"
uv pip install --python $P "${PIN[@]}" -r requirements-depth.txt
grep -v -i -e "flash.attn" -e "^torch==" -e "^torchvision==" requirements.txt > $TMPDIR/lb2_req.txt
uv pip install --python $P "${PIN[@]}" -r $TMPDIR/lb2_req.txt
uv pip install --python $P numpydantic==1.9.0 --no-deps
echo "$L/lingbotvla/models/vla/vision_models/morgbd_clean/3rd/utils3d" > $L/.venv/lib/python3.12/site-packages/stablevla_local_depth.pth
uv pip install --python $P --no-deps -e lingbotvla/models/vla/vision_models/lingbot-depth
uv pip install --python $P "${PIN[@]}" -e lingbotvla/models/vla/vision_models/MoGe
uv pip install --python $P "${PIN[@]}" huggingface_hub==0.34.0
$P - <<'PY'
import torch, lerobot, cv2, accelerate, mlflow, trimesh, moge, mdm, utils3d
from lerobot.datasets.lerobot_dataset import LeRobotDataset
print("torch", torch.__version__, torch.version.cuda, "| lerobot", getattr(lerobot, "__version__", "?"), "| depth imports ok")
assert torch.__version__.startswith("2.8.0"), torch.__version__
print("LINGBOT_ENV_OK")
PY
