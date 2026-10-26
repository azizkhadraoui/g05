#!/bin/bash -l
#SBATCH -o /export/home/kaziz/g05/logs/slurm-%x-%j.out
#SBATCH -p cpu-all
#SBATCH -c 4
#SBATCH --mem 16000MB
#SBATCH --time 03:00:00
# Awomo-0.5I base models at the revisions pinned in awomo_config.json (both Apache-2.0, not gated), plus the simulator-side
# requirement of its RoboCasa365 harness (msgpack) in the robocasa env.
set -e
source /export/home/kaziz/g05/env_g05.sh; export PATH=$HOME/.local/bin:$PATH
G=/export/home/kaziz/g05
$G/baselines/Awomo-0.5/.venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download
G = "/export/home/kaziz/g05/models"
p = snapshot_download("Qwen/Qwen3-VL-4B-Instruct", revision="ebb281ec70b05090aa6165b016eac8ec08e71b17", local_dir=f"{G}/Qwen3-VL-4B-Instruct")
print("qwen", p)
p = snapshot_download("black-forest-labs/FLUX.2-klein-base-4B", revision="a3b4f4849157f664bdbc776fd7453c2783562f4d",
                      allow_patterns=["vae/*"], local_dir=f"{G}/FLUX.2-klein-base-4B")
print("vae", p)
PY
uv pip install -q --python $G/robocasa365/.venv/bin/python msgpack==1.2.3
$G/robocasa365/.venv/bin/python -c "import msgpack; print('msgpack', msgpack.version)"
du -sh $G/models/Qwen3-VL-4B-Instruct $G/models/FLUX.2-klein-base-4B
echo ALL_DONE
