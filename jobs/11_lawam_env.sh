#!/bin/bash -l
#SBATCH -J g05_lawam_env
#SBATCH -o /export/home/kaziz/g05/logs/slurm-%x-%j.out
#SBATCH -p cpu-all
#SBATCH -c 8
#SBATCH --mem 32000MB
#SBATCH --time 04:00:00
# B01 prerequisites: separate LaWAM venv (transformers 5.2 / torch 2.6; sdpa attention, no flash-attn) + weights.
source /export/home/kaziz/g05/env_g05.sh
export HF_XET_HIGH_PERFORMANCE=1 UV_CACHE_DIR=${TMPDIR:-/tmp}/uv_cache_$USER UV_LINK_MODE=copy UV_HTTP_TIMEOUT=300
HF=/export/home/kaziz/motion/miniconda3/envs/ardy/bin/hf
L=$LAWAM_ROOT; V=$G05_ROOT/lawam_venv
declare -A RC
run() { local n=$1; shift; echo ">>> [$n] $(date +%T)"; "$@"; RC[$n]=$?; echo "<<< [$n] rc=${RC[$n]} $(date +%T)"; }
run sft $HF download jialei02/lawam_libero_sft_release --local-dir $L/results/Checkpoints/libero/lawam_libero_sft_release
run qwen3vl $HF download Qwen/Qwen3-VL-2B-Instruct --local-dir $L/results/Checkpoints/qwen3_weights
[ -x $V/bin/python ] || run venv uv venv $V --python 3.10
cd $L
run reqs uv pip install --python $V/bin/python -r requirements.txt
run pkg uv pip install --python $V/bin/python -e . --no-deps
run imports $V/bin/python -c "import torch, transformers, starVLA; print(torch.__version__, transformers.__version__, 'starVLA ok')"
echo "== summary"; FAIL=0; for k in sft qwen3vl reqs pkg imports; do echo "$k rc=${RC[$k]}"; [[ ${RC[$k]} != 0 ]] && FAIL=1; done
exit $FAIL
