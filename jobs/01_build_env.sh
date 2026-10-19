#!/bin/bash -l
#SBATCH -J g05_build_env
#SBATCH -o /export/home/kaziz/g05/logs/slurm-%x-%j.out
#SBATCH -p cpu-all
#SBATCH -c 8
#SBATCH --mem 32000MB
#SBATCH --time 04:00:00
source /export/home/kaziz/g05/env_g05.sh
module load gcc/11.2.0 2>/dev/null
export UV_CACHE_DIR=${TMPDIR:-/tmp}/uv_cache_$USER UV_LINK_MODE=copy   # node-local cache: NFS is slow on many small files
echo "node: $(hostname)"; curl -sfI --max-time 15 https://pypi.org/simple/ >/dev/null || { echo "NO INTERNET ON COMPUTE NODE"; exit 3; }
cd $PROJECT_ROOT
export CMAKE_POLICY_VERSION_MINIMUM=3.5
uv tool install --force "cmake<4" && cmake --version | head -1 || exit 1
export UV_HTTP_TIMEOUT=300 UV_CONCURRENT_DOWNLOADS=8
for a in 1 2 3; do uv sync --frozen && break; echo "uv sync attempt $a failed"; [ $a = 3 ] && exit 1; sleep 30; done
SP=$(.venv/bin/python -c 'import site; print(site.getsitepackages()[0])')
echo "$LIBERO_ROOT" > $SP/libero_ardy_vla.pth     # import shared LIBERO without writing into its checkout
declare -A RC
chk() { local n=$1; shift; echo ">>> [$n]"; "$@"; RC[$n]=$?; echo "<<< [$n] rc=${RC[$n]}"; }
chk core   .venv/bin/python -c "import torch, torchvision, g05, transformers; print(torch.__version__, torch.version.cuda, transformers.__version__)"
chk fla    .venv/bin/python -c "import fla; print('fla', fla.__version__)"
chk fa4    .venv/bin/python -c "import flash_attn.cute as c; print('flash-attn-4 ok')"
chk libero .venv/bin/python -c "from libero.libero import benchmark, get_libero_path; print('libero ok', list(benchmark.get_benchmark_dict())[:5])"
chk lawam  .venv/bin/python -c "import sys; sys.path.insert(0, '$LAWAM_ROOT'); import latent_action_model.core.lam_model as m; print('lawam lam_model ok')"
echo "== summary"; FAIL=0; for k in "${!RC[@]}"; do echo "$k rc=${RC[$k]}"; [[ $k != lawam && $k != fa4 && ${RC[$k]} != 0 ]] && FAIL=1; done
exit $FAIL
