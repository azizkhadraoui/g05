#!/bin/bash -l
#SBATCH -J g05_fix_deps
#SBATCH -o /export/home/kaziz/g05/logs/slurm-%x-%j.out
#SBATCH -p cpu-all
#SBATCH -c 8
#SBATCH --mem 32000MB
#SBATCH --time 02:00:00
# Post-sync fixes for the GalaxeaVLA venv:
#  - conda-forge FFmpeg 7 (dav1d) for AV1 videos; torchcodec finds it via RPATH, not LD_LIBRARY_PATH
#  - CPU torchcodec 0.4.0 from PyPI (the locked cu128 build needs NVIDIA NPP)
#  - lightning (LaWAM's latent_action_model/core/__init__.py imports it)
source /export/home/kaziz/g05/env_g05.sh
CONDA=/export/home/kaziz/motion/miniconda3/bin/conda
PATCHELF=$HOME/.local/bin/patchelf
TC_WHEEL=https://files.pythonhosted.org/packages/49/16/6713f9b26165f33f09e7b637f7189ec308b7f2ec3050cc60397f1b9f232e/torchcodec-0.4.0-cp310-cp310-manylinux_2_28_x86_64.whl
export UV_CACHE_DIR=${TMPDIR:-/tmp}/uv_cache_$USER
declare -A RC
run() { local n=$1; shift; echo; echo ">>> [$n] $(date +%T)"; "$@"; RC[$n]=$?; echo "<<< [$n] rc=${RC[$n]} $(date +%T)"; }
[ -x $G05_ROOT/ffmpeg/bin/ffmpeg ] || run ffmpeg_install $CONDA create -y -q -p $G05_ROOT/ffmpeg -c conda-forge --override-channels "ffmpeg=7"
run ffmpeg_av1 bash -c "$G05_ROOT/ffmpeg/bin/ffmpeg -hide_banner -decoders 2>/dev/null | grep -iE 'av1|dav1d'"

# diagnosis: does the conda ffmpeg lib dir on LD_LIBRARY_PATH break gcc?
printf 'int f(int x){return x*2;}\n' > /tmp/g05_t.c
run gcc_plain bash -c "gcc /tmp/g05_t.c -O3 -shared -fPIC -o /tmp/g05_t1.so && echo ok"
run gcc_triton_like bash -c "gcc /tmp/g05_t.c -O3 -shared -fPIC -Wno-psabi -I$(dirname $(dirname $(readlink -f $PY)))/include/python3.10 -o /tmp/g05_t2.so && echo ok"

cd $PROJECT_ROOT
run lightning uv pip install --python .venv/bin/python --no-deps "lightning==2.5.5"
# exact PyPI wheel, outside the project config (its cu128 index + unsafe-best-match would pick 0.4.0+cu128)
run torchcodec_cpu bash -c "cd /tmp && uv pip install --no-config --python $PROJECT_ROOT/.venv/bin/python --no-deps --reinstall --no-cache torchcodec@$TC_WHEEL"
run torchcodec_is_cpu bash -c "ls -d .venv/lib/python3.10/site-packages/torchcodec-*.dist-info; ! ls -d .venv/lib/python3.10/site-packages/torchcodec-*+cu*.dist-info 2>/dev/null"
[ -x $PATCHELF ] || run patchelf_install uv tool install patchelf
run rpath bash -c "for f in .venv/lib/python3.10/site-packages/torchcodec/libtorchcodec_*7.so; do $PATCHELF --add-rpath $G05_ROOT/ffmpeg/lib \$f && echo \$(basename \$f): \$($PATCHELF --print-rpath \$f) || exit 1; done"
run robosuite_nofilelog sed -i "s|^FILE_LOGGING_LEVEL = \"DEBUG\"|FILE_LOGGING_LEVEL = None  # g05: robosuite hard-codes the log file in tmp, owned by other users on shared nodes|" .venv/lib/python3.10/site-packages/robosuite/macros.py
run lawam_import $PY -c "import sys; sys.path.insert(0, '$LAWAM_ROOT'); import latent_action_model.core.lam_model; print('lawam lam_model ok')"
run decode $PY - <<'PYEOF'
import os, time, random, glob
assert 'ffmpeg' not in os.environ.get('LD_LIBRARY_PATH', ''), 'ffmpeg must come from RPATH, not LD_LIBRARY_PATH'
from torchcodec.decoders import VideoDecoder
files = sorted(glob.glob('data/libero/libero_merged_no_noops/videos/observation.images.image/chunk-000/*.mp4'))
d = VideoDecoder(files[0]); md = d.metadata
print('torchcodec ok:', md.codec, md.width, 'x', md.height, 'fps', md.average_fps, 'frames', md.num_frames)
random.seed(0); ts = [random.uniform(0, md.duration_seconds - 1) for _ in range(32)]
t = time.time(); fr = d.get_frames_played_at(ts); dt = time.time() - t
print(f'torchcodec: 32 random frames {tuple(fr.data.shape)} in {dt:.2f}s ({dt/32*1000:.0f} ms/frame)')
PYEOF
echo; echo "== summary"; FAIL=0
for k in ffmpeg_av1 gcc_plain gcc_triton_like lightning torchcodec_cpu torchcodec_is_cpu rpath lawam_import decode; do
  echo "$k rc=${RC[$k]}"; [[ ${RC[$k]} != 0 ]] && FAIL=1; done
exit $FAIL
