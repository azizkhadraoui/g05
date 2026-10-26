# g05: G0.5-FM + LaWAM world model. Source from any job: source /export/home/kaziz/g05/env_g05.sh
export G05_ROOT=/export/home/kaziz/g05
export PROJECT_ROOT=$G05_ROOT/GalaxeaVLA
export LAWAM_ROOT=$G05_ROOT/LaWAM
export G05_OUTPUT_DIR=$G05_ROOT/runs
export HF_HOME=$G05_ROOT/hf_cache
export HF_HUB_CACHE=$HF_HOME/hub
export HF_DATASETS_CACHE=$HF_HOME/datasets
[ -f ~/.cache/huggingface/token ] && export HF_TOKEN=$(cat ~/.cache/huggingface/token)
export PATH=$HOME/.local/bin:$PATH
export UV_PYTHON_INSTALL_DIR=$G05_ROOT/.uv_python
export LIBERO_ROOT=/export/home/kaziz/ardy_vla/LIBERO      # shared with the ardy-vla experiment: read-only use
export LIBERO_CONFIG_PATH=$PROJECT_ROOT/experiments/libero
export PYTHONPATH=$PROJECT_ROOT:${PYTHONPATH:-}
export TOKENIZERS_PARALLELISM=false HYDRA_FULL_ERROR=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl LC_CTYPE=en_US.UTF-8
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 CV_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_PROJECT=g05_lawam
export PY=$PROJECT_ROOT/.venv/bin/python
[ -f $PROJECT_ROOT/.venv/bin/activate ] && source $PROJECT_ROOT/.venv/bin/activate
# V100 nodes (sm70): the .venv torch build (cu128) has no Volta kernels -> its cu126 twin (jobs/39_venv_sm70.sh, pins: 139)
if [ -x $PROJECT_ROOT/.venv_sm70/bin/python ] && nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | grep -q V100; then
    export PY=$PROJECT_ROOT/.venv_sm70/bin/python; source $PROJECT_ROOT/.venv_sm70/bin/activate
fi
# torchcodec: the venv carries the CPU torchcodec 0.4.0 from PyPI with RPATH -> $G05_ROOT/ffmpeg/lib (jobs/03_fix_deps.sh);
# re-run 03 after any `uv sync`. Do NOT put ffmpeg/lib on LD_LIBRARY_PATH: it shadows system libs (suspected gcc/Triton crash).
# Newer C++ runtime first (conda-forge libstdc++ / libgcc_s only, symlinked from ffmpeg/lib): the conda FFmpeg's ICU
# needs GLIBCXX_3.4.30, but the cluster's GCC 11 lib64 on LD_LIBRARY_PATH otherwise wins. Backward compatible.
export LD_LIBRARY_PATH=$G05_ROOT/libstdcxx:${LD_LIBRARY_PATH:-}
