#!/bin/bash -l
#SBATCH -J g05_B01
#SBATCH -o /export/home/kaziz/g05/logs/slurm-%x-%j.out
#SBATCH -A h200
#SBATCH --qos h200_qos
#SBATCH -p gpu-H200
#SBATCH --gres gpu:1
#SBATCH -c 32
#SBATCH --mem 200000MB
#SBATCH --time 12:00:00
# B01: released LaWAM LIBERO SFT through LaWAM's own benchmark script (system baseline, not an ablation).
# Policy server: separate LaWAM venv (transformers 5.2, sdpa). Simulator client: G0.5 venv (LIBERO + mujoco 3.3.2).
source /export/home/kaziz/g05/env_g05.sh
cd $LAWAM_ROOT
CFG=$G05_ROOT/results/lawam_libero_cfg; mkdir -p $CFG
L=$LIBERO_ROOT/libero/libero
[ -f $CFG/config.yaml ] || printf "benchmark_root: $L\nbddl_files: $L/bddl_files\ninit_states: $L/init_files\ndatasets: /export/home/kaziz/ardy_vla/libero_hdf5\nassets: $L/assets\n" > $CFG/config.yaml
export STAR_VLA_PYTHON=$G05_ROOT/lawam_venv/bin/python LIBERO_PYTHON=$PY LIBERO_HOME=$LIBERO_ROOT LIBERO_CONFIG_PATH=$CFG
export NUM_GPUS=1 GPU_IDS=0 EVAL_GPU_IDS=0 NUM_TRIALS_PER_TASK=${TRIALS:-50} NUM_WORKERS=${WORKERS:-10}
export OUTPUT_ROOT=$G05_ROOT/results/B01_lawam RUN_TAG=${RUN_TAG:-full} PORT_LOCK_ROOT=$G05_ROOT/results/lawam_port_locks
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONFAULTHANDLER=1
bash examples/LIBERO/eval_files/auto_eval_scripts/run_libero_benchmark.sh \
    results/Checkpoints/libero/lawam_libero_sft_release/final_model/pytorch_model.pt
echo "rc=$?"
find $OUTPUT_ROOT -name "*.json" -newer $CFG/config.yaml | head -20
