#!/bin/bash -l
#SBATCH -o /export/home/kaziz/g05/logs/slurm-%x-%j.out
#SBATCH -p gpu-all
#SBATCH --gres gpu:T4_16GB:1
#SBATCH -c 16
#SBATCH --mem 100000MB
#SBATCH --time 06:00:00
# P01 closed loop: world-model planning over G0.5 action chunks (49_wm_planner.py). Server returns full 32-step chunks.
# Env: MODES="wm random ..." TASKS="0,1,..." TRIALS=10 NAME=...
source /export/home/kaziz/g05/env_g05.sh
cd $PROJECT_ROOT
NAME=${NAME:-run}; OUT=$G05_ROOT/results/wmplan/$NAME; mkdir -p $OUT
PORT=$(( 20000 + SLURM_JOB_ID % 20000 ))
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONFAULTHANDLER=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
$PY $G05_ROOT/jobs/serve_policy_hybrid.py --ckpt_path checkpoints/g05-libero/model.pt --host 0.0.0.0 --port $PORT \
    eval_embodiment=libero --action_steps 32 --max_batch_size 10 --max_wait_ms 200 --seed 7 &> $OUT/server_${SLURM_JOB_ID}.log &
SPID=$!
for i in $(seq 1 300); do ss -tlnpH "sport = :$PORT" 2>/dev/null | grep -q "$PORT" && break; kill -0 $SPID 2>/dev/null || { tail -30 $OUT/server_${SLURM_JOB_ID}.log; exit 1; }; sleep 2; done
for m in ${MODES:-oracle wm}; do
  t0=$(date +%s)
  $PY -u $G05_ROOT/jobs/49_wm_planner.py --server_uri ws://127.0.0.1:$PORT --mode $m --task_ids ${TASKS:-0,1,2,3,4,5,6,7,8,9} \
      --trials ${TRIALS:-10} --num_parallel ${PAR:-5} --output_dir $OUT/$m ${EXTRA:-} &> $OUT/$m.log
  echo "$m rc=$? wall=$(( $(date +%s) - t0 ))s $(grep -h '"mode"' $OUT/$m.log | tail -1)"
done
kill $SPID
