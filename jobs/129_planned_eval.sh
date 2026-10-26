#!/bin/bash -l
#SBATCH -o /export/home/kaziz/g05/logs/slurm-%x-%j.out
#SBATCH -p gpu-all
#SBATCH --gres gpu:T4_16GB:2
#SBATCH -c 12
#SBATCH --mem 64000MB
#SBATCH --time 08:00:00
# High-level planner (Qwen3-VL-4B, 128_planner_server.py, GPU 1) + pi0.5 (planned_rc365_policy.py around pi05_rc365_policy.py,
# GPU 0) on Awomo's RoboCasa365 harness; the simulator worker renders on GPU 0. Several jobs may share OUT.
# Env: OUT (required), TASK_SET (composite_seen | composite_unseen | target50), NUM_TRIALS (2), TASKS, PLAN_EVERY (64)
set -u
G=/export/home/kaziz/g05; A=$G/baselines/Awomo-0.5; RP=$G/robocasa365/.venv/bin/python
CKPT=${CKPT:-$G/models/robocasa365_official/pi05_pretrain_human300/multitask_learning/75000}
: "${OUT:?}"; mkdir -p $OUT/logs
GPUS=(${CUDA_VISIBLE_DEVICES//,/ }); G0=${GPUS[0]}; G1=${GPUS[1]:-${GPUS[0]}}
PPORT=$((17000 + SLURM_JOB_ID % 1000)); PORT=$((18000 + SLURM_JOB_ID % 2000))
echo "node=$(hostname) gpus=$G0,$G1 out=$OUT set=${TASK_SET:-composite_seen}"
CUDA_VISIBLE_DEVICES=$G1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 $A/.venv/bin/python -u $G/jobs/128_planner_server.py --port $PPORT \
  > $OUT/logs/planner_$SLURM_JOB_ID.log 2>&1 &
PL=$!; trap 'kill $PL $SV 2>/dev/null' EXIT
for i in $(seq 1 120); do grep -q "listening on" $OUT/logs/planner_$SLURM_JOB_ID.log 2>/dev/null && break
  kill -0 $PL 2>/dev/null || { echo "planner died"; tail -20 $OUT/logs/planner_$SLURM_JOB_ID.log; exit 1; }; sleep 5; done
cd $A
CUDA_VISIBLE_DEVICES=$G0 XLA_PYTHON_CLIENT_PREALLOCATE=false HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 PLANNER_PORT=$PPORT \
  PLAN_EVERY=${PLAN_EVERY:-64} PLAN_LOG=$OUT/logs/plan_$SLURM_JOB_ID.jsonl INNER_POLICY=${INNER_POLICY:-pi05_rc365_policy} \
  PYTHONPATH=$A:$G/jobs $G/baselines/rcb-openpi/.venv/bin/python -u deploy/server.py --policy planned_rc365_policy --checkpoint $CKPT \
  --port $PORT > $OUT/logs/server_$SLURM_JOB_ID.log 2>&1 &
SV=$!
for i in $(seq 1 120); do grep -q "listening" $OUT/logs/server_$SLURM_JOB_ID.log 2>/dev/null && break
  kill -0 $SV 2>/dev/null || { echo "server died"; tail -20 $OUT/logs/server_$SLURM_JOB_ID.log; exit 1; }; sleep 10; done
export CUDA_VISIBLE_DEVICES=$G0 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0
if mkdir $OUT/.init_lock 2>/dev/null; then $RP eval_robocasa365/dynamic_eval.py init --out $OUT --task-set ${TASK_SET:-composite_seen} --trials ${NUM_TRIALS:-2} ${TASKS:+--tasks $TASKS}; fi
for i in $(seq 1 60); do [ -f $OUT/config.json ] && break; sleep 5; done
$RP -u eval_robocasa365/dynamic_eval.py worker --out $OUT --port $PORT > $OUT/logs/worker_$SLURM_JOB_ID.log 2>&1
$RP eval_robocasa365/dynamic_eval.py merge --out $OUT
