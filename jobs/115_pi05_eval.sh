#!/bin/bash -l
#SBATCH -o /export/home/kaziz/g05/logs/slurm-%x-%j.out
#SBATCH -p gpu-all
#SBATCH --gres gpu:T4_16GB:1
#SBATCH -c 8
#SBATCH --mem 40000MB
#SBATCH --time 05:00:00
# pi0.5 (official RoboCasa365 multitask checkpoint) on Awomo's RoboCasa365 harness: one policy server (rcb-openpi venv,
# JAX, pi05_rc365_policy.py) and one simulator worker (robocasa env) on this GPU. Several jobs may share OUT: the harness
# queue is file based. Env: OUT (required), CKPT (default: the official checkpoint), TASK_SET (target50), NUM_TRIALS (2), TASKS
set -u
G=/export/home/kaziz/g05; A=$G/baselines/Awomo-0.5; RP=$G/robocasa365/.venv/bin/python
CKPT=${CKPT:-$G/models/robocasa365_official/pi05_pretrain_human300/multitask_learning/75000}
: "${OUT:?}"; mkdir -p $OUT/logs
[ -f $CKPT/SHA256SUMS ] || (cd $CKPT && sha256sum _CHECKPOINT_METADATA assets/norm_stats.json > SHA256SUMS)
PORT=$((18000 + SLURM_JOB_ID % 2000))
# JAX: allocate on demand (the simulator renders with EGL on the same GPU)
export XLA_PYTHON_CLIENT_PREALLOCATE=false HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4
echo "node=$(hostname) ckpt=$CKPT out=$OUT port=$PORT"
cd $A
PYTHONPATH=$A:$G/jobs $G/baselines/rcb-openpi/.venv/bin/python -u deploy/server.py --policy pi05_rc365_policy --checkpoint $CKPT \
  --port $PORT > $OUT/logs/server_$SLURM_JOB_ID.log 2>&1 &
SPID=$!; trap 'kill $SPID 2>/dev/null' EXIT
for i in $(seq 1 120); do grep -q "listening" $OUT/logs/server_$SLURM_JOB_ID.log 2>/dev/null && break
  kill -0 $SPID 2>/dev/null || { echo "server died"; tail -20 $OUT/logs/server_$SLURM_JOB_ID.log; exit 1; }; sleep 10; done
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0
# init only once per OUT: a second init would move episodes another job is running back to pending (duplicates)
if mkdir $OUT/.init_lock 2>/dev/null; then $RP eval_robocasa365/dynamic_eval.py init --out $OUT --task-set ${TASK_SET:-target50} --trials ${NUM_TRIALS:-2} ${TASKS:+--tasks $TASKS}; fi
for i in $(seq 1 60); do [ -f $OUT/config.json ] && break; sleep 5; done
$RP -u eval_robocasa365/dynamic_eval.py worker --out $OUT --port $PORT > $OUT/logs/worker_$SLURM_JOB_ID.log 2>&1
tail -3 $OUT/logs/server_$SLURM_JOB_ID.log
$RP eval_robocasa365/dynamic_eval.py merge --out $OUT
