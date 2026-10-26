#!/bin/bash -l
#SBATCH -o /export/home/kaziz/g05/logs/slurm-%x-%j.out
#SBATCH -p gpu-all
#SBATCH --gres gpu:T4_16GB:1
#SBATCH -c 8
#SBATCH --mem 40000MB
#SBATCH --time 05:00:00
# GR00T N1.7 (RoboCasa365 fine-tune) on Awomo's RoboCasa365 harness: one policy server (GR00T venv, gr00t_rc365_policy.py)
# and one simulator worker (robocasa env) on this GPU. Several jobs may share OUT: the harness queue is file based and
# every worker claims its own episodes. Env: CKPT (checkpoint dir, or a training output dir: its latest checkpoint-N),
# OUT (required), TASK_SET (target50), NUM_TRIALS (2), TASKS, WAIT_CKPT (1 = wait for the first checkpoint)
set -u
G=/export/home/kaziz/g05; A=$G/baselines/Awomo-0.5; GR=$G/baselines/Isaac-GR00T; RP=$G/robocasa365/.venv/bin/python
: "${CKPT:?}"; : "${OUT:?}"; mkdir -p $OUT/logs
# CKPT may be the training output dir: use its latest checkpoint-N (WAIT_CKPT=1: wait for the first one to be written)
if [ "${WAIT_CKPT:-0}" = 1 ]; then
  for i in $(seq 1 240); do ls -d $CKPT/checkpoint-* > /dev/null 2>&1 && break; sleep 60; done; sleep 180
fi
last=$(ls -d $CKPT/checkpoint-* 2> /dev/null | sort -V | tail -1); [ -n "$last" ] && CKPT=$last
[ -f $CKPT/SHA256SUMS ] || (cd $CKPT && find . -maxdepth 1 -type f ! -name SHA256SUMS -name "*.json" -exec sha256sum {} \; > SHA256SUMS)
PORT=$((18000 + SLURM_JOB_ID % 2000))
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 CUDA_HOME=/cm/shared/apps/cuda12.6/toolkit/12.6.2
echo "node=$(hostname) ckpt=$CKPT out=$OUT port=$PORT"
cd $A
PYTHONPATH=$A:$G/jobs $GR/.venv/bin/python -u deploy/server.py --policy gr00t_rc365_policy --checkpoint $CKPT --port $PORT \
  > $OUT/logs/server_$SLURM_JOB_ID.log 2>&1 &
SPID=$!; trap 'kill $SPID 2>/dev/null' EXIT
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0
# init only once per OUT: a second init would move episodes another job is running back to pending (duplicates)
if mkdir $OUT/.init_lock 2>/dev/null; then $RP eval_robocasa365/dynamic_eval.py init --out $OUT --task-set ${TASK_SET:-target50} --trials ${NUM_TRIALS:-2} ${TASKS:+--tasks $TASKS}; fi
for i in $(seq 1 60); do [ -f $OUT/config.json ] && break; sleep 5; done
$RP -u eval_robocasa365/dynamic_eval.py worker --out $OUT --port $PORT > $OUT/logs/worker_$SLURM_JOB_ID.log 2>&1
tail -3 $OUT/logs/server_$SLURM_JOB_ID.log
$RP eval_robocasa365/dynamic_eval.py merge --out $OUT
