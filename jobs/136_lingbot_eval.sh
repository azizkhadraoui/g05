#!/bin/bash -l
#SBATCH -o /export/home/kaziz/g05/logs/slurm-%x-%j.out
#SBATCH -p gpu-A100
#SBATCH -A a100
#SBATCH --qos a100_qos
#SBATCH --gres gpu:A100_80GB:1
#SBATCH -c 16
#SBATCH --mem 96000MB
#SBATCH --time 12:00:00
# LingBot-VLA 2.0 (RoboCasa365 post-training, lingbot_rc365_policy.py) on Awomo's RoboCasa365 harness: one policy server
# (LingBot venv, bf16) and one simulator worker (robocasa env) on this A100. Several jobs may share OUT (file queue).
# Env: OUT (required), CKPT (post-training hf_ckpt dir, required), TASK_SET (target50), NUM_TRIALS (2), TASKS
set -u
G=/export/home/kaziz/g05; A=$G/baselines/Awomo-0.5; RP=$G/robocasa365/.venv/bin/python; L=$G/baselines/lingbot-vla-v2
: "${OUT:?}"; : "${CKPT:?}"; mkdir -p $OUT/logs
PORT=$((18000 + SLURM_JOB_ID % 2000))
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 QWEN3VL_PATH=$G/models/Qwen3-VL-4B-Instruct
echo "node=$(hostname) ckpt=$CKPT out=$OUT port=$PORT"
# the harness server fingerprints the checkpoint by hashing its SHA256SUMS: write one for post-training checkpoints
[ -f $CKPT/SHA256SUMS ] || (cd $CKPT && sha256sum $(ls | grep -v SHA256SUMS) > SHA256SUMS.tmp && mv SHA256SUMS.tmp SHA256SUMS)
cd $A
PYTHONPATH=$A:$G/jobs $L/.venv/bin/python -u deploy/server.py --policy lingbot_rc365_policy --checkpoint $CKPT \
  --port $PORT > $OUT/logs/server_$SLURM_JOB_ID.log 2>&1 &
SPID=$!; trap 'kill $SPID 2>/dev/null' EXIT
for i in $(seq 1 120); do grep -q "listening" $OUT/logs/server_$SLURM_JOB_ID.log 2>/dev/null && break
  kill -0 $SPID 2>/dev/null || { echo "server died"; tail -20 $OUT/logs/server_$SLURM_JOB_ID.log; exit 1; }; sleep 10; done
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0
if mkdir $OUT/.init_lock 2>/dev/null; then $RP eval_robocasa365/dynamic_eval.py init --out $OUT --task-set ${TASK_SET:-target50} --trials ${NUM_TRIALS:-2} ${TASKS:+--tasks $TASKS}; fi
for i in $(seq 1 60); do [ -f $OUT/config.json ] && break; sleep 5; done
$RP -u eval_robocasa365/dynamic_eval.py worker --out $OUT --port $PORT > $OUT/logs/worker_$SLURM_JOB_ID.log 2>&1
tail -3 $OUT/logs/server_$SLURM_JOB_ID.log
$RP eval_robocasa365/dynamic_eval.py merge --out $OUT
