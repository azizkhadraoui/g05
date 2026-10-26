#!/bin/bash -l
#SBATCH -o /export/home/kaziz/g05/logs/slurm-%x-%j.out
#SBATCH -p gpu-all
#SBATCH --gres gpu:v100nv_32GB:1
#SBATCH -c 8
#SBATCH --mem 64000MB
#SBATCH --time 12:00:00
# Awomo-0.5I (as released) on RoboCasa365 with its own harness (eval_robocasa365/dynamic_eval.py): NSERV policy servers on
# this GPU (awomo env, one client each) and one simulator worker per server (robocasa env, EGL rendering on the same GPU).
# The output directory is resumable: rerunning with the same OUT continues unfinished episodes.
# Env: OUT (required), TASK_SET (target50 | atomic_seen | composite_seen | composite_unseen), NUM_TRIALS (30), TASKS,
#      NSERV (1), AWOMO_DTYPE (bf16 default; see 81_awomo_fp16.py)
set -u
G=/export/home/kaziz/g05; A=$G/baselines/Awomo-0.5; RP=$G/robocasa365/.venv/bin/python
OUT=${OUT:?set OUT}; mkdir -p $OUT/logs
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4
NS=${NSERV:-1}; BASE=$((18000 + (SLURM_JOB_ID % 500) * 4))  # distinct ports per job (jobs may share a node)
echo "node=$(hostname) gpu=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader) nserv=$NS out=$OUT dtype=${AWOMO_DTYPE:-bf16}"
cd $A
SP=()
for ((i = 0; i < NS; i++)); do
  PYTHONPATH=$A:$G/jobs $A/.venv/bin/python -u ${AWOMO_LAUNCH:-deploy/server.py} --policy eval_robocasa365.awomo_05i.policy \
    --checkpoint $G/models/Awomo-0.5I-RoboCasa365 --qwen-path $G/models/Qwen3-VL-4B-Instruct \
    --vae-path $G/models/FLUX.2-klein-base-4B/vae --port $((BASE + i)) > $OUT/logs/server_${SLURM_JOB_ID}_$i.log 2>&1 &
  SP+=($!)
done
trap 'kill ${SP[@]} 2>/dev/null' EXIT
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0
# init only once per OUT (several jobs may share it): a second init would requeue episodes another job is running
if mkdir $OUT/.init_lock 2>/dev/null; then $RP eval_robocasa365/dynamic_eval.py init --out $OUT --task-set ${TASK_SET:-target50} --trials ${NUM_TRIALS:-30} ${TASKS:+--tasks $TASKS}; fi
for i in $(seq 1 60); do [ -f $OUT/config.json ] && break; sleep 5; done
WP=()
for ((i = 0; i < NS; i++)); do
  $RP -u eval_robocasa365/dynamic_eval.py worker --out $OUT --port $((BASE + i)) > $OUT/logs/worker_${SLURM_JOB_ID}_$i.log 2>&1 &
  WP+=($!)
done
( while sleep 300; do nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader; done ) &
for p in ${WP[@]}; do wait $p; done
tail -3 $OUT/logs/server_${SLURM_JOB_ID}_0.log
$RP eval_robocasa365/dynamic_eval.py merge --out $OUT
