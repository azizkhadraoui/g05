#!/bin/bash -l
#SBATCH -o /export/home/kaziz/g05/logs/slurm-%x-%j.out
#SBATCH -p gpu-all
#SBATCH --gres gpu:T4_16GB:1
#SBATCH -c 8
#SBATCH --mem 64000MB
#SBATCH --time 12:00:00
# RoboJEPA planning its own actions in RoboCasa with the authors' reproduction scripts
# (wm/robo_jepa/scripts/deployment/examples/reproduce_<task>.sh: CEM over action sequences in latent space toward a goal
# image, the env server executes the chosen end-effector targets, one .mp4 + .npz per episode).
# Adapted without editing the scripts: their model-dir layout (wm-scaling-<size>-norms) points at our released
# checkpoints, the venv stands in for conda, the RoboCasa kitchen assets come from our robocasa365 checkout (same
# release line), the job's own GPU, and NB candidate batches per CEM iteration (theirs: 64 x 16 = 1024; NB=32 -> 512).
# Env: TASK (reach | object_reach | obstacle_reach), SIZE (22m), STEP (150k), SEEDS ("0 1"), NB (32), RECORD (0|1),
#      PLANNER (unset = CEM; grad | cemgrad = 118_rj_gradplan_client.py, results in results/robojepa_plan_<planner>)
set -u
G=/export/home/kaziz/g05; REPO=$G/wm/robo_jepa
TASK=${TASK:-reach}; SIZE=${SIZE:-22m}; STEP=${STEP:-150k}; SEEDS=${SEEDS:-0 1}; NB=${NB:-32}
export TMPDIR=$G/tmp/plan_$SLURM_JOB_ID; mkdir -p $TMPDIR
M=$G/wm/checkpoints/robojepa_plan/wm-scaling-$SIZE-norms; mkdir -p $M
ln -sfn $G/wm/checkpoints/robojepa/wm-$SIZE/config.json $M/config.json
ln -sfn $G/wm/checkpoints/robojepa/wm-$SIZE/cooldown_${SIZE}_${STEP}.pth.tar $M/cooldown_${SIZE}_${STEP}.pth.tar
ln -sfn $G/wm/checkpoints/robojepa/vjepa2_1_vitG_384.pt $M/encoder.pth.tar
SP=$REPO/.venv/lib/python3.11/site-packages/robocasa/models
if [ ! -L $SP/assets ]; then mv $SP/assets $SP/assets_pkg 2>/dev/null; ln -s $G/robocasa365/robocasa/robocasa/models/assets $SP/assets; fi
printf 'conda() { source %s/.venv/bin/activate; }\n' $REPO > $TMPDIR/conda.sh
sed "s/--num_batches 64/--num_batches $NB/; s#/tmp/#$TMPDIR/#g" $REPO/scripts/deployment/examples/reproduce_$TASK.sh > $TMPDIR/run.sh
if [ -n "${PLANNER:-}" ]; then  # gradient planning through the world model instead of CEM (118_rj_gradplan_client.py)
  sed -i "s#python policy_client.py#python $G/jobs/118_rj_gradplan_client.py#; s#--use_cache ##" $TMPDIR/run.sh
  export PLANNER
fi
if [ "${RECORD:-0}" = 1 ]; then  # full-motion recording of the executed sub-steps (105_rj_server_record.py)
  sed -i "s#python scripts/deployment/robots/robocasa/robot_env_server.py#python $G/jobs/105_rj_server_record.py#" $TMPDIR/run.sh
  export REC_DIR=$G/results/robojepa_plan/recordings/${TASK}_${SIZE}_job$SLURM_JOB_ID
fi
export REPO MODELS=$G/wm/checkpoints/robojepa_plan CONDA=robo_jepa CONDA_SH=$TMPDIR/conda.sh OUT=$G/results/robojepa_plan${PLANNER:+_$PLANNER}${GP_LR:+_lr$GP_LR} \
       PORT=$((7300 + SLURM_JOB_ID % 1000)) MASTER_PORT=$((29500 + SLURM_JOB_ID % 400)) GPU=${CUDA_VISIBLE_DEVICES:-0}
echo "task=$TASK size=$SIZE step=$STEP seeds=[$SEEDS] candidates=$((NB * 16)) node=$(hostname) gpu=$GPU"
bash $TMPDIR/run.sh $SIZE $STEP $SEEDS
ls -la $OUT/${TASK}_repro/*/ 2>/dev/null
