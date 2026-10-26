#!/bin/bash -l
#SBATCH -o /export/home/kaziz/g05/logs/slurm-%x-%j.out
#SBATCH -p gpu-all
#SBATCH --gres gpu:T4_16GB:2
#SBATCH -c 12
#SBATCH --mem 80000MB
#SBATCH --time 05:00:00
# End-to-end mission benchmark (125_e2e_missions.py). GPU 0: G0.5 server with the steering modes (94_frs_serve.py, 32-step
# chunks; the client executes 10, or 32 in the long_chunks arm). GPU 1: RoboBrain planner, world model, DINOv3, memory.
# Env: ARM (full | no_planner | no_memory | no_steering | no_wm | long_chunks), EXTRA (more client args), PAR (3),
#      OUTD (results root, default results/e2e), TAG (output subdirectory, default ARM),
#      RUNS="tag|client args;tag|client args" runs several configurations one after another against this server,
#      DSBC_CKPT (noise policy for --steer dsbc, read by the server),
#      SCRIPT (client, default 125_e2e_missions.py; 130_memory_missions.py takes ARM=event | transcript | view)
source /export/home/kaziz/g05/env_g05.sh
cd $PROJECT_ROOT
D=${OUTD:-$G05_ROOT/results/e2e}; mkdir -p $D
PORT=$(( 20000 + SLURM_JOB_ID % 20000 ))
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONFAULTHANDLER=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "node=$(hostname) arm=${ARM:?} gpus=$CUDA_VISIBLE_DEVICES out=$D dsbc=${DSBC_CKPT:-none}"
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES%%,*} $PY $G05_ROOT/jobs/94_frs_serve.py --ckpt_path checkpoints/g05-libero/model.pt --host 0.0.0.0 \
    --port $PORT eval_embodiment=libero --action_steps 32 --max_batch_size 10 --max_wait_ms 200 --seed 7 &> $D/server_$SLURM_JOB_ID.log &
SPID=$!; trap 'kill $SPID 2>/dev/null' EXIT
for i in $(seq 1 300); do
    grep -q "server listening on" $D/server_$SLURM_JOB_ID.log 2>/dev/null && break
    kill -0 $SPID 2>/dev/null || { echo "server died"; tail -30 $D/server_$SLURM_JOB_ID.log; exit 1; }
    sleep 2
done
CLIENT="$PY -u $G05_ROOT/jobs/${SCRIPT:-125_e2e_missions.py} --server_uri ws://127.0.0.1:$PORT --arm $ARM --num_parallel ${PAR:-3} --output_dir $D"
if [ -n "${RUNS:-}" ]; then
    IFS=';' read -ra RR <<< "$RUNS"; rc=0
    for r in "${RR[@]}"; do
        echo "== run ${r%%|*}: ${r#*|}"
        $CLIENT --tag ${r%%|*} ${r#*|} || rc=1
    done
    exit $rc
fi
$CLIENT --tag ${TAG:-$ARM} ${EXTRA:-}
