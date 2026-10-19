#!/bin/bash -l
#SBATCH -J g05_collect
#SBATCH -o /export/home/kaziz/g05/logs/slurm-%x-%j.out
#SBATCH -A h200
#SBATCH --qos h200_qos
#SBATCH -p gpu-H200
#SBATCH --gres gpu:1
#SBATCH -c 64
#SBATCH --mem 220000MB
#SBATCH --time 06:00:00
# Steerability data: released G0.5 on LIBERO-Goal, command task s for 2 s then target t; keep successful episodes
# (post-switch segment). Training pairs only: s = t+1, t+2, t+4, t+5 (mod 10). Evaluation pairs (s = t+3) never used.
source /export/home/kaziz/g05/env_g05.sh
cd $PROJECT_ROOT
TRIALS=${TRIALS:-20}; SEED=7; MAXCONC=${MAXCONC:-3}
OUT=$G05_ROOT/results/recovery; mkdir -p $OUT
PORT=$(( 20000 + SLURM_JOB_ID % 20000 ))
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONFAULTHANDLER=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
$PY $G05_ROOT/jobs/serve_policy_hybrid.py --ckpt_path checkpoints/g05-libero/model.pt --host 0.0.0.0 --port $PORT \
    eval_embodiment=libero --action_steps 10 --max_batch_size ${MAXB:-30} --max_wait_ms 1000 --seed $SEED &> $OUT/server_${SLURM_JOB_ID}.log &
SPID=$!
for i in $(seq 1 300); do ss -tlnpH "sport = :$PORT" 2>/dev/null | grep -q "$PORT" && break; kill -0 $SPID 2>/dev/null || exit 1; sleep 2; done
run_one() {
    local t=$1 s=$2 d=$OUT/t$1_s$2
    mkdir -p $d
    for attempt in 1 2 3; do
        rm -rf $d/rec $d/*_parallel_results.json
        $PY $G05_ROOT/jobs/18_libero_steer_client.py --server_uri ws://127.0.0.1:$PORT --task_suite_name libero_goal \
            --task_id $t --init_task_id $s --prompt_schedule "0:task$s,40:target" --num_trials_per_task $TRIALS \
            --num_steps_wait 20 --num_parallel 10 --env_resolution 256 --seed $SEED --output_dir $d \
            --record_dir $d/rec --record_from 40 &> $d/client_attempt$attempt.log
        ls $d/*_parallel_results.json &>/dev/null && { echo "  done t$t s$s: $(ls $d/rec 2>/dev/null | wc -l) episodes"; return 0; }
        echo "  RETRY t$t s$s"
    done
}
RUNNING=0
for t in ${TARGETS:-0 1 2 3 4 5 6 7 8 9}; do for off in 1 2 4 5; do
    run_one $t $(( (t + off) % 10 )) &
    RUNNING=$((RUNNING + 1)); if (( RUNNING >= MAXCONC )); then wait -n; RUNNING=$((RUNNING - 1)); fi
done; done
while (( RUNNING > 0 )); do wait -n; RUNNING=$((RUNNING - 1)); done
kill $SPID 2>/dev/null
echo "TOTAL recovery episodes: $(ls $OUT/*/rec/*.npz 2>/dev/null | wc -l)"; du -sh $OUT
