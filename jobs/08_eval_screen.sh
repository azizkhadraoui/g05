#!/bin/bash -l
#SBATCH -J g05_screen
#SBATCH -o /export/home/kaziz/g05/logs/slurm-%x-%j.out
#SBATCH -A h200
#SBATCH --qos h200_qos
#SBATCH -p gpu-H200
#SBATCH --gres gpu:1
#SBATCH -c 64
#SBATCH --mem 220000MB
#SBATCH --time 06:00:00
# Paired screen: one policy server (released G0.5, or G0.5 + LaWAM adapter) + one LIBERO client per selected task.
# Same init states (trials 0..TRIALS-1), same env seed, same client settings as scripts/run/eval_libero.sh.
#   NAME=...  TASKS="libero_10:3 libero_goal:2 ..."  [HYBRID_CKPT=... BRANCH=future|current FUTURE_MODE=normal|current|shuffled|suppressed]
#   [TRIALS=20 PAR=10 SEED=7]
source /export/home/kaziz/g05/env_g05.sh
cd $PROJECT_ROOT
: "${NAME:?}" "${TASKS:?}"
TRIALS=${TRIALS:-20}; PAR=${PAR:-10}; SEED=${SEED:-7}; BRANCH=${BRANCH:-future}; FUTURE_MODE=${FUTURE_MODE:-normal}
(( PAR > TRIALS )) && PAR=$TRIALS   # eval_libero_parallel.py over-counts when num_parallel > num_trials
OUT=$G05_ROOT/results/screen/$NAME
mkdir -p $OUT
PORT=$(( 20000 + SLURM_JOB_ID % 20000 ))
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONFAULTHANDLER=1
echo "name=$NAME tasks=[$TASKS] trials=$TRIALS par=$PAR hybrid=${HYBRID_CKPT:-none} branch=$BRANCH future_mode=$FUTURE_MODE node=$(hostname) port=$PORT"
HY_ARGS=()
[[ -n "${HYBRID_CKPT:-}" ]] && HY_ARGS=(--hybrid_ckpt "$HYBRID_CKPT" --hybrid_branch "$BRANCH" --future_mode "$FUTURE_MODE")
t0=$(date +%s)
$PY $G05_ROOT/jobs/serve_policy_hybrid.py --ckpt_path checkpoints/g05-libero/model.pt --host 0.0.0.0 --port $PORT \
    eval_embodiment=libero --action_steps 10 --max_batch_size 30 --max_wait_ms 1000 --seed $SEED "${HY_ARGS[@]}" \
    &> $OUT/server.log &
SPID=$!
for i in $(seq 1 300); do
    ss -tlnpH "sport = :$PORT" 2>/dev/null | grep -q "$PORT" && break
    kill -0 $SPID 2>/dev/null || { echo "server died"; tail -30 $OUT/server.log; exit 1; }
    sleep 2
done
grep -E "^\[hybrid\]" $OUT/server.log
# One client per task, at most MAXCONC at a time, each retried up to 3x: MuJoCo's EGL read_pixels occasionally
# aborts a simulator worker under heavy parallel rendering (seen in jobs 426867 and 427059); a retry reruns only that task.
MAXCONC=${MAXCONC:-3}
run_task() {
    local t=$1 suite=${1%%:*} tid=${1##*:} d=$OUT/${1%%:*}_t${1##*:}
    mkdir -p $d
    for attempt in 1 2 3; do
        rm -f $d/*_parallel_results.json
        $PY experiments/libero/eval_libero_parallel.py --server_uri ws://127.0.0.1:$PORT --task_suite_name $suite \
            --task_id $tid --num_trials_per_task $TRIALS --num_steps_wait 20 --num_parallel $PAR --env_resolution 256 \
            --seed $SEED --output_dir $d &> $d/client_attempt${attempt}.log
        if ls $d/*_parallel_results.json &>/dev/null; then echo "  done $t (attempt $attempt)"; return 0; fi
        echo "  RETRY $t (attempt $attempt failed: $(grep -m1 -oE 'Fatal Python error: [A-Za-z]+|[A-Za-z]+Error' $d/client_attempt${attempt}.log))"
    done
    return 1
}
FAIL=0; RUNNING=0
for t in $TASKS; do
    run_task $t &
    RUNNING=$((RUNNING + 1))
    if (( RUNNING >= MAXCONC )); then wait -n || FAIL=1; RUNNING=$((RUNNING - 1)); fi
done
while (( RUNNING > 0 )); do wait -n || FAIL=1; RUNNING=$((RUNNING - 1)); done
kill $SPID 2>/dev/null
echo "wall=$(( $(date +%s) - t0 ))s"
$PY - "$OUT" "$TASKS" <<'PYEOF'
import glob, json, os, sys
out, tasks = sys.argv[1], sys.argv[2].split()
rows, tot_s, tot_n = [], 0, 0
for t in tasks:
    suite, tid = t.split(":")
    f = glob.glob(f"{out}/{suite}_t{tid}/*_parallel_results.json")
    if not f:
        rows.append({"task": t, "missing": True}); continue
    r = json.load(open(f[0]))
    for task in r["tasks"]:
        if int(task["task_id"]) != int(tid):
            continue
        rows.append({"task": t, "description": task["task_description"], "successes": task["successes"],
                     "total": task["total_episodes"], "trial_successes": task.get("trial_successes")})
        tot_s += task["successes"]; tot_n += task["total_episodes"]
summary = {"name": os.path.basename(out), "tasks": rows, "successes": tot_s, "total": tot_n,
           "rate": tot_s / max(tot_n, 1)}
json.dump(summary, open(f"{out}/screen_summary.json", "w"), indent=2)
for r in rows:
    print(r.get("task"), r.get("successes"), "/", r.get("total"), r.get("description", ""))
print(f"TOTAL {tot_s}/{tot_n} = {tot_s / max(tot_n, 1):.3f}")
PYEOF
grep -E "infer=|Batch timing" $OUT/server.log | tail -3
exit $FAIL
