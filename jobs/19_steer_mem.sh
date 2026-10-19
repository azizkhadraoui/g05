#!/bin/bash -l
#SBATCH -J g05_steer
#SBATCH -o /export/home/kaziz/g05/logs/slurm-%x-%j.out
#SBATCH -A h200
#SBATCH --qos h200_qos
#SBATCH -p gpu-H200
#SBATCH --gres gpu:1
#SBATCH -c 64
#SBATCH --mem 220000MB
#SBATCH --time 08:00:00
# Steering (S02) and memory (M02-style) probes on LIBERO-Goal (one shared scene, 10 commands).
# Pairs: target t = 0..9, source s = (t+3) % 10; 10 paired init states per pair (init states of the scene's task).
#   ref        init t, "0:target"                          (own scene, own command)
#   cf         init s, "0:target"                          S02 counterfactual command from an identical scene
#   switch     init s, "0:task_s,40:target"                S02 mid-execution command switch (= revision with correct memory)
#   m_none     init t, "0:target,40:blank"                 instruction withdrawn after 2 s, no memory
#   m_stale    init t, "0:target,40:task_s"                stale memory re-supplied
#   rev_none   init s, "0:task_s,40:target,80:blank"       revision then withdrawal, no memory
#   rev_stale  init s, "0:task_s,40:target,80:task_s"      revision then stale (pre-revision) memory
# Env: POLICY=name [HYBRID_CKPT=... BRANCH=...] [TRIALS=10 MAXCONC=3 CONDS="ref cf ..."]
source /export/home/kaziz/g05/env_g05.sh
cd $PROJECT_ROOT
: "${POLICY:?}"
TRIALS=${TRIALS:-10}; PAR=$TRIALS; SEED=${SEED:-7}; BRANCH=${BRANCH:-future}; MAXCONC=${MAXCONC:-3}
CONDS=${CONDS:-"ref cf switch m_none m_stale rev_none rev_stale"}
OUT=$G05_ROOT/results/steer/$POLICY; mkdir -p $OUT
PORT=$(( 20000 + SLURM_JOB_ID % 20000 ))
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONFAULTHANDLER=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
HY_ARGS=(); [[ -n "${HYBRID_CKPT:-}" ]] && HY_ARGS=(--hybrid_ckpt "$HYBRID_CKPT" --hybrid_branch "$BRANCH")
echo "policy=$POLICY hybrid=${HYBRID_CKPT:-none} node=$(hostname) conds=[$CONDS]"
$PY $G05_ROOT/jobs/serve_policy_hybrid.py --ckpt_path checkpoints/g05-libero/model.pt --host 0.0.0.0 --port $PORT \
    eval_embodiment=libero --action_steps 10 --max_batch_size ${MAXB:-30} --max_wait_ms 1000 --seed $SEED "${HY_ARGS[@]}" &> $OUT/server_${SLURM_JOB_ID}.log &
SPID=$!
for i in $(seq 1 300); do
    ss -tlnpH "sport = :$PORT" 2>/dev/null | grep -q "$PORT" && break
    kill -0 $SPID 2>/dev/null || { echo "server died"; tail -30 $OUT/server_${SLURM_JOB_ID}.log; exit 1; }
    sleep 2
done
spec() {  # cond t s -> "init schedule"
    case $1 in
        ref) echo "$2 0:target" ;;
        cf) echo "$3 0:target" ;;
        switch) echo "$3 0:task$3,40:target" ;;
        m_none) echo "$2 0:target,40:blank" ;;
        m_stale) echo "$2 0:target,40:task$3" ;;
        rev_none) echo "$3 0:task$3,40:target,80:blank" ;;
        rev_stale) echo "$3 0:task$3,40:target,80:task$3" ;;
    esac
}
run_one() {
    local c=$1 t=$2 s=$(( ($2 + 3) % 10 )) d=$OUT/${1}_t$2
    read init sched <<< "$(spec $c $t $s)"
    mkdir -p $d
    for attempt in 1 2 3; do
        rm -f $d/*_parallel_results.json
        $PY $G05_ROOT/jobs/18_libero_steer_client.py --server_uri ws://127.0.0.1:$PORT --task_suite_name libero_goal \
            --task_id $t --init_task_id $init --prompt_schedule "$sched" --num_trials_per_task $TRIALS --num_steps_wait 20 \
            --num_parallel $PAR --env_resolution 256 --seed $SEED --output_dir $d &> $d/client_attempt$attempt.log
        ls $d/*_parallel_results.json &>/dev/null && { echo "  done $c t$t"; return 0; }
        echo "  RETRY $c t$t ($(grep -m1 -oE 'Fatal Python error: [A-Za-z]+|[A-Za-z]+Error' $d/client_attempt$attempt.log))"
    done
    return 1
}
RUNNING=0
for c in $CONDS; do for t in ${TARGETS:-0 1 2 3 4 5 6 7 8 9}; do
    run_one $c $t &
    RUNNING=$((RUNNING + 1))
    if (( RUNNING >= MAXCONC )); then wait -n; RUNNING=$((RUNNING - 1)); fi
done; done
while (( RUNNING > 0 )); do wait -n; RUNNING=$((RUNNING - 1)); done
kill $SPID 2>/dev/null
$PY - "$OUT" <<'PYEOF'
import glob, json, os, sys
out = sys.argv[1]
summary = {}
for d in sorted(glob.glob(f"{out}/*_t*")):
    cond, t = os.path.basename(d).rsplit("_t", 1)
    f = glob.glob(f"{d}/*_parallel_results.json")
    if not f:
        summary.setdefault(cond, {"pairs": {}})["pairs"][t] = None
        continue
    task = json.load(open(f[0]))["tasks"][0]
    summary.setdefault(cond, {"pairs": {}})["pairs"][t] = {
        "successes": task["successes"], "total": task["total_episodes"], "init_task_id": task.get("init_task_id"),
        "schedule": task.get("prompt_schedule"), "trial_successes": task.get("trial_successes")}
for cond, v in summary.items():
    ok = [p for p in v["pairs"].values() if p]
    v["successes"] = sum(p["successes"] for p in ok)
    v["total"] = sum(p["total"] for p in ok)
    v["rate"] = v["successes"] / max(v["total"], 1)
    v["missing_pairs"] = sum(1 for p in v["pairs"].values() if not p)
    print(f"{cond:10s} {v['successes']:4d}/{v['total']:<4d} = {v['rate']:.3f}  missing_pairs={v['missing_pairs']}")
json.dump(summary, open(f"{out}/steer_summary.json", "w"), indent=2)
PYEOF
