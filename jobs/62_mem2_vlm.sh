#!/bin/bash -l
#SBATCH -J g05_mem2vlm
#SBATCH -o /export/home/kaziz/g05/logs/slurm-%x-%j.out
#SBATCH -p gpu-all
#SBATCH --gres gpu:T4_16GB:2
#SBATCH -c 16
#SBATCH --mem 100000MB
#SBATCH --time 06:00:00
# RoboBrain as the MEM-2 memory module (59_vlm_memclient.py). GPU 0: G0.5 server, GPU 1: RoboBrain.
# Conditions: vlm_multi (5 frames over 2 s), vlm_current (current frame), rev_vlm_multi / rev_vlm_current (after a revision).
# Env: POLICY (output name), VLM (checkpoint dir), CONDS, TRIALS=10, MAXCONC=1, MAXB=10
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
CONDS=${CONDS:-"m_none mem_current mem_first mem_offsets mem_snapshot rev_none rev_mem_current rev_mem_offsets rev_mem_snapshot"}
DEC=${DEC:-$G05_ROOT/results/mem_intent}
OUT=$G05_ROOT/results/mem2/$POLICY; mkdir -p $OUT
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
        vlm_*) echo "$2 0:target,40:memory" ;;
        rev_vlm_*) echo "$3 0:task$3,40:target,80:memory" ;;
    esac
}
run_one() {
    local c=$1 t=$2 s=$(( ($2 + 3) % 10 )) d=$OUT/${1}_t$2
    read init sched <<< "$(spec $c $t $s)"
    local MARGS=""
    case $c in *vlm_multi) MARGS="--memory_vlm $VLM --vlm_frames multi" ;; *vlm_current) MARGS="--memory_vlm $VLM --vlm_frames current" ;; esac
    mkdir -p $d
    for attempt in 1 2 3; do
        rm -f $d/*_parallel_results.json
        $PY $G05_ROOT/jobs/59_vlm_memclient.py --server_uri ws://127.0.0.1:$PORT --task_suite_name libero_goal \
            --task_id $t --init_task_id $init --prompt_schedule "$sched" --num_trials_per_task $TRIALS --num_steps_wait 20 \
            --num_parallel $PAR --env_resolution 256 --seed $SEED --output_dir $d $MARGS &> $d/client_attempt$attempt.log
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
        "schedule": task.get("prompt_schedule"), "trial_successes": task.get("trial_successes"),
        "trial_inferred": task.get("trial_inferred"), "target": task.get("task_description")}
for cond, v in summary.items():
    ok = [p for p in v["pairs"].values() if p]
    v["successes"] = sum(p["successes"] for p in ok)
    v["total"] = sum(p["total"] for p in ok)
    v["rate"] = v["successes"] / max(v["total"], 1)
    v["missing_pairs"] = sum(1 for p in v["pairs"].values() if not p)
    nrm = lambda x: " ".join(str(x).lower().split())
    inf = [(nrm(i) == nrm(p["target"])) for p in ok for i in (p.get("trial_inferred") or {}).values()]
    v["memory_inference_acc"] = (sum(inf) / len(inf)) if inf else None
    print(f"{cond:16s} {v['successes']:4d}/{v['total']:<4d} = {v['rate']:.3f}  missing_pairs={v['missing_pairs']}  memory_inference_acc={v['memory_inference_acc']}")
json.dump(summary, open(f"{out}/mem2_summary.json", "w"), indent=2)
PYEOF
