#!/bin/bash -l
#SBATCH -o /export/home/kaziz/g05/logs/slurm-%x-%j.out
#SBATCH -p gpu-all
#SBATCH --gres gpu:T4_16GB:2
#SBATCH -c 16
#SBATCH --mem 100000MB
#SBATCH --time 06:00:00
# pi0.7-inspired probes on LIBERO-Goal (one shared scene, 10 commands), served by gc_serve.py.
# Pairs: target t = 0..9, source s = (t+3) % 10 (the same held-out pairs as 19_steer_mem.sh); 10 paired init states per pair.
#   cond          init  prompt               goal                 CFG negative     w
#   ref           t     target               -                    -
#   switch        s     s for 2 s, then t    -                    -
#   cfg_old{15,2,3} s   s for 2 s, then t    -                    s (after 2 s)    1.5 / 2 / 3
#   cfg_blank{15,2,3} s s for 2 s, then t    -                    blank (after 2 s) 1.5 / 2 / 3
#   ref_cfgblank2 t     target               -                    blank            2
#   cfg_blank2_w{20,40} s s for 2 s, then t  -                    blank for 1 / 2 s after the switch, then none   2
#   ref_cfgblank2_w40 t  target              -                    blank from 2 s to 4 s only                      2
#   ref_g         t     target               t                    -
#   blank         t     blank                -                    -
#   gonly         t     blank                t                    -
#   switch_g      s     s for 2 s, then t    s for 2 s, then t    -
#   gswitch       s     blank                s for 2 s, then t    -
# Env: POLICY=name [GC_CKPT=...] [EXPERT_CKPT=...] [GATE=goal_eval.json] [GOAL_SOURCE=demo|gen] CONDS="..." [TRIALS=10 PAR=TRIALS MAXCONC=3 MAXB=20]
# TRIALS > 10 adds initial states 10.. (trial k uses init state k); keep PAR <= 10 so the client GPU holds the envs.
source /export/home/kaziz/g05/env_g05.sh
cd $PROJECT_ROOT
: "${POLICY:?}"
TRIALS=${TRIALS:-10}; PAR=${PAR:-$TRIALS}; SEED=${SEED:-7}; MAXCONC=${MAXCONC:-3}; GOAL_SOURCE=${GOAL_SOURCE:-demo}
OUT=$G05_ROOT/results/steer_gc/$POLICY; mkdir -p $OUT
PORT=$(( 20000 + SLURM_JOB_ID % 20000 ))
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONFAULTHANDLER=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
if [[ -n "${GATE:-}" ]]; then  # skip closed-loop evaluation when the offline check shows the goal content is unused
    $PY - "$GATE" <<'PYEOF' || { echo "gate: real goal no better than a shuffled one offline; evaluation skipped"; exit 0; }
import json, sys
r = json.load(open(sys.argv[1]))
gain = (r["blank_shuffled_goal"] - r["blank_goal"]) / r["blank_shuffled_goal"]
print(f"gate: blank-instruction FM loss, real goal {r['blank_goal']:.5f} vs shuffled {r['blank_shuffled_goal']:.5f} (gain {gain:+.2%})")
sys.exit(0 if gain > 0.02 else 1)
PYEOF
fi
GC_ARGS=(); [[ -n "${GC_CKPT:-}" ]] && GC_ARGS=(--gc_ckpt "$GC_CKPT")
[[ -n "${EXPERT_CKPT:-}" ]] && GC_ARGS+=(--expert_ckpt "$EXPERT_CKPT")
# With 2 GPUs the server gets the first and the clients (MuJoCo EGL contexts, ~0.3 GB per env) the second: on a single
# 16 GB T4, 30 envs' render contexts plus the fp16 server ran out of memory.
GPUS=(${CUDA_VISIBLE_DEVICES//,/ }); SRV_GPU=${GPUS[0]}; CLI_GPU=${GPUS[1]:-${GPUS[0]}}
echo "policy=$POLICY gc=${GC_CKPT:-none} expert=${EXPERT_CKPT:-released} goal_source=$GOAL_SOURCE node=$(hostname) server_gpu=$SRV_GPU client_gpu=$CLI_GPU conds=[$CONDS]"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
CUDA_VISIBLE_DEVICES=$SRV_GPU $PY $G05_ROOT/jobs/gc_serve.py --ckpt_path checkpoints/g05-libero/model.pt --host 0.0.0.0 --port $PORT \
    eval_embodiment=libero --action_steps 10 --max_batch_size ${MAXB:-20} --max_wait_ms 1000 --seed $SEED "${GC_ARGS[@]}" &> $OUT/server_${SLURM_JOB_ID}.log &
SPID=$!
for i in $(seq 1 300); do
    ss -tlnpH "sport = :$PORT" 2>/dev/null | grep -q "$PORT" && break
    kill -0 $SPID 2>/dev/null || { echo "server died"; tail -30 $OUT/server_${SLURM_JOB_ID}.log; exit 1; }
    sleep 2
done
spec() {  # cond t s -> "init|prompt|goal|cfg_neg|w"
    case $1 in
        ref) echo "$2|0:target|none|none|1" ;;
        switch) echo "$3|0:task$3,40:target|none|none|1" ;;
        cfg_old15) echo "$3|0:task$3,40:target|none|40:task$3|1.5" ;;
        cfg_old2) echo "$3|0:task$3,40:target|none|40:task$3|2" ;;
        cfg_old3) echo "$3|0:task$3,40:target|none|40:task$3|3" ;;
        cfg_blank15) echo "$3|0:task$3,40:target|none|40:blank|1.5" ;;
        cfg_blank2) echo "$3|0:task$3,40:target|none|40:blank|2" ;;
        cfg_blank3) echo "$3|0:task$3,40:target|none|40:blank|3" ;;
        cfg_blank2_w20) echo "$3|0:task$3,40:target|none|40:blank,60:none|2" ;;
        cfg_blank2_w40) echo "$3|0:task$3,40:target|none|40:blank,80:none|2" ;;
        ref_cfgblank2_w40) echo "$2|0:target|none|40:blank,80:none|2" ;;
        ref_cfgblank2) echo "$2|0:target|none|0:blank|2" ;;
        ref_g) echo "$2|0:target|0:target|none|1" ;;
        blank) echo "$2|0:blank|none|none|1" ;;
        gonly) echo "$2|0:blank|0:target|none|1" ;;
        switch_g) echo "$3|0:task$3,40:target|0:task$3,40:target|none|1" ;;
        gswitch) echo "$3|0:blank|0:task$3,40:target|none|1" ;;
        *) echo "unknown condition $1" >&2; return 1 ;;
    esac
}
run_one() {
    local c=$1 t=$2 s=$(( ($2 + 3) % 10 )) d=$OUT/${1}_t$2
    IFS='|' read init sched goal neg w <<< "$(spec $c $t $s)"
    mkdir -p $d
    for attempt in 1 2 3; do
        rm -f $d/*_parallel_results.json
        CUDA_VISIBLE_DEVICES=$CLI_GPU $PY $G05_ROOT/jobs/gc_steer_client.py --server_uri ws://127.0.0.1:$PORT --task_suite_name libero_goal \
            --task_id $t --init_task_id $init --prompt_schedule "$sched" --goal_schedule "$goal" --goal_source $GOAL_SOURCE \
            --cfg_neg_schedule "$neg" --cfg_w $w --num_trials_per_task $TRIALS --num_steps_wait 20 \
            --num_parallel $PAR --env_resolution 256 --seed $SEED --output_dir $d &> $d/client_attempt$attempt.log
        ls $d/*_parallel_results.json &>/dev/null && { echo "  done $c t$t"; return 0; }
        echo "  RETRY $c t$t ($(grep -m1 -oE 'Fatal Python error: [A-Za-z]+|[A-Za-z]+Error' $d/client_attempt$attempt.log))"
    done
    return 1
}
RUNNING=0
for c in $CONDS; do spec $c 0 3 > /dev/null || exit 1; done
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
        "schedule": task.get("prompt_schedule"), "goal_schedule": task.get("goal_schedule"), "goal_source": task.get("goal_source"),
        "cfg_neg_schedule": task.get("cfg_neg_schedule"), "cfg_w": task.get("cfg_w"), "trial_successes": task.get("trial_successes")}
for cond, v in summary.items():
    ok = [p for p in v["pairs"].values() if p]
    v["successes"] = sum(p["successes"] for p in ok)
    v["total"] = sum(p["total"] for p in ok)
    v["rate"] = v["successes"] / max(v["total"], 1)
    v["missing_pairs"] = sum(1 for p in v["pairs"].values() if not p)
    print(f"{cond:14s} {v['successes']:4d}/{v['total']:<4d} = {v['rate']:.3f}  missing_pairs={v['missing_pairs']}")
json.dump(summary, open(f"{out}/steer_summary.json", "w"), indent=2)
PYEOF
