#!/bin/bash -l
#SBATCH -o /export/home/kaziz/g05/logs/slurm-%x-%j.out
#SBATCH -p gpu-all
#SBATCH --gres gpu:T4_16GB:2
#SBATCH -c 8
#SBATCH --mem 36000MB
#SBATCH --time 06:00:00
# Flow Reversal Steering on the LIBERO-Goal command-switch probe (same pairs and initial states as gc_steer.sh):
# target t = 0..9, source s = (t+3) % 10; the robot starts on task s, the command switches to t at policy step 40 (2 s).
# During steps [40, 80) every replan carries a coarse reference (straight-line approach to the new task's contact point,
# 95_frs_client.py) that the server (94_frs_serve.py) uses as below; afterwards plain G0.5 with the new command.
#   cond      init  prompt               reference use in [40, 80)
#   ref       t     target               -                       (normal episodes: precision baseline)
#   switch    s     s for 2 s, then t    -                       (G0.5 alone)
#   frs_ref   s     s for 2 s, then t    execute the reference directly (no policy)
#   frs_b{2,5,10} s s for 2 s, then t    FRS: K backward Euler steps to noise, 10 forward
#   sde_{06,08}  s  s for 2 s, then t    partial noising to t0 = 0.6 / 0.8, then forward
#   ref_frs_b5   t  target               FRS b5 in [40, 80) on normal episodes (precision cost)
#   ref_frs_ref  t  target               reference executed directly in [40, 80) on normal episodes (precision cost)
#   frs_ts{03,05}  FRS inverting only to t = 0.3 / 0.5 (keeps more of the reference), then forward from there
#   sde_{02,03}    partial noising to t0 = 0.2 / 0.3
#   guide_l{1,3}   plain sampling from noise with reference guidance lam = 1 / 3 (v += lam (x0_hat - ref))
#   frs_cfg, frs_ts05_cfg  FRS (full / to 0.5) with classifier-free guidance away from the old command (w = 2)
#   frs_ref_n{3,6,10}  raw 2 s nudge with a fixed random error (std 3 / 6 / 10 cm) on the target point per trial
#                (robustness to a human click or VLM pointing instead of the simulator's true contact point)
#   frs_ref_rec  raw nudge on initial states 0..TRIALS-1, every window request recorded with its exact noise (DSBC data)
#   frs_exact    FRS with EXACT inversion of the nudge through the 10-step sampler (should reproduce the nudge)
#   dsbc_ho      DSBC noise policy (DSBC_CKPT) in the window, no reference, held-out initial states 30..39
#   switch_ho / frs_ref_ho  baselines on the same held-out states; dsbc_tr = DSBC on training states 0..9
#   frs_ref_rec5 as frs_ref_rec, records tagged frs_ref_rec5 (DSBC data for the second pair set, PAIR_SHIFT=5)
#   dsbc2_ho     as dsbc_ho (the checkpoint is DSBC_CKPT), result dirs kept apart from the first DSBC policy's
#   smsteer_ho   the SuperMimic-style noise policy (119_smsteer.py, DSBC_CKPT) on held-out states 30..39
#   smsteer_rec  the same policy on training states, window requests recorded (on-policy data for the next Dyna round)
#   smsteer_rec5 the same for the second pair set (PAIR_SHIFT=5; tag kept apart because trial ids repeat across sets)
#   smsteer2_ho  the second-round SuperMimic-style policy (refit on its own on-policy data) on held-out states 30..39
#   smsteer2b_ho as smsteer2_ho, with an early-stopped critic and a tighter trust region (DSBC_CKPT=policy_sm2b.pt)
#   smsteer3_rec(5) round-2b policy on FRESH start states 40..49 (never used for training or evaluation), recorded
#   smsteer3_ho  round-3 policy (refit with the fresh data) on held-out states 30..39
#   smpol_ho     command switch with no steering module, G0.5's action expert trained SuperMimic-style (EXPERT_CKPT;
#                119_smsteer.py policy), held-out states 30..39; smpol_ref_ho: same expert, own command (regression check)
#   ref_ho       plain G0.5 on its own command, held-out states 30..39 (baseline for smpol_ref_ho)
#   smpol2_ho, smpol2_ref_ho  as smpol_*, second expert (tighter anchor; EXPERT_CKPT=expert_smpol2.pt)
#   zero_ho      zero starting noise in the window (DSBC_CKPT = a noise policy with its output layer zeroed): control for the noise policies
#   PAIR_SHIFT   source task s = (t + PAIR_SHIFT) % 10 (default 3; 5 = a second, disjoint set of command switches)
#   *_arr        as above, but the reference stays on from step 40 until the arm is within 5 cm of the target point
#                ("point until the robot gets there"), then plain G0.5 with the new command
# Env: POLICY=name CONDS="..." [TRIALS=10 PAR=TRIALS MAXCONC=3 MAXB=20 TARGETS="0 .. 9"]
source /export/home/kaziz/g05/env_g05.sh
cd $PROJECT_ROOT
: "${POLICY:?}"
TRIALS=${TRIALS:-10}; PAR=${PAR:-$TRIALS}; SEED=${SEED:-7}; MAXCONC=${MAXCONC:-3}
OUT=$G05_ROOT/results/steer_frs/$POLICY; mkdir -p $OUT
PORT=$(( 20000 + SLURM_JOB_ID % 20000 ))
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONFAULTHANDLER=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
GPUS=(${CUDA_VISIBLE_DEVICES//,/ }); SRV_GPU=${GPUS[0]}; CLI_GPU=${GPUS[1]:-${GPUS[0]}}
echo "policy=$POLICY node=$(hostname) server_gpu=$SRV_GPU client_gpu=$CLI_GPU conds=[$CONDS] trials=$TRIALS"
export REC_DIR=$G05_ROOT/results/steer_frs/dsbc_records  # requests with a rec_tag are saved here
CUDA_VISIBLE_DEVICES=$SRV_GPU $PY $G05_ROOT/jobs/94_frs_serve.py --ckpt_path checkpoints/g05-libero/model.pt --host 0.0.0.0 --port $PORT \
    eval_embodiment=libero --action_steps 10 --max_batch_size ${MAXB:-20} --max_wait_ms 1000 --seed $SEED &> $OUT/server_${SLURM_JOB_ID}.log &
SPID=$!
for i in $(seq 1 300); do
    grep -q "server listening on" $OUT/server_${SLURM_JOB_ID}.log 2>/dev/null && break  # the server's own ready line
    kill -0 $SPID 2>/dev/null || { echo "server died"; tail -30 $OUT/server_${SLURM_JOB_ID}.log; exit 1; }
    sleep 2
done
grep -q "server listening on" $OUT/server_${SLURM_JOB_ID}.log || { echo "server not ready after 10 min on $(hostname)"; tail -5 $OUT/server_${SLURM_JOB_ID}.log; kill $SPID; exit 1; }
spec() {  # cond t s -> "init|prompt|frs_mode|back|t0"
    case $1 in
        ref) echo "$2|0:target|none|5|0.6" ;;
        switch) echo "$3|0:task$3,40:target|none|5|0.6" ;;
        frs_ref) echo "$3|0:task$3,40:target|ref|5|0.6" ;;
        frs_b2) echo "$3|0:task$3,40:target|frs|2|0.6" ;;
        frs_b5) echo "$3|0:task$3,40:target|frs|5|0.6" ;;
        frs_b10) echo "$3|0:task$3,40:target|frs|10|0.6" ;;
        sde_06) echo "$3|0:task$3,40:target|sde|5|0.6" ;;
        sde_08) echo "$3|0:task$3,40:target|sde|5|0.8" ;;
        ref_frs_b5) echo "$2|0:target|frs|5|0.6" ;;
        ref_frs_ref) echo "$2|0:target|ref|5|0.6" ;;
        frs_ref_rec) echo "$3|0:task$3,40:target|ref|5|0.6|40:80|0|1|0|none|0|0|frs_ref_rec" ;;
        frs_exact) echo "$3|0:task$3,40:target|exact|5|0.6" ;;
        switch_ho) echo "$3|0:task$3,40:target|none|5|0.6|40:80|0|1|0|none|0|30" ;;
        frs_ref_ho) echo "$3|0:task$3,40:target|ref|5|0.6|40:80|0|1|0|none|0|30" ;;
        dsbc_ho) echo "$3|0:task$3,40:target|dsbc|5|0.6|40:80|0|1|0|none|0|30" ;;
        dsbc_tr) echo "$3|0:task$3,40:target|dsbc|5|0.6|40:80|0|1|0|none|0|0" ;;
        frs_ref_rec5) echo "$3|0:task$3,40:target|ref|5|0.6|40:80|0|1|0|none|0|0|frs_ref_rec5" ;;
        dsbc2_ho) echo "$3|0:task$3,40:target|dsbc|5|0.6|40:80|0|1|0|none|0|30" ;;
        smsteer_ho) echo "$3|0:task$3,40:target|dsbc|5|0.6|40:80|0|1|0|none|0|30" ;;
        smsteer_rec) echo "$3|0:task$3,40:target|dsbc|5|0.6|40:80|0|1|0|none|0|0|smsteer_rec" ;;
        smsteer_rec5) echo "$3|0:task$3,40:target|dsbc|5|0.6|40:80|0|1|0|none|0|0|smsteer_rec5" ;;
        smsteer2_ho) echo "$3|0:task$3,40:target|dsbc|5|0.6|40:80|0|1|0|none|0|30" ;;
        smsteer2b_ho) echo "$3|0:task$3,40:target|dsbc|5|0.6|40:80|0|1|0|none|0|30" ;;
        smsteer3_rec) echo "$3|0:task$3,40:target|dsbc|5|0.6|40:80|0|1|0|none|0|40|smsteer3_rec" ;;
        smsteer3_rec5) echo "$3|0:task$3,40:target|dsbc|5|0.6|40:80|0|1|0|none|0|40|smsteer3_rec5" ;;
        smsteer3_ho) echo "$3|0:task$3,40:target|dsbc|5|0.6|40:80|0|1|0|none|0|30" ;;
        smpol_ho) echo "$3|0:task$3,40:target|none|5|0.6|40:80|0|1|0|none|0|30" ;;
        smpol_ref_ho) echo "$2|0:target|none|5|0.6|40:80|0|1|0|none|0|30" ;;
        ref_ho) echo "$2|0:target|none|5|0.6|40:80|0|1|0|none|0|30" ;;
        zero_ho) echo "$3|0:task$3,40:target|dsbc|5|0.6|40:80|0|1|0|none|0|30" ;;
        smpol2_ho) echo "$3|0:task$3,40:target|none|5|0.6|40:80|0|1|0|none|0|30" ;;
        smpol2_ref_ho) echo "$2|0:target|none|5|0.6|40:80|0|1|0|none|0|30" ;;
        frs_ref_n3) echo "$3|0:task$3,40:target|ref|5|0.6|40:80|0|1|0|none|0.03" ;;
        frs_ref_n6) echo "$3|0:task$3,40:target|ref|5|0.6|40:80|0|1|0|none|0.06" ;;
        frs_ref_n10) echo "$3|0:task$3,40:target|ref|5|0.6|40:80|0|1|0|none|0.10" ;;
        frs_ts03) echo "$3|0:task$3,40:target|frs|3|0.6|40:80|0|0.3" ;;
        frs_ts05) echo "$3|0:task$3,40:target|frs|5|0.6|40:80|0|0.5" ;;
        sde_02) echo "$3|0:task$3,40:target|sde|5|0.2" ;;
        sde_03) echo "$3|0:task$3,40:target|sde|5|0.3" ;;
        guide_l1) echo "$3|0:task$3,40:target|sde|5|1.0|40:80|0|1|1" ;;
        guide_l3) echo "$3|0:task$3,40:target|sde|5|1.0|40:80|0|1|3" ;;
        frs_cfg) echo "$3|0:task$3,40:target|frs|5|0.6|40:80|0|1|0|old" ;;
        frs_ts05_cfg) echo "$3|0:task$3,40:target|frs|5|0.6|40:80|0|0.5|0|old" ;;
        frs_b5_arr) echo "$3|0:task$3,40:target|frs|5|0.6|40:600|0.05" ;;
        frs_ref_arr) echo "$3|0:task$3,40:target|ref|5|0.6|40:600|0.05" ;;
        sde_06_arr) echo "$3|0:task$3,40:target|sde|5|0.6|40:600|0.05" ;;
        ref_frs_b5_arr) echo "$2|0:target|frs|5|0.6|40:600|0.05" ;;
        ref_frs_ref_arr) echo "$2|0:target|ref|5|0.6|40:600|0.05" ;;
        *) echo "unknown condition $1" >&2; return 1 ;;
    esac
}
run_one() {
    local c=$1 t=$2 s=$(( ($2 + ${PAIR_SHIFT:-3}) % 10 )) d=$OUT/${1}_t$2
    IFS='|' read init sched mode kb t0 win until ts lam cfg noise off rec <<< "$(spec $c $t $s)"
    mkdir -p $d
    for attempt in 1 2 3; do
        rm -f $d/*_parallel_results.json
        CUDA_VISIBLE_DEVICES=$CLI_GPU $PY $G05_ROOT/jobs/95_frs_client.py --server_uri ws://127.0.0.1:$PORT --task_suite_name libero_goal \
            --task_id $t --init_task_id $init --prompt_schedule "$sched" --frs_mode $mode --frs_back $kb --sde_t0 $t0 \
            --frs_window ${win:-${WINDOW:-40:80}} --frs_until ${until:-0} --frs_ts ${ts:-1} --guide_lam ${lam:-0} --frs_cfg ${cfg:-none} --frs_target_noise ${noise:-0} --init_offset ${off:-0} ${rec:+--frs_rec $rec} --num_trials_per_task $TRIALS --num_steps_wait 20 \
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
grep -c "with reference" $OUT/server_${SLURM_JOB_ID}.log | sed 's/^/server batches carrying a reference: /'
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
        "schedule": task.get("prompt_schedule"), "frs": task.get("frs"), "trial_successes": task.get("trial_successes")}
for cond, v in summary.items():
    ok = [p for p in v["pairs"].values() if p]
    v["successes"] = sum(p["successes"] for p in ok)
    v["total"] = sum(p["total"] for p in ok)
    v["rate"] = v["successes"] / max(v["total"], 1)
    v["missing_pairs"] = sum(1 for p in v["pairs"].values() if not p)
    print(f"{cond:14s} {v['successes']:4d}/{v['total']:<4d} = {v['rate']:.3f}  missing_pairs={v['missing_pairs']}")
json.dump(summary, open(f"{out}/steer_summary.json", "w"), indent=2)
PYEOF
