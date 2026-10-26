#!/bin/bash -l
#SBATCH -o /export/home/kaziz/g05/logs/slurm-%x-%j.out
#SBATCH -p gpu-all
#SBATCH --gres gpu:T4_16GB:1
#SBATCH -c 16
#SBATCH --mem 100000MB
#SBATCH --time 06:00:00
# LIBERO-Mem closed loop with a memory module (51_lmem_memclient.py). Full-length episodes: success = the benchmark's
# overshoot-aware done at the last step. Repetition tasks 0-5, 20 held-out initial states (demos 100-119).
#   A  original instruction, no module           (fine-tuned G0.5 alone)
#   B  single-cycle instruction, no module       (never stops -> overshoot)
#   C  single-cycle + memory module (full-episode memory counts cycles, halts at N)
#   D  single-cycle + current-frame module      (Markovian control)
#   E  single-cycle + oracle counter             (upper bound)
#   P  single-cycle + activity-log counter (own grasp-lift-release cycles from gripper and hand height; lmem_proprio.py)
#   F  single-cycle + RoboBrain counter, 8 frames over the episode + now (needs VLM=<ckpt dir>, 2 GPUs)
#   G  single-cycle + RoboBrain counter, current frame only
# Env: NAME, CONDS="A B C D E", TASKS="0 1 2 3 4 5", HYBRID_CKPT, EXPERT_CKPT (optional), TRIALS=20
source /export/home/kaziz/g05/env_g05.sh
source /export/home/kaziz/g05/jobs/lmem_env.sh
cd $PROJECT_ROOT
NAME=${NAME:-run}; TRIALS=${TRIALS:-20}; OUT=$G05_ROOT/results/lmem_cl/$NAME; mkdir -p $OUT
PORT=$(( 20000 + SLURM_JOB_ID % 20000 ))
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONFAULTHANDLER=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True LMEM_FULL=1
HY=(); [ -n "${HYBRID_CKPT:-}" ] && HY=(--hybrid_ckpt $HYBRID_CKPT --hybrid_branch current)
[ -n "${EXPERT_CKPT:-}" ] && HY+=(--expert_ckpt $EXPERT_CKPT)
$PY $G05_ROOT/jobs/serve_policy_hybrid.py --ckpt_path checkpoints/g05-libero/model.pt --host 0.0.0.0 --port $PORT \
    eval_embodiment=libero --action_steps 10 --max_batch_size 10 --max_wait_ms 1000 --seed 7 "${HY[@]}" &> $OUT/server_${SLURM_JOB_ID}.log &
SPID=$!
for i in $(seq 1 300); do ss -tlnpH "sport = :$PORT" 2>/dev/null | grep -q "$PORT" && break; kill -0 $SPID 2>/dev/null || { tail -30 $OUT/server_${SLURM_JOB_ID}.log; exit 1; }; sleep 2; done
grep -E "^\[hybrid\]|^\[lmem\]|^\[t4\] policy" $OUT/server_${SLURM_JOB_ID}.log
PAPER=(200 290 320 370 450 655 480 675 390 390); NREP=(1 1 3 3 5 7); OBJ=(bowl bottle bowl bottle bowl bowl)
declare -A SINGLE=([bowl]="pick up the bowl and place it back on the plate" [bottle]="lift the bottle and put it down on the plate")
for c in ${CONDS:-A B C D E}; do for t in ${TASKS:-0 1 2 3 4 5}; do
  name=$(ls $LM/libero-mem/libero/libero/bddl_files/libero_mem | grep "^KITCHEN_SCENE1_$((t+1))_" | sed 's/.bddl$//')
  dl=$(python3 -c "import json;print(json.load(open('$LM/demo_meta.json'))['$name']['len_max'])")
  ms=$(( (dl > PAPER[t] ? dl : PAPER[t]) * 13 / 10 ))
  ARGS=(--n_target ${NREP[$t]})
  case $c in
    A) ;;
    B) ARGS+=(--prompt_text "${SINGLE[${OBJ[$t]}]}") ;;
    C) ARGS+=(--prompt_text "${SINGLE[${OBJ[$t]}]}" --halt_mode memory --count_decoder $G05_ROOT/results/lmem_count/cyc_uniform8_decoder.pt) ;;
    D) ARGS+=(--prompt_text "${SINGLE[${OBJ[$t]}]}" --halt_mode current --count_decoder $G05_ROOT/results/lmem_count/cyc_current_decoder.pt) ;;
    E) ARGS+=(--prompt_text "${SINGLE[${OBJ[$t]}]}" --halt_mode oracle) ;;
    P) ARGS+=(--prompt_text "${SINGLE[${OBJ[$t]}]}" --halt_mode proprio) ;;  # activity log: own grasp-lift-release cycles
    F) ARGS+=(--prompt_text "${SINGLE[${OBJ[$t]}]}" --halt_mode vlm --count_vlm $VLM --count_object ${OBJ[$t]}) ;;
    G) ARGS+=(--prompt_text "${SINGLE[${OBJ[$t]}]}" --halt_mode vlm_current --count_vlm $VLM --count_object ${OBJ[$t]}) ;;
  esac
  d=$OUT/$c/t$t; mkdir -p $d
  for attempt in 1 2; do
    rm -f $d/*_parallel_results.json
    $PY $G05_ROOT/jobs/51_lmem_memclient.py --server_uri ws://127.0.0.1:$PORT --task_suite_name libero_mem --task_id $t \
        --max_steps $ms --init_offset 100 --prompt_schedule "0:target" --num_trials_per_task $TRIALS --num_steps_wait 20 \
        --num_parallel 10 --env_resolution 256 --seed 7 --output_dir $d "${ARGS[@]}" &> $d/client_attempt$attempt.log
    ls $d/*_parallel_results.json &>/dev/null && break
    echo "RETRY $c t$t ($(grep -m1 -oE '[A-Za-z]+Error.{0,120}' $d/client_attempt$attempt.log))"
  done
  python3 -c "
import json,glob
r=json.load(open(glob.glob('$d/*_parallel_results.json')[0]))['tasks'][0]
g=r.get('goal_len') or 1; sub=r.get('trial_subgoals',{}); fin=r.get('trial_final',{})
s=sum(v['final_done'] for v in fin.values()); h=sum(v['halt_step'] is not None for v in fin.values())
print('$c t$t N=${NREP[$t]} success', s,'/',len(fin), 'subgoal_completion', round(sum(min(v,g) for v in sub.values())/max(len(sub),1)/g,3), 'halted', h)" 2>/dev/null || echo "$c t$t FAILED"
done; done
kill $SPID
