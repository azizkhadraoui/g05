#!/bin/bash -l
#SBATCH -o /export/home/kaziz/g05/logs/slurm-%x-%j.out
#SBATCH -p gpu-all
#SBATCH --gres gpu:T4_16GB:1
#SBATCH -c 16
#SBATCH --mem 100000MB
#SBATCH --time 08:00:00
# Level-2 memory test: LIBERO-Mem object-identity tasks 6-9 (swap 2 bowls, rotate 3 bowls, cheese -> that / empty basket),
# G0.5 fine-tuned on these tasks' whole demos (original instruction) and step episodes (explicit step instructions)
# by 07b_train_lmem.py on results/lmem_steps (131_lmem_steps.py). Full-length episodes, 20 held-out starting states
# (demos 100-119); success = the benchmark's overshoot-aware sequence check at the last step.
#   A  original instruction, no module                     (fine-tuned G0.5 alone, Markovian)
#   C  step instructions from a video-memory decoder       (8 frames over the episode + now; halts on "done")
#   D  step instructions from a current-frame decoder      (Markovian module)
#   E  step instructions from the simulator (oracle)       (upper bound)
# Env: NAME, CONDS="A C D E", TASKS="6 7 8 9", HYBRID_CKPT, EXPERT_CKPT, TRIALS=20, NPAR=10 (parallel envs; 5 if the GPU runs out of memory)
source /export/home/kaziz/g05/env_g05.sh
source /export/home/kaziz/g05/jobs/lmem_env.sh
cd $PROJECT_ROOT
NAME=${NAME:-steps}; TRIALS=${TRIALS:-20}; OUT=$G05_ROOT/results/lmem_cl/$NAME; mkdir -p $OUT
PORT=$(( 20000 + SLURM_JOB_ID % 20000 ))
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONFAULTHANDLER=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True LMEM_FULL=1
HY=(); [ -n "${HYBRID_CKPT:-}" ] && HY=(--hybrid_ckpt $HYBRID_CKPT --hybrid_branch current)
[ -n "${EXPERT_CKPT:-}" ] && HY+=(--expert_ckpt $EXPERT_CKPT)
$PY $G05_ROOT/jobs/serve_policy_hybrid.py --ckpt_path checkpoints/g05-libero/model.pt --host 0.0.0.0 --port $PORT \
    eval_embodiment=libero --action_steps 10 --max_batch_size 10 --max_wait_ms 1000 --seed 7 "${HY[@]}" &> $OUT/server_${SLURM_JOB_ID}.log &
SPID=$!
for i in $(seq 1 300); do ss -tlnpH "sport = :$PORT" 2>/dev/null | grep -q "$PORT" && break; kill -0 $SPID 2>/dev/null || { tail -30 $OUT/server_${SLURM_JOB_ID}.log; exit 1; }; sleep 2; done
grep -E "^\[hybrid\]|^\[lmem\]|^\[t4\] policy" $OUT/server_${SLURM_JOB_ID}.log
PAPER=(200 290 320 370 450 655 480 675 390 390); DEC=$G05_ROOT/results/lmem_count
for c in ${CONDS:-A C D E}; do for t in ${TASKS:-6 7 8 9}; do
  name=$(ls $LM/libero-mem/libero/libero/bddl_files/libero_mem | grep "^KITCHEN_SCENE1_$((t+1))_" | sed 's/.bddl$//')
  dl=$(python3 -c "import json;print(json.load(open('$LM/demo_meta.json'))['$name']['len_max'])")
  ms=$(( (dl > PAPER[t] ? dl : PAPER[t]) * 13 / 10 ))
  case $c in
    A) ARGS=() ;;
    C) ARGS=(--halt_mode instr_memory --count_decoder $DEC/instr_uniform8_decoder.pt) ;;
    D) ARGS=(--halt_mode instr_current --count_decoder $DEC/instr_current_decoder.pt) ;;
    E) ARGS=(--halt_mode instr_oracle) ;;
  esac
  d=$OUT/$c/t$t; mkdir -p $d
  for attempt in 1 2; do
    rm -f $d/*_parallel_results.json
    $PY $G05_ROOT/jobs/51_lmem_memclient.py --server_uri ws://127.0.0.1:$PORT --task_suite_name libero_mem --task_id $t \
        --max_steps $ms --init_offset 100 --prompt_schedule "0:target" --num_trials_per_task $TRIALS --num_steps_wait 20 \
        --num_parallel ${NPAR:-10} --env_resolution 256 --seed 7 --output_dir $d "${ARGS[@]}" &> $d/client_attempt$attempt.log
    ls $d/*_parallel_results.json &>/dev/null && break
    echo "RETRY $c t$t ($(grep -m1 -oE '[A-Za-z]+Error.{0,120}' $d/client_attempt$attempt.log))"
  done
  python3 -c "
import json,glob
r=json.load(open(glob.glob('$d/*_parallel_results.json')[0]))['tasks'][0]
g=r.get('goal_len') or 1; sub=r.get('trial_subgoals',{}); fin=r.get('trial_final',{})
s=sum(v['final_done'] for v in fin.values()); h=sum(v['halt_step'] is not None for v in fin.values())
print('$c t$t success', s,'/',len(fin), 'subgoal_completion', round(sum(min(v,g) for v in sub.values())/max(len(sub),1)/g,3), 'halted', h)" 2>/dev/null || echo "$c t$t FAILED"
done; done
kill $SPID
