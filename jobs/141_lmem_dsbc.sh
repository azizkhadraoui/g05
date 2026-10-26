#!/bin/bash -l
#SBATCH -o /export/home/kaziz/g05/logs/slurm-%x-%j.out
#SBATCH -p gpu-all
#SBATCH --gres gpu:v100_16GB:1
#SBATCH -c 16
#SBATCH --mem 100000MB
#SBATCH --time 08:00:00
# Noise policy for a weak skill (experiment 2a): the LIBERO-Mem basket tasks (8: cheese -> nearest basket, that basket to
# the centre; 9: ... the empty basket to the centre) with G0.5 fine-tuned on LIBERO-Mem (LaWAM adapter + action expert,
# served by 94_frs_serve.py so the sampler's starting noise can be recorded or chosen), step instructions as in 132.
#   MODE=rec    oracle step instructions on training start states (OFFSET=0), every request saved with its exact noise
#               (REC_DIR, tags PREFIX_t{task}_tr{state}_s{step}) and the completed-step count logged per request
#   MODE=train  140_lmem_dsbc.py on the recorded sets (SETS="prefix:results_root,...") -> DSBC_CKPT
#   MODE=eval   held-out start states (OFFSET=100, demos 100-119), conditions:
#     E   oracle step instructions, G0.5's own Gaussian noise     C   video-memory decoder's step instructions (132 C)
#     Ez  E with zero noise (the sampler's deterministic mode)    Cz  C with zero noise
#     Ed  E with the noise policy (DSBC_CKPT)                     Cd  C with the noise policy
# Env: MODE, NAME, TASKS="8 9", CONDS, TRIALS, OFFSET, PREFIX, SETS, HYBRID_CKPT, EXPERT_CKPT, DSBC_CKPT, NPAR=10
source /export/home/kaziz/g05/env_g05.sh
source /export/home/kaziz/g05/jobs/lmem_env.sh
cd $PROJECT_ROOT
R=$G05_ROOT/results/lmem_dsbc; export REC_DIR=$R/records
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONFAULTHANDLER=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True LMEM_FULL=1
echo "node=$(hostname) mode=${MODE:?} name=${NAME:-} tasks=${TASKS:-8 9} conds=${CONDS:-} dsbc=${DSBC_CKPT:-none}"
if [ "$MODE" = train ]; then
    cd $G05_ROOT/jobs && exec $PY -u 140_lmem_dsbc.py --rec_dir $REC_DIR --sets ${SETS:?} --out ${DSBC_CKPT:?} ${EXTRA:-}
fi
OUT=$R/${NAME:?}; mkdir -p $OUT
export HYBRID_CKPT=${HYBRID_CKPT:?} EXPERT_CKPT=${EXPERT_CKPT:?}
PORT=$(( 20000 + SLURM_JOB_ID % 20000 ))
$PY $G05_ROOT/jobs/94_frs_serve.py --ckpt_path checkpoints/g05-libero/model.pt --host 0.0.0.0 --port $PORT \
    eval_embodiment=libero --action_steps 10 --max_batch_size 10 --max_wait_ms 1000 --seed 7 &> $OUT/server_${SLURM_JOB_ID}.log &
SPID=$!; trap 'kill $SPID 2>/dev/null' EXIT
for i in $(seq 1 300); do
    grep -q "server listening on" $OUT/server_${SLURM_JOB_ID}.log 2>/dev/null && break
    kill -0 $SPID 2>/dev/null || { echo "server died"; tail -30 $OUT/server_${SLURM_JOB_ID}.log; exit 1; }
    sleep 2
done
grep -E "^\[hybrid\]|^\[frs\]" $OUT/server_${SLURM_JOB_ID}.log
PAPER=(200 290 320 370 450 655 480 675 390 390); DEC=$G05_ROOT/results/lmem_count
[ "$MODE" = rec ] && CONDS=R
for c in ${CONDS:?}; do for t in ${TASKS:-8 9}; do
  name=$(ls $LM/libero-mem/libero/libero/bddl_files/libero_mem | grep "^KITCHEN_SCENE1_$((t+1))_" | sed 's/.bddl$//')
  dl=$(python3 -c "import json;print(json.load(open('$LM/demo_meta.json'))['$name']['len_max'])")
  ms=$(( (dl > PAPER[t] ? dl : PAPER[t]) * 13 / 10 ))
  case $c in
    R) ARGS=(--halt_mode instr_oracle --rec_prefix ${PREFIX:?}) ;;
    E) ARGS=(--halt_mode instr_oracle) ;;
    Ez) ARGS=(--halt_mode instr_oracle --frs_mode noise --noise_temp 0) ;;
    Ed) ARGS=(--halt_mode instr_oracle --frs_mode dsbc) ;;
    C) ARGS=(--halt_mode instr_memory --count_decoder $DEC/instr_uniform8_decoder.pt) ;;
    Cz) ARGS=(--halt_mode instr_memory --count_decoder $DEC/instr_uniform8_decoder.pt --frs_mode noise --noise_temp 0) ;;
    Cd) ARGS=(--halt_mode instr_memory --count_decoder $DEC/instr_uniform8_decoder.pt --frs_mode dsbc) ;;
  esac
  d=$OUT/$c/t$t; mkdir -p $d
  TMO=$(( ${TRIALS:-20} * ms / ${NPAR:-10} / 6 + 1800 ))  # a hung client (e.g. an exception lost in a worker thread) is killed and retried
  for attempt in 1 2; do
    rm -f $d/*_parallel_results.json
    timeout -k 60 $TMO $PY $G05_ROOT/jobs/51_lmem_memclient.py --server_uri ws://127.0.0.1:$PORT --task_suite_name libero_mem --task_id $t \
        --max_steps $ms --init_offset ${OFFSET:-100} --prompt_schedule "0:target" --num_trials_per_task ${TRIALS:-20} --num_steps_wait 20 \
        --num_parallel ${NPAR:-10} --env_resolution 256 --seed 7 --output_dir $d "${ARGS[@]}" &> $d/client_attempt$attempt.log
    ls $d/*_parallel_results.json &>/dev/null && break
    echo "RETRY $c t$t ($(grep -m1 -oE '[A-Za-z]+Error.{0,120}' $d/client_attempt$attempt.log))"
  done
  python3 -c "
import json,glob
r=json.load(open(glob.glob('$d/*_parallel_results.json')[0]))['tasks'][0]
fin=r.get('trial_final',{}); s=sum(v['final_done'] for v in fin.values())
sub=r.get('trial_subgoals',{}); g=r.get('goal_len') or 2
print('$c t$t success', s,'/',len(fin), 'subgoals reached', sum(min(v,g) for v in sub.values()),'/',g*len(sub))" 2>/dev/null || echo "$c t$t FAILED"
done; done
