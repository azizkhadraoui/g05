#!/bin/bash -l
#SBATCH -o /export/home/kaziz/g05/logs/slurm-%x-%j.out
#SBATCH -p gpu-all
#SBATCH --gres gpu:T4_16GB:1
#SBATCH -c 16
#SBATCH --mem 100000MB
#SBATCH --time 04:00:00
# LIBERO-Mem closed-loop eval of G0.5 (T4 fp16 serving). 20 held-out initial states per task (demos 100-119),
# LIBERO-Mem subgoal-aware success (sequence counter advanced every step), step budget = max(paper, longest demo) x 1.1.
# Env: NAME, TASKS="0 1 ...", SCHED (prompt schedule, default "0:target"), MEMDEC (optional memory decoder)
source /export/home/kaziz/g05/env_g05.sh
source /export/home/kaziz/g05/jobs/lmem_env.sh
cd $PROJECT_ROOT
NAME=${NAME:-zeroshot}; TRIALS=${TRIALS:-20}; SCHED=${SCHED:-0:target}
OUT=$G05_ROOT/results/lmem/$NAME; mkdir -p $OUT
PORT=$(( 20000 + SLURM_JOB_ID % 20000 ))
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONFAULTHANDLER=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
$PY $G05_ROOT/jobs/serve_policy_hybrid.py --ckpt_path checkpoints/g05-libero/model.pt --host 0.0.0.0 --port $PORT \
    eval_embodiment=libero --action_steps 10 --max_batch_size 10 --max_wait_ms 1000 --seed 7 &> $OUT/server_${SLURM_JOB_ID}.log &
SPID=$!
for i in $(seq 1 300); do ss -tlnpH "sport = :$PORT" 2>/dev/null | grep -q "$PORT" && break; kill -0 $SPID 2>/dev/null || { tail -30 $OUT/server_${SLURM_JOB_ID}.log; exit 1; }; sleep 2; done
PAPER=(200 290 320 370 450 655 480 675 390 390)
for t in ${TASKS:-0 1 2 3 4 5 6 7 8 9}; do
  name=$(ls $LM/libero-mem/libero/libero/bddl_files/libero_mem | grep "^KITCHEN_SCENE1_$((t+1))_" | sed 's/.bddl$//')
  dl=$(python3 -c "import json;print(json.load(open('$LM/demo_meta.json'))['$name']['len_max'])")
  ms=$(( (dl > PAPER[t] ? dl : PAPER[t]) * 11 / 10 ))
  d=$OUT/t$t; mkdir -p $d
  for attempt in 1 2; do
    rm -f $d/*_parallel_results.json
    $PY $G05_ROOT/jobs/43_lmem_client.py --server_uri ws://127.0.0.1:$PORT --task_suite_name libero_mem --task_id $t \
        --max_steps $ms --init_offset 100 --prompt_schedule "$SCHED" --num_trials_per_task $TRIALS --num_steps_wait 20 \
        --num_parallel 10 --env_resolution 256 --seed 7 --output_dir $d ${MEMDEC:+--memory_decoder $MEMDEC} &> $d/client_attempt$attempt.log
    ls $d/*_parallel_results.json &>/dev/null && break
    echo "RETRY t$t ($(grep -m1 -oE '[A-Za-z]+Error[^\n]{0,120}' $d/client_attempt$attempt.log))"
  done
  python3 -c "
import json,glob
r=json.load(open(glob.glob('$d/*_parallel_results.json')[0]))['tasks'][0]
g=r.get('goal_len') or 1; sub=r.get('trial_subgoals',{})
print('t$t', '$name'[15:], 'steps=$ms', 'success', r['successes'],'/',r['total_episodes'], 'subgoal_completion', round(sum(min(v,g) for v in sub.values())/max(len(sub),1)/g,3), 'goal_len', g)" 2>/dev/null || echo "t$t FAILED"
done
kill $SPID
