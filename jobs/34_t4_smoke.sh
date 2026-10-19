#!/bin/bash -l
#SBATCH -J g05_t4smoke
#SBATCH -o /export/home/kaziz/g05/logs/slurm-%x-%j.out
#SBATCH -p gpu-all
#SBATCH --gres gpu:T4_16GB:1
#SBATCH -c 16
#SBATCH --mem 100000MB
#SBATCH --time 00:50:00
# Can G0.5 serve LIBERO on a T4 in fp16? 1 task x 6 trials (released policy) + the same with a memory decoder.
source /export/home/kaziz/g05/env_g05.sh
cd $PROJECT_ROOT
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONFAULTHANDLER=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PORT=$(( 20000 + SLURM_JOB_ID % 20000 )); OUT=$G05_ROOT/results/t4smoke; rm -rf $OUT; mkdir -p $OUT
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
$PY $G05_ROOT/jobs/serve_policy_hybrid.py --ckpt_path checkpoints/g05-libero/model.pt --host 0.0.0.0 --port $PORT \
    eval_embodiment=libero --action_steps 10 --max_batch_size 6 --max_wait_ms 1000 --seed 7 &> $OUT/server.log &
SPID=$!
for i in $(seq 1 300); do ss -tlnpH "sport = :$PORT" 2>/dev/null | grep -q "$PORT" && break; kill -0 $SPID 2>/dev/null || { tail -30 $OUT/server.log; exit 1; }; sleep 2; done
grep -E "^\[t4\]|\[hybrid\]" $OUT/server.log
t0=$(date +%s)
$PY $G05_ROOT/jobs/18_libero_steer_client.py --server_uri ws://127.0.0.1:$PORT --task_suite_name libero_goal --task_id 0 \
    --prompt_schedule "0:target" --num_trials_per_task 6 --num_steps_wait 20 --num_parallel 6 --env_resolution 256 --seed 7 \
    --output_dir $OUT/plain &> $OUT/plain.log
echo "plain rc=$? wall=$(( $(date +%s) - t0 ))s"; python3 -c "import json,glob;t=json.load(open(glob.glob('$OUT/plain/*_parallel_results.json')[0]))['tasks'][0];print('plain',t['successes'],'/',t['total_episodes'])" || tail -20 $OUT/plain.log
t0=$(date +%s)
$PY $G05_ROOT/jobs/18_libero_steer_client.py --server_uri ws://127.0.0.1:$PORT --task_suite_name libero_goal --task_id 0 \
    --prompt_schedule "0:target,40:memory" --num_trials_per_task 6 --num_steps_wait 20 --num_parallel 6 --env_resolution 256 --seed 7 \
    --memory_decoder $G05_ROOT/results/mem_intent/offsets_decoder.pt --output_dir $OUT/mem &> $OUT/mem.log
echo "mem rc=$? wall=$(( $(date +%s) - t0 ))s"; python3 -c "import json,glob;t=json.load(open(glob.glob('$OUT/mem/*_parallel_results.json')[0]))['tasks'][0];print('mem',t['successes'],'/',t['total_episodes'],t.get('trial_inferred'))" || tail -20 $OUT/mem.log
grep -E "infer=|Batch timing" $OUT/server.log | tail -2; nvidia-smi --query-gpu=memory.used --format=csv,noheader
kill $SPID
