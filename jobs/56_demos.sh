#!/bin/bash -l
#SBATCH -o /export/home/kaziz/g05/logs/slurm-%x-%j.out
#SBATCH -p gpu-all
#SBATCH --gres gpu:T4_16GB:1
#SBATCH -c 16
#SBATCH --mem 100000MB
#SBATCH --time 03:00:00
# Demo rollouts (videos) for the README. GROUP = planner | mem2 | lmem | steer
source /export/home/kaziz/g05/env_g05.sh
cd $PROJECT_ROOT
D=$G05_ROOT/results/demos/$GROUP; mkdir -p $D
PORT=$(( 20000 + SLURM_JOB_ID % 20000 ))
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONFAULTHANDLER=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
H=$G05_ROOT/results/hybrid; J=$G05_ROOT/jobs
start_server() {  # $1 = action_steps, rest = extra server args
    local as=$1; shift
    $PY $J/serve_policy_hybrid.py --ckpt_path checkpoints/g05-libero/model.pt --host 0.0.0.0 --port $PORT \
        eval_embodiment=libero --action_steps $as --max_batch_size 10 --max_wait_ms 500 --seed 7 "$@" &> $D/server.log &
    SPID=$!
    for i in $(seq 1 300); do ss -tlnpH "sport = :$PORT" 2>/dev/null | grep -q "$PORT" && break; kill -0 $SPID 2>/dev/null || { tail -30 $D/server.log; exit 1; }; sleep 2; done
}
C18="--server_uri ws://127.0.0.1:$PORT --num_steps_wait 20 --env_resolution 256 --seed 7 --save_videos"
case $GROUP in
planner)
    start_server 32
    for m in wm blank; do $PY $J/49_wm_planner.py --server_uri ws://127.0.0.1:$PORT --mode $m --task_ids 2,7,1 --trials 3 --num_parallel 3 \
        --save_gifs 3 --output_dir $D/$m &> $D/$m.log; echo "$m: $(grep '"mode"' $D/$m.log)"; done ;;
mem2)
    start_server 10
    $PY $J/18_libero_steer_client.py $C18 --task_suite_name libero_goal --task_id 0 --prompt_schedule "0:target,40:blank" \
        --num_trials_per_task 4 --num_parallel 4 --output_dir $D/no_memory &> $D/no_memory.log
    $PY $J/18_libero_steer_client.py $C18 --task_suite_name libero_goal --task_id 0 --prompt_schedule "0:target,40:memory" \
        --memory_decoder $G05_ROOT/results/mem_intent/offsets_decoder.pt --num_trials_per_task 4 --num_parallel 4 --output_dir $D/memory &> $D/memory.log ;;
lmem)
    source $J/lmem_env.sh; export LMEM_FULL=1
    start_server 10 --hybrid_ckpt $H/LMEM_ex_v100/hybrid_last.pt --hybrid_branch current --expert_ckpt $H/LMEM_ex_v100/expert_last.pt
    S="pick up the bowl and place it back on the plate"
    for t in 0 2; do
      ms=$([ $t = 0 ] && echo 260 || echo 473); n=$([ $t = 0 ] && echo 1 || echo 3)
      $PY $J/51_lmem_memclient.py $C18 --task_suite_name libero_mem --task_id $t --max_steps $ms --init_offset 100 --prompt_schedule "0:target" \
          --num_trials_per_task 6 --num_parallel 6 --n_target $n --output_dir $D/A_t$t &> $D/A_t$t.log
      $PY $J/51_lmem_memclient.py $C18 --task_suite_name libero_mem --task_id $t --max_steps $ms --init_offset 100 --prompt_schedule "0:target" \
          --num_trials_per_task 6 --num_parallel 6 --n_target $n --prompt_text "$S" --halt_mode memory \
          --count_decoder $G05_ROOT/results/lmem_count/cyc_uniform8_decoder.pt --output_dir $D/C_t$t &> $D/C_t$t.log
    done ;;
steer)
    start_server 10
    for t in 1 4; do s=$(( (t + 3) % 10 ))
      $PY $J/18_libero_steer_client.py $C18 --task_suite_name libero_goal --task_id $t --init_task_id $s --prompt_schedule "0:task$s,40:target" \
          --num_trials_per_task 3 --num_parallel 3 --output_dir $D/switch_t$t &> $D/switch_t$t.log
    done ;;
esac
kill $SPID; find $D -name "*.mp4" | head -40
