#!/bin/bash -l
#SBATCH -o /export/home/kaziz/g05/logs/slurm-%x-%j.out
#SBATCH -p gpu-all
#SBATCH -c 16
#SBATCH --mem 100000MB
#SBATCH --time 03:00:00
# Instrumented rollouts for the team explainer video. GROUP = planner (2 GPUs: G0.5 + RoboBrain) | memb (1 GPU)
source /export/home/kaziz/g05/env_g05.sh
cd $PROJECT_ROOT
D=$G05_ROOT/results/video/$GROUP; mkdir -p $D
PORT=$(( 20000 + SLURM_JOB_ID % 20000 ))
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONFAULTHANDLER=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
H=$G05_ROOT/results/hybrid; J=$G05_ROOT/jobs; V=$G05_ROOT/models/RoboBrain2.5-4B
start_server() {
    local as=$1; shift
    $PY $J/serve_policy_hybrid.py --ckpt_path checkpoints/g05-libero/model.pt --host 0.0.0.0 --port $PORT \
        eval_embodiment=libero --action_steps $as --max_batch_size 10 --max_wait_ms 300 --seed 7 "$@" &> $D/server.log &
    SPID=$!
    for i in $(seq 1 300); do ss -tlnpH "sport = :$PORT" 2>/dev/null | grep -q "$PORT" && break; kill -0 $SPID 2>/dev/null || { tail -30 $D/server.log; exit 1; }; sleep 2; done
}
P49="$PY $J/49_wm_planner.py --server_uri ws://127.0.0.1:$PORT"
case $GROUP in
planner)
    start_server 32
    $P49 --mode oracle --task_ids 3 --trials 2 --num_parallel 2 --log_trials 2 --output_dir $D/oracle &> $D/oracle.log
    $P49 --mode blank --task_ids 3 --trials 2 --num_parallel 2 --log_trials 2 --output_dir $D/blank &> $D/blank.log
    $P49 --mode wm --task_ids 3,7,1 --trials 3 --num_parallel 3 --log_trials 3 --output_dir $D/wm &> $D/wm.log
    $P49 --mode vlm --goal_type image --vlm $V --task_ids 2,7,1 --trials 3 --num_parallel 3 --log_trials 3 --output_dir $D/vlm &> $D/vlm.log
    for m in oracle blank wm vlm; do echo "$m: $(grep '"mode"' $D/$m.log)"; done ;;
memb)
    source $J/lmem_env.sh; export LMEM_FULL=1
    start_server 10 --hybrid_ckpt $H/LMEM_ex_v100/hybrid_last.pt --hybrid_branch current --expert_ckpt $H/LMEM_ex_v100/expert_last.pt
    S="pick up the bowl and place it back on the plate"
    C51="$PY $J/51_lmem_memclient.py --server_uri ws://127.0.0.1:$PORT --task_suite_name libero_mem --num_steps_wait 20 --env_resolution 256 --seed 7 --save_videos --init_offset 100 --prompt_schedule 0:target"
    for t in 2 0; do
      ms=$([ $t = 0 ] && echo 260 || echo 473); n=$([ $t = 0 ] && echo 1 || echo 3)
      $C51 --task_id $t --max_steps $ms --num_trials_per_task 8 --num_parallel 8 --n_target $n --output_dir $D/A_t$t &> $D/A_t$t.log
      $C51 --task_id $t --max_steps $ms --num_trials_per_task 8 --num_parallel 8 --n_target $n --prompt_text "$S" --halt_mode memory \
          --count_decoder $G05_ROOT/results/lmem_count/cyc_uniform8_decoder.pt --output_dir $D/C_t$t &> $D/C_t$t.log
    done ;;
esac
kill $SPID
