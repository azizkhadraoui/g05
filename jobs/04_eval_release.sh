#!/bin/bash -l
#SBATCH -J g05_eval_rel
#SBATCH -o /export/home/kaziz/g05/logs/slurm-%x-%j.out
#SBATCH -A h200
#SBATCH --qos h200_qos
#SBATCH -p gpu-H200
#SBATCH --gres gpu:1
#SBATCH -c 24
#SBATCH --mem 128000MB
#SBATCH --time 04:00:00
# Released g05-libero (FM) through the stock G0.5 LIBERO harness. Usage: sbatch [--export=ALL,SUITES=..,TRIALS=..,TAG=..] 04_eval_release.sh
source /export/home/kaziz/g05/env_g05.sh
cd $PROJECT_ROOT
SUITES=${SUITES:-libero_goal}; TRIALS=${TRIALS:-10}; PAR=${PAR:-10}; TAG=${TAG:-smoke}
# eval_libero_parallel.py over-counts when num_parallel > num_trials (it assigns a trial to every env up front and
# counts whichever finish in the stopping step, biased toward fast successes). Never let PAR exceed TRIALS.
(( PAR > TRIALS )) && PAR=$TRIALS
OUT=$G05_ROOT/results/eval_release_${TAG}
echo "== GPU / EGL preflight"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES SLURM_JOB_GPUS=$SLURM_JOB_GPUS SLURM_STEP_GPUS=$SLURM_STEP_GPUS"
nvidia-smi --query-gpu=index,pci.bus_id,memory.used,memory.total --format=csv
$PY $G05_ROOT/jobs/egl_probe.py
export PYTHONFAULTHANDLER=1   # native crashes in simulator workers print a stack
echo "suites=$SUITES trials=$TRIALS parallel=$PAR out=$OUT node=$(hostname)"; t0=$(date +%s)
env TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 bash scripts/run/eval_libero.sh checkpoints/g05-libero/model.pt \
    --suites "$SUITES" --num_trials $TRIALS --num_parallel $PAR --output_dir $OUT; rc=$?
t1=$(date +%s); echo "rc=$rc wall=$((t1-t0))s"
[ -f $OUT/summary.json ] && $PY -c "
import json; s=json.load(open('$OUT/summary.json'))
for k,v in s['suites'].items(): print(k, v['successes'], '/', v['total'], f\"{v['success_rate']:.3f}\")
print('grand', s.get('grand_successes'), s.get('grand_success_rate'))"
for c in $OUT/*/client.log; do echo "== $c"; tr '\r' '\n' < $c | grep -E 'Error|rror:|Traceback' | tail -3; done
exit $rc
