#!/bin/bash
# Submit a whole PrefPI run as one Slurm dependency chain (run on the login node; it only calls sbatch):
#   rollouts(iter 0 = BASE, shared) -> train(1) -> rollouts(1, CFG) -> train(2) -> ... -> rollouts(K)
# Usage: OBJ=speed|height BASE=<iteration-0 rollout dir> [K=8] [START=0 PREV_JOB=<jobid>] [NAME=<run name>]
#        [EXTRA="<more trainer args, e.g. --anchor_frac 0.5>"] bash 122_prefpi_chain.sh
# START>0 resumes after a failure: iterations < START must already have their rollouts and expert files.
set -eu
G=/export/home/kaziz/g05; J=$G/jobs; NAME=${NAME:-${OBJ:?}}; R=$G/results/prefpi/$NAME
K=${K:-8}; START=${START:-0}; BASE=${BASE:?}; mkdir -p $R
dep=${PREV_JOB:+--dependency=afterok:$PREV_JOB}
dirs=$BASE
for ((k = 1; k <= START; k++)); do dirs="$dirs,$R/iter$k"; done
for ((k = START; k < K; k++)); do
  init=""; (( k > 0 )) && init="--init_expert $R/expert_$k.pt"
  t=$(ARGS="--rollouts $dirs --obj $OBJ $init --out $R/expert_$((k + 1)).pt ${EXTRA:-}" \
      sbatch --parsable -J g05_pp_${NAME}_t$((k + 1)) $dep --export=ALL $J/124_prefpi_train.sh)
  r=$(OUT=$R/iter$((k + 1)) EXPERT=$R/expert_$((k + 1)).pt CFG_W=5 PREFIX="[cfg] " SEED=$((7 + k + 1)) \
      sbatch --parsable -J g05_pp_${NAME}_r$((k + 1)) --dependency=afterok:$t ${EXCLUDE_NODES:+--exclude=$EXCLUDE_NODES} --export=ALL $J/123_prefpi_rollout.sh)
  echo "iter $((k + 1)): train $t -> rollouts $r"
  dirs="$dirs,$R/iter$((k + 1))"; dep="--dependency=afterok:$r"
done
