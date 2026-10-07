#!/bin/bash
# The per-decision variance decomposition along five seeds, under the offline protocol (offline.py): the strict reward along the
# Q + rowΔ run (fitted table) and the graded reward along the Q + rowΔ (exact) run, as the paper's Figure 1.  One job per GPU slot.
# usage: bash launch_offline.sh [gpus="1 3 4 5 6 7"] [seeds="0 1 2 3 4"]
cd "$(dirname "$0")/../.."
PY=${PY:-python}   # the interpreter of the black_box_opt env
GPUS=(${1:-1 3 4 5 6 7}); SEEDS=(${2:-0 1 2 3 4}); i=0
export DECOMP_LR=${DECOMP_LR:-0.045} DECOMP_WARM=_sgd_long/warm_offline DECOMP_OFFLINE=_sgd_long/offline_data.json DECOMP_OUT=_sgd_long/decomp/offline${DECOMP_SUFFIX}
jobs=(); for s in "${SEEDS[@]}"; do jobs+=("strict:$s" "graded:$s"); done
for j in "${jobs[@]}"; do
  g=${GPUS[$((i % ${#GPUS[@]}))]}; r=${j%%:*}; s=${j##*:}
  if [ "$r" = strict ]; then DECOMP_SEED=$s "$PY" _sgd_long/decomp/decompose.py $g > _sgd_long/decomp/offline_strict_s${s}.log 2>&1 &
  else DECOMP_SEED=$s DECOMP_REWARD=graded DECOMP_METHOD="Q + rowΔ (exact)" "$PY" _sgd_long/decomp/decompose.py $g > _sgd_long/decomp/offline_graded_s${s}.log 2>&1 &
  fi
  i=$((i+1)); [ $((i % ${#GPUS[@]})) -eq 0 ] && wait
done
wait
