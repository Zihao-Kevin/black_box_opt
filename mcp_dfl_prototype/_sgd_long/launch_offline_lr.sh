#!/bin/bash
# A second learning rate for the strict reward under the offline protocol, queued after launch_offline.sh: the same methods and
# seeds, no graded phase.  usage: bash launch_offline_lr.sh <lr> [gpus="1 3 4 5 6 7"]
cd "$(dirname "$0")/.."
PY=${PY:-python}   # the interpreter of the black_box_opt env
LR=$1; GPUS=(${2:-1 3 4 5 6 7}); N=$(( ${#GPUS[@]} * 2 )); i=0
while pgrep -f "launch_offline[.]sh" > /dev/null; do sleep 60; done
export SGD_WARM=_sgd_long/warm_offline SGD_OFFLINE=_sgd_long/offline_data.json SGD_FIXED_STEPS=100 SGD_SEEDS=0-20
for g in "${GPUS[@]}"; do for k in 0 1; do
  ( export SGD_WORKER=$i/$N
    SGD_JOBS="$LR:_sgd_long/offline_lr$LR" SGD_METHODS="Q + rowΔ,Q,Q + Δ,REINFORCE,exact gradient" "$PY" _sgd_long/worker.py $g
    SGD_JOBS="$LR:_sgd_long/offline_lr${LR}_fix" SGD_METHODS="RLOO,GRPO,OTB" "$PY" _sgd_long/worker_fix.py $g
  ) > _sgd_long/offline${LR}_worker_gpu${g}_${k}.log 2>&1 &
  i=$((i+1))
done; done
wait
