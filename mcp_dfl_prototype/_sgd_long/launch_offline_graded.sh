#!/bin/bash
# The graded-reward phase alone at a given learning rate, queued after the other launchers.  usage: bash launch_offline_graded.sh <lr> [gpus]
cd "$(dirname "$0")/.."
PY=${PY:-python}   # the interpreter of the black_box_opt env
LR=$1; GPUS=(${2:-1 3 4 5 6 7}); N=$(( ${#GPUS[@]} * 2 )); i=0
while pgrep -f "launch_offline(_lr)?[.]sh" > /dev/null; do sleep 60; done
export SGD_WARM=_sgd_long/warm_offline SGD_OFFLINE=_sgd_long/offline_data.json SGD_FIXED_STEPS=100 SGD_SEEDS=0-20
for g in "${GPUS[@]}"; do for k in 0 1; do
  ( export SGD_WORKER=$i/$N
    SGD_REWARD=graded SGD_JOBS="$LR:_sgd_long/offline_lr${LR}_graded" SGD_METHODS="Q + rowΔ (exact),Q (exact),V (exact),REINFORCE,exact gradient,Q + rowΔ,Q,V" "$PY" _sgd_long/worker.py $g
  ) > _sgd_long/offline${LR}g_worker_gpu${g}_${k}.log 2>&1 &
  i=$((i+1))
done; done
wait
