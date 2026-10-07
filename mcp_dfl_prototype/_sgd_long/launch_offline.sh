#!/bin/bash
# The paper's live-agent runs with the fixed offline dataset and the one shared warm start (offline.py), lr 0.045, 20 seeds:
#   strict reward: the control-variate rows, REINFORCE, V and the exact gradient (worker.py), then RLOO/GRPO/OTB (worker_fix.py);
#   graded reward: the plug-in and exact-table rows.  Two workers per GPU, a static split of the (method, seed) jobs.
# usage: bash launch_offline.sh [lr=0.045] [gpus="1 3 4 5 6 7"]
cd "$(dirname "$0")/.."
PY=${PY:-python}   # the interpreter of the black_box_opt env
LR=${1:-0.045}; GPUS=(${2:-1 3 4 5 6 7}); N=$(( ${#GPUS[@]} * 2 )); i=0
export SGD_WARM=_sgd_long/warm_offline SGD_OFFLINE=_sgd_long/offline_data.json SGD_FIXED_STEPS=100 SGD_SEEDS=0-20
for g in "${GPUS[@]}"; do for k in 0 1; do
  ( export SGD_WORKER=$i/$N
    SGD_JOBS="$LR:_sgd_long/offline_lr$LR" SGD_METHODS="Q + rowΔ,Q,Q + Δ,REINFORCE,exact gradient,V" "$PY" _sgd_long/worker.py $g
    SGD_JOBS="$LR:_sgd_long/offline_lr${LR}_fix" SGD_METHODS="RLOO,GRPO,OTB" "$PY" _sgd_long/worker_fix.py $g
    SGD_REWARD=graded SGD_JOBS="$LR:_sgd_long/offline_lr${LR}_graded" SGD_METHODS="Q + rowΔ,Q,V,REINFORCE,exact gradient,Q + rowΔ (exact),Q (exact),V (exact)" "$PY" _sgd_long/worker.py $g
  ) > _sgd_long/offline${LR}_worker_gpu${g}_${k}.log 2>&1 &
  i=$((i+1))
done; done
wait
