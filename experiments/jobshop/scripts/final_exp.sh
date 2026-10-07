#!/bin/bash

# Needs 8 GPUs
set -euo pipefail

PY=${PY:-"/net/csefiles/siemens/warriors-ls-recovery/users/zzhao628/#113442817/anaconda3/envs/bilevel/bin/python"}
ROOT=/nethome/zzhao628/blogs/black_box_opt/experiments/jobshop
DATA=/data/zzhao628
CACHE=$DATA/jobshop-v3
CFG=configs/synthetic-val128-v3-groups.json
LR=0.5; K=400; SEEDS=10-49; STOP=256; SLOTS=48; GPUS=8
CFGDIR=$DATA/run13-lr$LR-configs; STEM=v3q-sgd-lr$LR-beta0-g2i1-save
SAVE=$DATA/run13-lr$LR/$STEM
RESULT=$ROOT/results/v3-fitted$K-curves-lr$LR.json
FROM=${FROM:-0}; UNTIL=${UNTIL:-9}
export HF_HOME=$DATA/hf
cd "$ROOT"
stage () { [ "$FROM" -le "$1" ] && [ "$UNTIL" -ge "$1" ] && { echo; echo "== stage $1: $2"; return 0; } || return 1; }
have  () { [ -e "$1" ] && [ "$FROM" -eq 0 ]; }

# 1. backbone features for the v3 group grammar (one file per train/val instance, ~3.4 GB)
if stage 1 "feature cache"; then
  if have "$CACHE/identity.json"; then echo "   have $CACHE"; else
    "$PY" -m jobshop_rl.row_prepare_sharded --config "$CFG" --out "$CACHE"
  fi
fi

# 2. exhaustive solver rewards: all 625 plans on the 128 train and the 128 validation instances.
#    The train pool is what the label budget is drawn from; the val pool is the exact curve metric.
if stage 2 "exhaustive reward pools"; then
  have "$DATA/offline-q-v3x/data/pool.json" && echo "   have train pool" || "$PY" scripts/curve/pool_v3x.py
  have "$DATA/val-table-v3/data/pool.json"  && echo "   have val pool"   || "$PY" scripts/curve/pool_val.py
fi

# 3. fitted table on the full pool: expected_curve.py reads it for the train curve
if stage 3 "fit exhaustive table"; then
  if have "$DATA/offline-q-v3x/model/complete.json"; then echo "   have v3x fit"; else
    "$PY" -m jobshop_rl.offline_reward --config "$CFG" --pool "$DATA/offline-q-v3x/data/pool.json" \
      --out "$DATA/offline-q-v3x/model" --grammar "$CACHE/grammar.json" 2>&1 | grep -E '"stage": "action_holdout"' | cut -c1-160
  fi
fi

# 4. the table the run actually trains against: K random labels per instance, refit
if stage 4 "fit $K-label table"; then
  if have "$DATA/offline-q-v3q/model/complete.json"; then echo "   have v3q fit"; else
    "$PY" scripts/curve/subpool.py "$K" "$DATA/offline-q-v3q"
    "$PY" -m jobshop_rl.offline_reward --config "$CFG" --pool "$DATA/offline-q-v3q/data/pool.json" \
      --out "$DATA/offline-q-v3q/model" --grammar "$CACHE/grammar.json" 2>&1 | grep -E '"stage": "action_holdout"' | cut -c1-160
  fi
  ln -sfn "$DATA/offline-q-v3q" runs/offline-q-v3q
fi

# 5. the run config: the v3 group task under plain SGD, two samples per update, no entropy bonus
if stage 5 "run config"; then
  mkdir -p "$CFGDIR"
  "$PY" - "$CFG" "$CFGDIR/$STEM.json" "$LR" <<'PYEOF'
import json,sys
src,dst,lr=sys.argv[1],sys.argv[2],float(sys.argv[3])
c=json.load(open(src))
c.update(lr=lr,learning_rate=lr,optimizer='sgd',entropy_beta=0.0,group_size=2,
         instances_per_batch=1,save_every=2,headroom_every=0,offline_q='runs/offline-q-v3q')
json.dump(c,open(dst,'w'),indent=1);print('   wrote',dst)
PYEOF
fi

# 6. training. Two halves so at most 4 methods x 8 GPUs of docker evaluation run at once;
#    launch.py resumes, so rerunning it after a crash only starts the unfinished runs.
if stage 6 "train 8 methods x seeds $SEEDS"; then
  for half in otb,vbase,qcv,row_delta reinforce,rloo,grpo,relax; do
    JOBSHOP_CACHE=$CACHE PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      "$PY" scripts/curve/launch.py "$DATA/run13-lr$LR" "$CFGDIR" "$half" "$SEEDS" "$STOP" "$SLOTS"
  done
  n=$(ls "$SAVE"/seed-*/*/val-epoch-2/summary.json 2>/dev/null | wc -l)
  e=$(grep -l -E 'Traceback|Error' "$SAVE"/seed-*/*/trainer.log 2>/dev/null | wc -l)
  echo "   $n/320 finished, $e with errors"
fi

# 7. exact expected reward of every saved actor, sharded one GPU per shard, then merged
if stage 7 "expected-reward curves"; then
  for i in $(seq 0 $((GPUS-1))); do
    JOBSHOP_SHARD=$i/$GPUS CUDA_VISIBLE_DEVICES=$i "$PY" scripts/curve/expected_curve.py \
      "$SAVE" "$DATA/curve-fitted$K-lr$LR-shard$i.json" > "$DATA/curve-fitted$K-lr$LR-shard$i.log" 2>&1 &
  done; wait
  "$PY" - "$DATA/curve-fitted$K-lr$LR-shard" "$RESULT" <<'PYEOF'
import json,glob,sys
out={}
for f in sorted(glob.glob(sys.argv[1]+'*.json')): out.update(json.load(open(f)))
json.dump(out,open(sys.argv[2],'w'));print('   merged',len(out),'runs ->',sys.argv[2])
PYEOF
  "$PY" scripts/curve/stats_curve.py "$RESULT"
fi

# 8. the exact-gradient reference: same init, schedule and SGD step, but the exact gradient of the expected reward on the
#    scheduled instance from the exhaustive train pool (nothing sampled, no docker) - the dashed line of the curve figure.
#    Seven seed ranges on GPUs 1-7 (GPU 0 is shared); scored like the campaign and merged INTO the results file.
if stage 8 "exact-gradient reference"; then
  EXACT=$DATA/run13-lr$LR/exact-gradient; mkdir -p "$EXACT"; i=1
  for R in 10-15 16-21 22-27 28-33 34-39 40-44 45-49; do
    CUDA_VISIBLE_DEVICES=$i OMP_NUM_THREADS=1 "$PY" -u scripts/curve/exact_gradient.py "$CFGDIR/$STEM.json" "$EXACT" $R > "$EXACT/train-gpu$i.log" 2>&1 &
    i=$((i+1))
  done; wait
  for i in $(seq 0 6); do
    JOBSHOP_SHARD=$i/7 CUDA_VISIBLE_DEVICES=$((i+1)) "$PY" scripts/curve/expected_curve.py "$EXACT" "$DATA/curve-exact-lr$LR-shard$i.json" > "$DATA/curve-exact-lr$LR-shard$i.log" 2>&1 &
  done; wait
  "$PY" scripts/curve/merge_curves.py "$RESULT" "$DATA/curve-exact-lr$LR-shard"
  "$PY" scripts/curve/stats_curve.py "$RESULT"
fi

# 9. the figure itself (redraws all four fitted panels; this one is drawn on seeds 10-29)
if stage 9 "figure"; then
  cd "$ROOT/.." && "$PY" paper_figs_src/make_paper_figs.py jobshop_curve_fitted row
  echo "   paper_figs/fig_jobshop_curve_fitted${K}_lr${LR}.pdf and paper_figs/fig_main_row.pdf"
fi
