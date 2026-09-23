"""Recompute 70-run summaries from committed records (stdlib only)."""
import csv,hashlib,json,math,statistics as st
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'results'
def close(a,b):assert abs(a-b)<1e-10,(a,b)
def main():
 training=json.loads((R/'training-summary.json').read_text());test=json.loads((R/'test/summary.json').read_text());selection=json.loads((R/'test/selection.json').read_text())['selected']
 assert len(training)==70 and len(test['runs'])==70
 rows=[]
 for run in training:
  seed,m=run['seed'],run['method'];key=f'seed-{seed}/{m}';reported=test['runs'][key];records=json.loads((R/'test'/f'seed-{seed}-{m}.json').read_text())
  assert len(records)==128 and len({x['name'] for x in records})==128
  assert all(len(x['candidates'])==4 and all(c['valid'] for c in x['candidates']) for x in records)
  mean=st.mean(c['reward'] for x in records for c in x['candidates']);best=st.mean(max(c['reward'] for c in x['candidates']) for x in records)
  close(mean,reported['mean_reward']);close(best,reported['best_of_B'])
  candidates=[(0,run['initial_validation'])]+[(e['epoch'],e['validation']) for e in run['epochs']]
  epoch,val=max(candidates,key=lambda x:(x[1]['mean_reward'],-x[0]))
  assert epoch==run['best']['epoch']==selection[key]['epoch']==reported['epoch']
  assert run['best_sha256']==selection[key]['sha256']
  close(val['mean_reward'],reported['validation']['mean_reward'])
  rows.append(dict(method=m,seed=seed,selected_epoch=epoch,train_epoch32=run['epochs'][-1]['train_mean_reward'],validation_mean=val['mean_reward'],validation_best4=val['best_of_B'],test_mean=mean,test_best4=best))
 with (R/'per-seed.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 aggregates=[]
 for m in sorted({r['method'] for r in rows}):
  local=[r for r in rows if r['method']==m];result=dict(method=m,seeds=len(local))
  for k in ['train_epoch32','validation_mean','validation_best4','test_mean','test_best4']:
   vals=[r[k] for r in local];result[k]=dict(mean=st.mean(vals),seed_std=st.stdev(vals))
  aggregates.append(result)
 aggregates.sort(key=lambda x:x['test_mean']['mean'],reverse=True)
 (R/'aggregate.json').write_text(json.dumps(aggregates,indent=2)+'\n')
 table=['| Method | Train epoch 32 | Selected validation | Test mean ± seed SD | Test Best-of-4 ± seed SD |','|---|---:|---:|---:|---:|']
 for r in aggregates:
  table.append(f"| {r['method']} | {r['train_epoch32']['mean']:.5f} | {r['validation_mean']['mean']:.5f} | {r['test_mean']['mean']:.5f} ± {r['test_mean']['seed_std']:.5f} | {r['test_best4']['mean']:.5f} ± {r['test_best4']['seed_std']:.5f} |")
 content='\n'.join(table)
 (ROOT/'RESULTS.md').write_text('''# Results: seven methods × ten new seeds

Seeds 10–19, entropy beta 0.1, 32 epochs, train/validation/test sizes 128/128/128. All 70 training runs and frozen-checkpoint test evaluations completed; all 35,840 test candidates were feasible.

'''+content+'''

QCV has the highest mean test reward; row delta has the highest mean test Best-of-4. Their paired differences are small: row minus QCV is -0.000893 for mean reward (95% paired t interval [-0.002745, 0.000959]) and +0.000310 for Best-of-4 ([-0.000449, 0.001069]). These intervals cover training-seed variation on a fixed test set; neither establishes superiority. No test-based checkpoint reselection was performed.

Train values above are sampled epoch-32 averages, not scores of the validation-selected checkpoints. Validation and test values are from each run's selected checkpoint. See `per-seed.csv` for selected epochs and individual scores.

## Learning and convergence

![Validation learning curves](results/validation-learning-curves.png)

Curves are per-epoch validation mean reward, averaged across ten seeds; bands are pointwise 95% t intervals. They are not best-so-far curves. At the descriptive threshold of three consecutive epochs with validation reward >= 0.76, all ten seeds succeed for QCV (median starting epoch 6), row delta (6), RLOO (11), and RELAX (15.5). OTB succeeds in 7/10 and REINFORCE in 6/10; their medians among successful seeds are 20 and 21, respectively. GRPO has 0/10. At 0.77, QCV and row delta both succeed 10/10, with medians 19.5 and 18. These post-hoc thresholds are descriptive, not preregistered significance tests, and do not measure wall-clock speed.

No per-epoch gradient variance was measured in this campaign. Epoch-average gradient norms are preserved as diagnostics and must not be relabeled as variance.

## Audit and scope

`scripts/summarize_results.py` verifies all test means/Best-of-4 against per-instance records, verifies validation selection including ties, and matches selected-checkpoint hashes. `results/provenance/` preserves the original campaign protocol, source snapshots and environment freeze. The historical source and portable wrappers are distinguished in the README. Earlier campaigns and later hyperparameter sweeps are excluded.
''')
 print(content);print('Verified 70 selections and 35,840 test candidates.')
if __name__=='__main__':main()
