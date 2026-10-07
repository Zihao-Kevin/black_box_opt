"""Summary of the exact expected-validation-reward curves for every method present. usage: stats_curve.py CURVES_JSON"""
import json,sys,numpy as np
from scipy import stats
d=json.load(open(sys.argv[1]));NM={'row_delta':'Q+rowD','qcv':'Q','vbase':'V','otb':'OTB','relax':'RELAX','rloo':'RLOO','grpo':'GRPO','reinforce':'REINFORCE','exact_gradient':'EXACT'}
L=max(len(v['val']) for v in d.values())
assert L>1,f'{sys.argv[1]}: every run is a single point - those trainers died before saving an actor'
d={k:v for k,v in d.items() if len(v['val'])==L}          # drop runs that died before the last checkpoint
S={m:np.array([v['val'] for k,v in d.items() if k.endswith('/'+m)]) for m in NM if any(k.endswith('/'+m) for k in d)}
S={m:v for m,v in S.items() if len(v)}
u=np.array(next(iter(d.values()))['updates'])*2;order=sorted(S,key=lambda m:-S[m][:,-1].mean())
print(f'{"method":10s} {"n":>3s} {"start":>7s} {"ep 20":>7s} {"ep 60":>7s} {"final":>7s} {"sd":>7s} {"AUC":>7s} {">=0.81":>7s}')
for m in order:
    r=lambda e:S[m][:,min(np.searchsorted(u,e),len(u)-1)].mean();ok=int((S[m][:,-1]>=0.81).sum())
    print(f'{NM[m]:10s} {len(S[m]):3d} {S[m][:,0].mean():7.4f} {r(20):7.4f} {r(60):7.4f} {S[m][:,-1].mean():7.4f} {S[m][:,-1].std(ddof=1):7.4f} {S[m].mean():7.4f} {ok:4d}/{len(S[m])}')
if 'qcv' in S:
    print('\npaired AUC vs Q:')
    for m in order:
        if m=='qcv' or len(S[m])!=len(S['qcv']):continue
        A,B=S[m].mean(1),S['qcv'].mean(1);t,p=stats.ttest_rel(A,B);print(f'  {NM[m]:>9s} - Q  {(A-B).mean():+.4f}  p={p:.4f}  wins {int((A>B).sum())}/{len(A)}')
