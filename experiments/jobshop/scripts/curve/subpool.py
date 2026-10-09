"""Fitted-table label budget by subsampling the exhaustive v3 pool.  usage: subpool.py K OUTDIR"""
import json,os,sys,collections,numpy as np
from pathlib import Path
K=int(sys.argv[1]);out=Path(sys.argv[2])
DATA=os.environ.get('JOBSHOP_DATA',os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),'data'))
rows=json.load(open(os.path.join(DATA,'offline-q-v3x/data/pool.json')))
by=collections.defaultdict(list)
for r in rows:by[r['name']].append(r)
rng=np.random.default_rng(20260924);keep=[]
for n in sorted(by):
    pool=sorted(by[n],key=lambda r:r['action'])
    for i in rng.choice(len(pool),K,replace=False):
        r=dict(pool[int(i)]);r['source']=f'v3_random{K}_offline';keep.append(r)
keep.sort(key=lambda r:(r['name'],r['action']))
(out/'data').mkdir(parents=True,exist_ok=True);(out/'data/pool.json').write_text(json.dumps(keep))
print('POOL',K,len(keep),'rows, mean reward %.4f'%np.mean([r['reward'] for r in keep]))
