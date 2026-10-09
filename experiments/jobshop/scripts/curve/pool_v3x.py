"""Offline label pool for the v3 group grammar: 64 random distinct actions per train instance, exact solver reward.
Same shape as the published pool (name, action, reward, actual_makespan, source, occurrences)."""
import json,os,sys,numpy as np
from multiprocessing import Pool
from pathlib import Path
ROOT=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))));DATA=os.environ.get('JOBSHOP_DATA',os.path.join(ROOT,'data'));sys.path.insert(0,ROOT);os.chdir(ROOT)
os.environ.setdefault('HF_HOME',os.path.join(DATA,'hf'))
from jobshop_rl.data import experiment_data
from jobshop_rl.actions import build_code,MENU5,GROUPS
cfg=json.loads(Path('configs/synthetic-val128-v3-groups.json').read_text());data,split,_=experiment_data(cfg)
def grammar():
    from transformers import AutoTokenizer
    from jobshop_rl.row_tree import make_grammar
    tok=AutoTokenizer.from_pretrained(cfg['planner']['model'],revision=cfg['planner']['revision'],local_files_only=True)
    return make_grammar(tok,groups=GROUPS,menu=MENU5)
G=grammar();plans=G['plans'];rng=np.random.default_rng(20260923)
jobs=[(n,int(a)) for n in split['train'] for a in range(len(plans))]
def work(chunk):
    from jobshop_rl.persistent_evaluator import PersistentEvaluator
    ev=PersistentEvaluator(cfg['image'],cfg['timeout'],score_mode='synthetic_lb');out=[]
    for n,a in chunk:
        r=ev.evaluate(build_code(plans[a]),data[n]);out.append(dict(name=n,action=a,reward=r['reward'],actual_makespan=r['actual_makespan'],source='v3_exhaustive_offline',occurrences=1))
    ev.close();return out
if __name__=='__main__':
    W=64;chunks=[jobs[i::W] for i in range(W)]
    with Pool(W) as p:rows=[r for part in p.map(work,chunks) for r in part]
    rows.sort(key=lambda r:(r['name'],r['action']));out=Path(DATA)/'offline-q-v3x';(out/'data').mkdir(parents=True,exist_ok=True)
    (out/'data/pool.json').write_text(json.dumps(rows));print('POOL DONE',len(rows),'rows, mean reward %.4f'%np.mean([r['reward'] for r in rows]),flush=True)
