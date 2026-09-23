"""Evaluate frozen validation-selected actors on each streamed test feature."""
import hashlib,json
from pathlib import Path
import numpy as np
import torch
from .row_tree import RowTree
from .evaluator import Evaluator
from .actions import build_code
from .data import write_json
ROOT=Path('runs/test-selected-beta010')
ACTORS={}
def evaluate_instance(name,index,features,grammar,common,cfg,instance):
 selection=json.loads((ROOT/'selection.json').read_text())['selected']
 ev=Evaluator(cfg['image'],cfg['timeout'],score_mode='synthetic_lb')
 W=RowTree(grammar,features,common)
 for method,choice in selection.items():
  target=ROOT/method;target.mkdir(exist_ok=True)
  dest=target/(name+'.json')
  if dest.exists():continue
  if method not in ACTORS:
   path=Path(choice['checkpoint']);assert hashlib.sha256(path.read_bytes()).hexdigest()==choice['sha256']
   ck=torch.load(path,map_location='cpu',weights_only=False)
   ACTORS[method]=(ck['A'].cuda(),ck['B'].cuda())
  with torch.no_grad():st=W.forward(*ACTORS[method])
  rng=np.random.default_rng(10000+1000*index)
  episodes=[W.sample(st,rng) for _ in range(4)]
  pending=target/'pending.json'
  if pending.exists():raise RuntimeError('Uncommitted test group requires reconciliation')
  write_json(pending,{'name':name,'actions':[int(W.conf[e]) for e,_ in episodes]})
  records=[]
  for e,_ in episodes:
   action=int(W.conf[e]);result=ev.evaluate(build_code(grammar['plans'][action]),instance)
   if not result['valid']:raise RuntimeError(str(result))
   records.append({'action':action,**result})
  write_json(dest,{'name':name,'candidates':records});pending.unlink()
 print(json.dumps({'stage':'test_instance_complete','name':name,'index':index}),flush=True)
def finish(names):
 summaries={}
 for method in json.loads((ROOT/'selection.json').read_text())['selected']:
  groups=[json.loads((ROOT/method/(n+'.json')).read_text())['candidates'] for n in names]
  summaries[method]={'partition':'test','instances':len(names),'B':4,'candidate_evaluations':4*len(names),'mean_reward':float(np.mean([r['reward'] for g in groups for r in g])),'best_of_B':float(np.mean([max(r['reward'] for r in g) for g in groups])),'feasible_rate':float(np.mean([r['valid'] for g in groups for r in g]))}
 write_json(ROOT/'summary.json',summaries)
