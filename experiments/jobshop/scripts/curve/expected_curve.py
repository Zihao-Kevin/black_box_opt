"""Exact expected reward of every saved actor: sum_plan p(plan | instance) * reward(instance, plan), over the 128 validation
instances (exhaustive val table) and the 128 train instances (exhaustive offline table). No sampling anywhere.
usage: expected_curve.py RUN_CONFIG_DIR OUT_JSON   (env JOBSHOP_SHARD=i/n to split runs across GPUs)"""
import json,os,sys,numpy as np,torch
from pathlib import Path
ROOT=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))));DATA=os.environ.get('JOBSHOP_DATA',os.path.join(ROOT,'data'));sys.path.insert(0,ROOT);os.chdir(ROOT)
from jobshop_rl.baseline_train import load_tree
from jobshop_rl.data import experiment_data
cdir,outp=Path(sys.argv[1]),Path(sys.argv[2]);sh,nsh=map(int,os.environ.get('JOBSHOP_SHARD','0/1').split('/'))
cache=Path(DATA)/'jobshop-v3';cfg=json.loads(Path('configs/synthetic-val128-v3-groups.json').read_text());data,split,_=experiment_data(cfg)
g=json.loads((cache/'grammar.json').read_text());common=torch.load(cache/'common.pt',map_location='cpu',weights_only=False)
conf=np.array(g['conf']);leaf=np.array(g['entry_child'])<0;leaf_action=conf[leaf]
tables={'train':json.load(open(os.path.join(DATA,'offline-q-v3x/model/selected-tables.json')))}
vt=json.load(open(os.path.join(DATA,'val-table-v3/data/pool.json')));T={}
for r in vt:T.setdefault(r['name'],np.zeros(625))[r['action']]=r['reward']
tables['val']=T
runs=sorted(cdir.glob('seed-*/*'))[sh::nsh];actors={}
for r in runs:
    fs=sorted(r.glob('actor-*.pt'),key=lambda p:int(p.stem.split('-')[1]))
    ini=torch.load(r/'initial.pt',map_location='cpu',weights_only=False)
    actors[r]=[(0,ini['A'].cuda(),ini['B'].cuda())]+[(int(f.stem.split('-')[1]),*[torch.load(f,map_location='cpu',weights_only=False)[k].cuda() for k in ('A','B')]) for f in fs]
print('runs',len(runs),'actors per run',len(next(iter(actors.values()))),flush=True)
acc={r:{'val':np.zeros(len(actors[r])),'train':np.zeros(len(actors[r]))} for r in runs}
for part in ('val','train'):
    for ni,name in enumerate(split[part]):
        W=load_tree(cache,name,g,common);tab=torch.as_tensor(np.asarray(tables[part][name])[leaf_action],dtype=torch.float64,device='cuda')
        for r in runs:
            for j,(u,A,B) in enumerate(actors[r]):
                st=W.forward(A,B);pl=(st['pn'][W.ent_node]*st['P'])[W.leaf];acc[r][part][j]+=float((pl*tab).sum())
        del W
        if ni%32==31:print(part,ni+1,flush=True)
out={str(r.relative_to(cdir)):dict(updates=[u for u,_,_ in actors[r]],val=(acc[r]['val']/len(split['val'])).tolist(),train=(acc[r]['train']/len(split['train'])).tolist()) for r in runs}
outp.write_text(json.dumps(out));print('WROTE',outp,flush=True)
