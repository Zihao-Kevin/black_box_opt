"""The exact-gradient reference for the JobShop curve figure.  Same actor initialization, same instance schedule and
same plain-SGD step as the eight estimators (kings_val128_train.py under the run config), but the update is the EXACT
gradient of the expected reward on the scheduled instance - sum over the 625 plans of p(plan | instance) reward(plan),
TreeMath.true_grad - from the exhaustive offline pool, so nothing is sampled and nothing is evaluated.  It is the
analogue of the "exact gradient" line of the MCP and QP figures (Live_agent_rowwise.ipynb table_grad(tree, f);
live_free.py W.true_grad(st, f)), and the x axis is charged the two episodes per update the sampled methods spend.
Writes OUT/seed-S/exact_gradient/{initial.pt, actor-N.pt every save_every, identity.json, metrics.jsonl, complete.json},
the layout expected_curve.py scores.
usage: exact_gradient.py RUN_CONFIG OUT SEEDS(a-b) [--stop 256] [--check]     (CUDA_VISIBLE_DEVICES picks the GPU)
--check (one seed) verifies the run against the sampled campaign before anything is written: the schedule's first
instance and the initial actor match the qcv run of that seed, true_grad agrees with the grad_w form of the same sum,
a finite difference of J along the gradient matches |g|^2, and the mean of REINFORCE samples converges to it."""
import argparse,json,os,sys,time,numpy as np,torch
from pathlib import Path
ROOT=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))));DATA=os.environ.get('JOBSHOP_DATA',os.path.join(ROOT,'data'));sys.path.insert(0,ROOT);os.chdir(ROOT)
from jobshop_rl.data import experiment_data,grouped_batches,write_json,digest
from jobshop_rl.baseline_train import load_tree,atomic_save
from jobshop_rl.kings_val128_train import actor_initialization,cache_relevant
CACHE=Path(DATA)/'jobshop-v3';POOL=os.path.join(DATA,'offline-q-v3x/data/pool.json')
CAMPAIGN=Path(DATA)/'run13-lr0.5/v3q-sgd-lr0.5-beta0-g2i1-save'      # the sampled runs the reference is matched to

def setup(config):
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    cfg=json.loads(Path(config).read_text());data,split,_=experiment_data(cfg)
    assert cache_relevant(json.loads((CACHE/'identity.json').read_text())['config'])==cache_relevant(cfg),'Backbone cache configuration mismatch'
    assert cfg.get('optimizer')=='sgd' and cfg.get('entropy_beta',.1)==0.0,'the reference is defined for the plain-SGD, no-entropy campaign'
    grammar=json.loads((CACHE/'grammar.json').read_text());common=torch.load(CACHE/'common.pt',map_location='cpu',weights_only=False)
    pool=json.load(open(POOL));table={}
    for r in pool:table.setdefault(r['name'],np.full(len(grammar['plans']),np.nan))[r['action']]=r['reward']
    assert all(n in table and np.isfinite(table[n]).all() for n in split['train']),'exhaustive pool incomplete'
    return cfg,split,grammar,common,table,digest(pool)

def objective(W,st,f):return float(W.values(st,f)[1][0])                  # expected reward at the root: sum_plan p(plan) reward(plan)

def check(cfg,split,grammar,common,table,seed):
    initA,initB=actor_initialization(common,seed);schedule=grouped_batches(split['train'],32,seed,cfg['group_size'],cfg['instances_per_batch'])
    run=CAMPAIGN/f'seed-{seed}'/'qcv';first=json.loads(open(run/'journal.jsonl').readline())['name'];ini=torch.load(run/'initial.pt',map_location='cpu',weights_only=False)
    print('schedule[0] instance',schedule[0][0],'== campaign journal',first,':',schedule[0][0]==first)
    print('initial actor == campaign initial.pt :',torch.equal(initA,ini['A']) and torch.equal(initB,ini['B']))
    A,B=initA.cuda(),initB.cuda();name=schedule[0][0];W=load_tree(CACHE,name,grammar,common);st=W.forward(A,B);f=W.rewards_to_entries(table[name])
    gA,gB=W.true_grad(st,f);g2=float(gA.double().pow(2).sum()+gB.double().pow(2).sum())
    Q,_=W.values(st,f);w=st['pn'][W.ent_node]*st['P']*Q;hA,hB=W.grad_w(st,torch.arange(W.E,device=W.device),w)        # sum_e p(reach e) Q_e s_e
    rel=lambda a,b:(float((a.double()-b.double()).norm()/b.double().norm()) if float(b.double().norm())>0 else float(a.double().norm()))
    print(f'true_grad vs grad_w(pn*P*Q) : rel diff A {rel(hA,gA):.2e}  B {rel(hB,gB):.2e}   (|gA| {float(gA.norm()):.2e}: zero at init since LoRA B = 0)')
    J0=objective(W,st,f);g=np.sqrt(g2);dA,dB=gA/g,gB/g                                                     # unit step along the gradient: dJ/dt = |g|
    for t in (3e-2,1e-2,3e-3):
        Jp=objective(W,W.forward(A+t*dA,B+t*dB),f);Jm=objective(W,W.forward(A-t*dA,B-t*dB),f)
        print(f'central difference t {t:g}: (J(+t)-J(-t))/2t = {(Jp-Jm)/(2*t):.6f}   |g| = {g:.6f}   ratio {(Jp-Jm)/(2*t)/g:.4f}')
    c=torch.zeros(W.E,dtype=torch.float64,device=W.device);pl=(st['pn'][W.ent_node]*st['P']);eA=torch.zeros_like(gA).double();eB=torch.zeros_like(gB).double()
    for e in W.leaf_idx:                                                                                   # EXACT expectation of the REINFORCE estimator over the 625 plans
        path=[int(e)]
        while W.pe_np[W.en_np[path[-1]]]>=0:path.append(int(W.pe_np[W.en_np[path[-1]]]))
        dA_,dB_=W.episode_grad(st,c,f[e],path);eA+=float(pl[e])*dA_.double();eB+=float(pl[e])*dB_.double()
    print(f'sum over all plans p(plan) * REINFORCE sample == true_grad : rel diff A {rel(eA,gA):.2e}  B {rel(eB,gB):.2e}   (plan probabilities sum to {float(pl[W.leaf].sum()):.6f})')
    print(f'J on {name} at init {J0:.4f}, |g| {np.sqrt(g2):.4f}')

def train(cfg,split,grammar,common,table,poolhash,out,seed,stop):
    out=Path(out)/f'seed-{seed}'/'exact_gradient';out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():print('have',out,flush=True);return
    initA,initB=actor_initialization(common,seed);A=initA.cuda().requires_grad_();B=initB.cuda().requires_grad_()
    opt=torch.optim.SGD([A,B],lr=cfg['learning_rate']);schedule=grouped_batches(split['train'],32,seed,cfg['group_size'],cfg['instances_per_batch'])
    ident=dict(version='exact-gradient-v1',method='exact_gradient',seed=seed,actor_init_seed=42+seed,config=cfg,cache=common['identity'],entropy_beta=0.0,
               pool_sha256=poolhash,reward_table='v3_exhaustive_offline (exact, all 625 plans)',sampling='none: TreeMath.true_grad on the scheduled instance')
    identity=digest(ident);write_json(out/'identity.json',dict(hash=identity,**ident))
    atomic_save(out/'initial.pt',dict(A=initA,B=initB,identity=identity,epoch=0,config=cfg,method='exact_gradient',seed=seed))
    (out/'metrics.jsonl').unlink(missing_ok=True);t0=time.time();J=[]
    for u,batch in enumerate(schedule[:stop],1):
        names=list(dict.fromkeys(batch));grad=[torch.zeros_like(A),torch.zeros_like(B)];Ju=[]
        for name in names:
            W=load_tree(CACHE,name,grammar,common);st=W.forward(A,B);f=W.rewards_to_entries(table[name])
            gA,gB=W.true_grad(st,f);grad[0].add_(gA/len(names));grad[1].add_(gB/len(names));Ju.append(objective(W,st,f));del W,st
        norm=float(sum(v.square().sum() for v in grad).sqrt())
        if not np.isfinite(norm):raise RuntimeError('Nonfinite gradient')
        opt.zero_grad(set_to_none=True);A.grad=-grad[0];B.grad=-grad[1];opt.step()                      # the trainer's step, verbatim
        if not torch.isfinite(A).all() or not torch.isfinite(B).all():raise RuntimeError('Nonfinite actor')
        if cfg.get('save_every') and u%cfg['save_every']==0:atomic_save(out/f'actor-{u}.pt',dict(A=A.detach().cpu(),B=B.detach().cpu(),update=u))
        J.append(float(np.mean(Ju)))
        with (out/'metrics.jsonl').open('a') as fh:fh.write(json.dumps(dict(update=u,names=names,expected_reward=J[-1],gradient_norm=norm))+'\n')
    write_json(out/'complete.json',dict(method='exact_gradient',seed=seed,updates=stop,episodes_charged=2*stop,training_candidates=0,validation_candidates=0,wall_seconds=time.time()-t0,completed_at=time.time()))
    print(f'seed {seed}: {stop} updates, J on the scheduled instance {J[0]:.4f} -> {J[-1]:.4f}, {time.time()-t0:.0f}s',flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('config');ap.add_argument('out');ap.add_argument('seeds');ap.add_argument('--stop',type=int,default=256);ap.add_argument('--check',action='store_true');a=ap.parse_args()
    cfg,split,grammar,common,table,poolhash=setup(a.config);lo,hi=map(int,a.seeds.split('-'))
    if a.check:check(cfg,split,grammar,common,table,lo)
    else:
        for seed in range(lo,hi+1):train(cfg,split,grammar,common,table,poolhash,a.out,seed,a.stop)
