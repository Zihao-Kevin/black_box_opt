"""Multi-seed Kings runs with bounded Docker concurrency and compact checkpoints."""
import argparse,fcntl,json,math,os,time
from pathlib import Path
import numpy as np
import torch
from .data import ROOT,experiment_data,grouped_batches,write_json,digest
from .baseline_train import atomic_save,load_tree,step_gradient,load_frozen_tables
from .baseline_adapter import BaselineAdapter
from .dynamic_reward import DynamicReward
from .dispatch_entropy import dispatch_entropy
from .evaluator import Evaluator,InfrastructureError
from .actions import build_code

METHODS=('qcv','scalar_delta','vbase','row_delta','grpo','relax','rloo','otb','reinforce')
# Cached features are the frozen backbone's hidden states at the grammar prefixes. They depend on
# the planner, the dataset and the batch schedule; these keys only steer the optimizer, so a cache
# built under one setting is valid under another. Everything else must still match exactly.
TRAINING_ONLY=('learning_rate','lr','optimizer','entropy_beta','headroom_every','group_size','instances_per_batch','offline_q','save_every')
# group_size / instances_per_batch shape the batch schedule, which the feature builder used only to pick WHICH
# instances to cache; the check in run() below verifies every scheduled instance is present instead.
def cache_relevant(cfg):return {k:v for k,v in cfg.items() if k not in TRAINING_ONLY}

def append(path,value):
    with Path(path).open('a') as f:f.write(json.dumps(value,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())

def actor_initialization(common,seed):
    A=common['A0'].clone();B=common['B0'].clone()
    if seed:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42+seed);torch.nn.init.kaiming_uniform_(A,a=math.sqrt(5))
    return A,B

def reinforce_gradient(W,st,episodes,rewards):
    # Raw sequence-sum REINFORCE: no baseline, normalization or KL.
    weights=torch.zeros(W.E,device=W.device,dtype=torch.float64)
    for (_,path),r in zip(episodes,rewards):
        idx=torch.tensor(path,device=W.device);weights.index_add_(0,idx,torch.full((len(path),),float(r)/len(episodes),dtype=torch.float64,device=W.device))
    nz=weights.nonzero().flatten()
    return W.grad_w(st,nz,weights[nz]),{}

class BoundedEvaluator:
    def __init__(self,cfg,slots,folder,out):
        backend=os.environ.get('JOBSHOP_EVALUATOR','oneshot')
        if backend=='persistent-v1':
            from .persistent_evaluator import PersistentEvaluator
            self.ev=PersistentEvaluator(cfg['image'],cfg['timeout'],score_mode='synthetic_lb')
        elif backend=='oneshot':self.ev=Evaluator(cfg['image'],cfg['timeout'],score_mode='synthetic_lb')
        else:raise ValueError('Unknown evaluation backend: '+backend)
        self.slots=slots;self.folder=Path(folder);self.folder.mkdir(parents=True,exist_ok=True);self.out=out
    def evaluate(self,code,instance,tag):
        start=time.time();handle=None
        # Wait outside the candidate timeout; at most slots concurrent containers.
        while handle is None:
            control=self.folder/'limit.json'
            limit=min(self.slots,max(1,int(json.loads(control.read_text())['slots']))) if control.exists() else self.slots
            offset=os.getpid()%limit
            for i in range(limit):
                f=(self.folder/f'docker-{(i+offset)%limit}.lock').open('a')
                try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);handle=f;break
                except BlockingIOError:f.close()
            if handle is None:time.sleep(.05)
        waited=time.time()-start
        try:
            r=self.ev.evaluate(code,instance)
        except Exception as e:
            append(self.out/'evaluation-attempts.jsonl',dict(tag=tag,status='exception',error=str(e),time=time.time(),wait_seconds=waited));raise
        finally:
            fcntl.flock(handle,fcntl.LOCK_UN);handle.close()
        append(self.out/'evaluation-attempts.jsonl',dict(tag=tag,status=r['status'],valid=r['valid'],time=time.time(),wait_seconds=waited,wall_seconds=r['wall_seconds']))
        if not r['valid']:raise InfrastructureError('Constrained solver evaluation failed: '+str(r))
        return r

def recover_committed_logs(out,state,last_records,last_metric):
    """Checkpoint is the commit boundary; rebuild trailing journal after interruption."""
    n=state['updates']
    for filename,limit,key,last in [('journal.jsonl',state['candidates'],lambda x:x['update'],last_records),('metrics.jsonl',n,lambda x:x['counts']['updates'],[last_metric] if last_metric else [])]:
        p=out/filename;rows=[]
        if p.exists():
            for line in p.read_text().splitlines():
                try:r=json.loads(line)
                except json.JSONDecodeError:continue
                if key(r)<=n:rows.append(r)
        rows=[r for r in rows if key(r)<n]+list(last) if n else []
        if len(rows)!=limit:raise RuntimeError(f'Cannot reconcile {filename}: {len(rows)} != {limit}')
        tmp=p.with_suffix('.tmp');tmp.write_text(''.join(json.dumps(r)+'\n' for r in rows));tmp.replace(p)
    pending=out/'pending.json'
    if pending.exists():
        archive=out/'recovery';archive.mkdir(exist_ok=True);pending.rename(archive/f'pending-{time.time_ns()}.json')

def evaluate(cfg,cache,grammar,common,A,B,data,names,ev,out,label,mismatch=False,train_names=None):
    target=out/label;target.mkdir(exist_ok=True)
    if (target/'summary.json').exists():return json.loads((target/'summary.json').read_text())
    records=json.loads((target/'records.json').read_text()) if (target/'records.json').exists() else []
    done={r['name'] for r in records};start=time.time()
    for ni,name in enumerate(names):
        if name in done:continue
        prompt=name
        if mismatch:
            shape=(len(data[name]['duration_matrix']),len(data[name]['duration_matrix'][0]))
            prompt=next(n for n in sorted(train_names) if (len(data[n]['duration_matrix']),len(data[n]['duration_matrix'][0]))==shape)
        W=load_tree(cache,prompt,grammar,common)
        with torch.no_grad():st=W.forward(A,B)
        rng=np.random.default_rng(10000+1000*ni);episodes=[W.sample(st,rng) for _ in range(4)];group=[]
        for e,path in episodes:
            action=int(W.conf[e]);r=ev.evaluate(build_code(grammar['plans'][action]),data[name],dict(stage=label,name=name,action=action));group.append(dict(action=action,**r))
        records.append(dict(name=name,prompt_name=prompt,candidates=group));write_json(target/'records.json',records)
        del W,st
    rewards=[c['reward'] for r in records for c in r['candidates']]
    s=dict(mean_reward=float(np.mean(rewards)),best_of_B=float(np.mean([max(c['reward'] for c in r['candidates']) for r in records])),candidate_evaluations=len(rewards),feasible_rate=1.,plan_legal_rate=1.,B=4,partition='val',mismatch=mismatch,wall_seconds=time.time()-start)
    write_json(target/'summary.json',s);return s

def run(args):
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    cfg=json.loads(Path(args.config).read_text());cache=Path(args.cache);out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    lock=(out/'trainer.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (out/'complete.json').exists():return
    data,split,_=experiment_data(cfg)
    assert cache_relevant(json.loads((cache/'identity.json').read_text())['config'])==cache_relevant(cfg),'Backbone cache configuration mismatch'
    grammar=json.loads((cache/'grammar.json').read_text());common=torch.load(cache/'common.pt',map_location='cpu',weights_only=False)
    initA,initB=actor_initialization(common,args.seed);A=initA.cuda().requires_grad_();B=initB.cuda().requires_grad_()
    refA,refB=initA.cuda(),initB.cuda();beta_ent=cfg.get('entropy_beta',.1);headroom=cfg.get('headroom_every',64)
    optimizer=(torch.optim.SGD if cfg.get('optimizer','adam')=='sgd' else torch.optim.Adam)([A,B],lr=cfg['learning_rate']);rng=np.random.default_rng(args.seed)
    extra=BaselineAdapter(grammar) if args.method in ('otb','rloo','relax') else None
    if extra:extra.gen.manual_seed(20260921+args.seed)
    q=None;poolhash=None;tablehash=None
    if args.method in ('qcv','row_delta','scalar_delta','vbase'):
        qdir=ROOT/cfg.get('offline_q','runs/offline-q-v1')
        tables,tablehash=load_frozen_tables(qdir/'model/selected-tables.json',split['train'],len(grammar['plans']))
        pool=json.loads((qdir/'data/pool.json').read_text());meta=json.loads((qdir/'model/complete.json').read_text());poolhash=digest(pool)
        assert poolhash==meta['pool_sha256'] and tablehash==meta['table_sha256']
        q=DynamicReward(pool,tables,data,grammar['plans'],steps=1000)
    ident=dict(version='kings-val128-n10-v1',method=args.method,seed=args.seed,actor_init_seed=42+args.seed,config=cfg,cache=common['identity'],entropy_beta=beta_ent,epochs=32,pool_sha256=poolhash,table_sha256=tablehash)
    identity=digest(ident);checkpoint=out/'checkpoint.pt';state=dict(updates=0,candidates=0,wall_seconds=0.,q_refit_seconds=0.)
    last_records=[];last_metric=None
    def save():
        atomic_save(checkpoint,dict(identity=identity,A=A.detach().cpu(),B=B.detach().cpu(),optimizer=optimizer.state_dict(),rng=rng.bit_generator.state,state=state,config=cfg,method=args.method,seed=args.seed,dynamic_q=q.state_dict() if q else None,baseline_extra=extra.state_dict() if extra else None,last_records=last_records,last_metric=last_metric))
    if checkpoint.exists():
        old=torch.load(checkpoint,map_location='cpu',weights_only=False);assert old['identity']==identity,'Run identity mismatch'
        with torch.no_grad():A.copy_(old['A']);B.copy_(old['B'])
        optimizer.load_state_dict(old['optimizer']);rng.bit_generator.state=old['rng'];state=old['state'];last_records=old['last_records'];last_metric=old['last_metric']
        if q:q.load_state_dict(old['dynamic_q'])
        if extra:extra.load_state_dict(old['baseline_extra'])
        recover_committed_logs(out,state,last_records,last_metric)
    else:save()
    write_json(out/'identity.json',dict(hash=identity,**ident));ev=BoundedEvaluator(cfg,args.docker_slots,args.slot_dir,out)
    def select_best(s,epoch):
        path=out/'best.json';prior=json.loads(path.read_text()) if path.exists() else None
        # Deterministic earlier-epoch tie break. best.pt is actor only, latest has resume state.
        if prior is None or s['mean_reward']>prior['validation']['mean_reward']:
            payload=dict(A=A.detach().cpu() if epoch else initA,B=B.detach().cpu() if epoch else initB,identity=identity,epoch=epoch,config=cfg,method=args.method,seed=args.seed)
            atomic_save(out/'best.pt',payload);write_json(path,dict(epoch=epoch,validation=s,checkpoint='best.pt'))
    atomic_save(out/'initial.pt',dict(A=initA,B=initB,identity=identity,epoch=0,config=cfg,method=args.method,seed=args.seed))
    s=evaluate(cfg,cache,grammar,common,refA,refB,data,split['val'],ev,out,'val-initial');select_best(s,0)
    def boundary():
        n=state['updates']
        if not n or n%64:return
        epoch=n//64
        atomic_save(out/f'epoch-{epoch}.pt',dict(A=A.detach().cpu(),B=B.detach().cpu(),identity=identity,epoch=epoch,config=cfg,method=args.method,seed=args.seed))
        if q and epoch<32 and q.refit_epoch<epoch:
            metrics,payload=q.refit(epoch);state['q_refit_seconds']+=metrics['wall_seconds'];save()
            write_json(out/f'q-refit-{epoch}.json',metrics)
        s=evaluate(cfg,cache,grammar,common,A,B,data,split['val'],ev,out,f'val-epoch-{epoch}');select_best(s,epoch)
        write_json(out/'progress.json',dict(updates=n,epoch=epoch,validation=s,updated_at=time.time()))
    boundary();schedule=grouped_batches(split['train'],32,args.seed,cfg['group_size'],cfg['instances_per_batch']);steps=0
    missing=sorted({n for b in schedule for n in b if not (cache/f'{n}.pt').exists()})
    if missing:raise RuntimeError(f'{len(missing)} scheduled instances have no cached features, e.g. {missing[:3]}')
    for batch in schedule[state['updates']:]:
        start=time.time();names=list(dict.fromkeys(batch));grad=[torch.zeros_like(A),torch.zeros_like(B)];records=[];diags=[]
        write_json(out/'pending.json',dict(update=state['updates']+1,names=names))
        for name in names:
            W=load_tree(cache,name,grammar,common);st=W.forward(A,B);episodes=[W.sample(st,rng) for _ in range(cfg['group_size'])];rewards=[];local=[]
            for e,path in episodes:
                action=int(W.conf[e]);r=ev.evaluate(build_code(grammar['plans'][action]),data[name],dict(stage='train',update=state['updates']+1,name=name,action=action))
                rewards.append(r['reward']);record=dict(name=name,action=action,update=state['updates']+1,**r);records.append(record);local.append(record)
            if extra:g,diag=extra.gradient(args.method,W,st,episodes,rewards,name if args.seed==0 else f'seed{args.seed}:{name}')
            elif args.method=='reinforce':g,diag=reinforce_gradient(W,st,episodes,rewards)
            else:
                table=q.tables[name] if q else np.zeros(len(grammar['plans']));ref=W.forward(refA,refB) if args.method=='grpo' else None
                g,diag=step_gradient(W,st,A,B,args.method,table,episodes,rewards,ref,cfg['grpo']['beta'],bool(headroom) and state['updates']%headroom==0)
                if q:diag.update(q.prediction_errors(local))
                del ref
            eg,ed=dispatch_entropy(W,st);diag.update(ed);diag['entropy_beta']=beta_ent
            for dest,value,ent in zip(grad,g,eg):dest.add_((value+beta_ent*ent)/len(names))
            diags.append(diag);del W,st,g,eg
        norm=float(sum(v.square().sum() for v in grad).sqrt())
        if not np.isfinite(norm):raise RuntimeError('Nonfinite gradient')
        optimizer.zero_grad(set_to_none=True);A.grad=-grad[0];B.grad=-grad[1];optimizer.step()
        if not torch.isfinite(A).all() or not torch.isfinite(B).all():raise RuntimeError('Nonfinite actor')
        if q:q.observe(records)
        if cfg.get('save_every') and (state['updates']+1)%cfg['save_every']==0:atomic_save(out/f"actor-{state['updates']+1}.pt",dict(A=A.detach().cpu(),B=B.detach().cpu(),update=state['updates']+1))   # per-update actors for exact learning curves
        state['updates']+=1;state['candidates']+=len(records);state['wall_seconds']+=time.time()-start
        last_records=records;last_metric=dict(method=args.method,seed=args.seed,counts=dict(state),mean_batch_reward=float(np.mean([r['reward'] for r in records])),gradient_norm=norm,diagnostics=diags,peak_gpu_bytes=torch.cuda.max_memory_allocated(),q_unique_labels=len(q.known) if q else 0)
        save()
        for r in records:append(out/'journal.jsonl',r)
        append(out/'metrics.jsonl',last_metric);write_json(out/'summary.json',last_metric);(out/'pending.json').unlink(missing_ok=True)
        print(json.dumps(dict(stage='train',method=args.method,seed=args.seed,update=state['updates'],reward=last_metric['mean_batch_reward'],seconds=time.time()-start)),flush=True)
        steps+=1;boundary()
        if args.stop_after and steps>=args.stop_after:return
    evaluate(cfg,cache,grammar,common,A,B,data,split['val'],ev,out,'val-final-mismatch',True,split['train'])
    write_json(out/'complete.json',dict(method=args.method,seed=args.seed,epochs=32,updates=2048,training_candidates=32768,validation_candidates=34*4*len(split['val']),test_candidates=0,q_refits=q.refit_epoch if q else 0,completed_at=time.time()))

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/synthetic-row-v1.json');p.add_argument('--cache',required=True);p.add_argument('--out',required=True);p.add_argument('--method',choices=METHODS,required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--docker-slots',type=int,default=16);p.add_argument('--slot-dir',required=True);p.add_argument('--stop-after',type=int,default=0);a=p.parse_args()
    try:run(a)
    except Exception as e:
        write_json(Path(a.out)/'failure.json',dict(error=str(e),time=time.time()));raise
if __name__=='__main__':main()
