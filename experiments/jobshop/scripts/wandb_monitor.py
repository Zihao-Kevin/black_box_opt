"""Read-only live/backfill monitor. Does not import torch or modify trainer state."""
import argparse,hashlib,json,math,time,os,fcntl
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def read_json(path):
    try:return json.loads(path.read_text())
    except (FileNotFoundError,json.JSONDecodeError):return None

def lines(path):
    if not path.exists():return
    with path.open() as f:
        for line in f:
            try:yield json.loads(line)
            except json.JSONDecodeError:continue

def stages(beta,method):
    return [Path(os.environ.get('JOBSHOP_RUN_ROOT',str(ROOT/'runs/kings-beta010-val128-n10')))/f'seed-{RUN_SEED}'/method]

def train_metrics(s):
    state=s['counts'];step=state['updates']
    result={'train/update':step,'train/epoch':step/64,'train/candidates':state['candidates']}
    for key in ['mean_batch_reward','gradient_norm','unique_observed_actions','peak_gpu_bytes']:
        if key in s:result['train/'+key]=s[key]
    ds=s.get('diagnostics',[])
    for key in ['dispatch_entropy','dispatch_entropy_normalized','dispatch_max_probability','dispatch_effective_count','entropy_gradient_norm','zero_reward_std_group_fraction','relax_surrogate_loss','relax_temperature','reward_prediction_mse','unseen_reward_prediction_mse']:
        vals=[d[key] for d in ds if key in d]
        if vals:result['train/'+key]=sum(vals)/len(vals)
    for name in sorted({k for d in ds for k in d.get('dispatch_probabilities',{})}):
        vals=[d['dispatch_probabilities'][name] for d in ds if name in d.get('dispatch_probabilities',{})]
        result['train/dispatch_probability/'+name]=sum(vals)/len(vals)
    return result

def collect(beta,method):
    paths=stages(beta,method);training={};evaluation={}
    if beta=='0':
        for s in lines(ROOT/'reports/offline-q-v1-followup.log'):
            if s.get('method')==method and 'counts' in s:training[s['counts']['updates']]=train_metrics(s)
    plans=read_json(ROOT/'cache/synthetic-row-v1/grammar.json')['plans']
    for stage,path in enumerate(paths):
        for s in lines(path/'metrics.jsonl'):
            if 'counts' in s:training[s['counts']['updates']]=train_metrics(s)
        s=read_json(path/'summary.json')
        if s and 'counts' in s:training[s['counts']['updates']]=train_metrics(s)
        for f in path.glob('val-*/summary.json'):
            value=read_json(f)
            if value is None:continue
            label=f.parent.name
            if label=='val-initial':epoch=0
            elif label.startswith('val-epoch-'):epoch=int(label.rsplit('-',1)[1])
            elif label=='val-final-mismatch':epoch=32
            else:continue
            prefix='mismatch' if label.endswith('mismatch') else 'val'
            row={prefix+'/update':epoch*64,prefix+'/epoch':epoch,prefix+'/training_candidates':epoch*1024}
            for k in ['mean_reward','best_of_B','feasible_rate','plan_legal_rate']:
                if k in value:row[prefix+'/'+k]=value[k]
            records=read_json(f.parent/'records.json')
            if records:
                counts={};total=0
                for x in records:
                    for c in x['candidates']:
                        name=plans[c['action']]['dispatch'];counts[name]=counts.get(name,0)+1;total+=1
                for name in {p['dispatch'] for p in plans}:row[prefix+'/dispatch_fraction/'+name]=counts.get(name,0)/total
                row[prefix+'/sampled_dispatch_count']=len(counts)
            evaluation[(prefix,epoch)]=row
    events=[(f'train:{step}',row) for step,row in sorted(training.items())]
    events += [(f'{prefix}:{epoch}',row) for (prefix,epoch),row in sorted(evaluation.items(),key=lambda x:(x[0][1],x[0][0]))]
    # Separate x-axes let validation be backfilled after newer training updates.
    return events, all((p/'complete.json').exists() for p in paths)

def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--beta',choices=['0.10'],required=True);p.add_argument('--method',choices=['qcv','row_delta','grpo','relax','rloo','otb','reinforce'],required=True);p.add_argument('--entity');p.add_argument('--project',default='jobshop-planner-rl');p.add_argument('--mode',choices=['online','offline'],default='online');p.add_argument('--once',action='store_true');a=p.parse_args()
    global RUN_SEED
    RUN_SEED=a.seed
    import wandb
    folder=ROOT/'runs/wandb-monitor';folder.mkdir(exist_ok=True)
    key=f"{a.entity or 'default'}:{a.project}:kings-val128-n10:{a.beta}:{a.method}:seed{a.seed}:{os.environ.get('JOBSHOP_RUN_ROOT', 'original')}:v1";rid=hashlib.sha256(key.encode()).hexdigest()[:16]
    lock=(folder/f'{rid}-{a.mode}.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    cursor_path=folder/f'{rid}-{a.mode}.json';cursor=read_json(cursor_path) or {'seen':[]};seen=set(cursor['seen'])
    run=wandb.init(entity=a.entity,project=a.project,id=rid,resume='allow' if a.mode=='online' else None,mode=a.mode,name=f'Kings {a.method.upper()} seed={a.seed} beta={a.beta}',group='kings-beta010-val128-n10',job_type='metrics-monitor',dir=str(folder),config=dict(method=a.method,entropy_beta=float(a.beta),seed=a.seed,actor_init_seed=42+a.seed,host="kings",train_instances=128,val_instances=128,group_size=8,instances_per_batch=2,target_epochs=32 if a.beta=='0.10' else 4,reward='LB/makespan',offline_unique_labels=8023 if a.method in ('qcv','row_delta') else 0,offline_new_solver_calls=0,offline_q_used=a.method in ('qcv','row_delta'),online_q_refit=a.method in ('qcv','row_delta'),reference_kl_beta=.01 if a.method=='grpo' else 0.,upstream_commit="33ffc0ab4d82c86e84dac722f5c71662d8a06dfe",metrics_source='existing trainer logs; no training restart'),settings=wandb.Settings(disable_git=True,disable_code=True,console='off',x_disable_stats=True))
    for prefix in ['train','val','mismatch']:
        run.define_metric(prefix+'/update');run.define_metric(prefix+'/*',step_metric=prefix+'/update')
    run.summary['metrics_note']='Train entropy is batch-instance mean. Validation dispatch fractions are sampled frequencies. Continuation uses cumulative updates. Monitor process hardware is not trainer hardware.'
    print(json.dumps({'run_id':rid,'url':run.url,'method':a.method,'beta':a.beta}),flush=True)
    while True:
        events,done=collect(a.beta,a.method)
        for eid,row in events:
            if eid in seen:continue
            clean={k:v for k,v in row.items() if not isinstance(v,float) or math.isfinite(v)}
            run.log(clean);seen.add(eid)
        run.summary['training_status']='complete' if done else 'running_or_queued'
        tmp=cursor_path.with_suffix('.tmp');tmp.write_text(json.dumps(dict(seen=sorted(seen),run_id=rid,url=run.url)));tmp.replace(cursor_path)
        if done or a.once:break
        time.sleep(60)
    run.finish()
if __name__=='__main__':main()
