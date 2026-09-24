"""Train-only terminal reward regression, independent action holdout and group CV.
No instance ID/family label or solver output enters the features. All labels come
from the frozen offline pool. Predictions define one coherent terminal table.
"""
import argparse,hashlib,json,math,time
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .data import ROOT,experiment_data,write_json,digest
from .actions import DISPATCH,TIES


def stable_key(value):return hashlib.sha256(str(value).encode()).hexdigest()

def dispatch_key(plan):
    """Stratum for the balanced holdout and the per-rule CV baseline. With one rule per machine group every
    full combination is a singleton in a random pool, so stratify by the first group's rule instead."""
    d=plan['dispatch'];return d[0] if isinstance(d,list) else d

def instance_features(x):
    d=np.asarray(x['duration_matrix'],dtype=float);m=np.asarray(x['machines_matrix'],dtype=int);n,k=d.shape;avg=d.mean();total=d.sum()
    jobs=d.sum(1);loads=np.bincount(m.ravel(),weights=d.ravel(),minlength=k)
    normalized=lambda z:list(np.quantile(z/np.mean(z),[0,.1,.25,.5,.75,.9,1]))+[float(np.std(z)/np.mean(z))]
    f=[n/30,k/15,math.log1p(avg)/6,n/k/6,*normalized(d.ravel()),*normalized(jobs),*normalized(loads),max(jobs.max(),loads.max())/total]
    # Machine labels themselves are arbitrary: use counts and job-route agreement.
    for position in [0,k//2,k-1]:
        counts=np.bincount(m[:,position],minlength=k)/n;nonzero=counts[counts>0]
        f.extend([counts.max(),float(-(nonzero*np.log(nonzero)).sum()/max(math.log(k),1)),float((d[:,position]/avg).mean())])
    f.extend([float(np.mean([np.mean(m[i]==m[j]) for i in range(n) for j in range(i)])),float((d[:,:max(1,k//2)].sum()/total)),float(np.corrcoef(d.ravel(),np.tile(np.arange(k),n))[0,1]) if np.std(d)>0 else 0.])
    return np.asarray(f,dtype=np.float32)

def action_features(plan):
    def one(value,values):return [float(value==x) for x in values]
    d=plan['dispatch'];dispatch=sum((one(r,list(DISPATCH)) for r in d),[]) if isinstance(d,list) else one(d,list(DISPATCH))
    return np.asarray(dispatch+one(plan['tie_break'],list(TIES))+one(plan['postprocess'],['none','local_sort','local_insert','local_swap'])+one(plan['search_radius'],[0,2,4,8]),dtype=np.float32)

def features(inst,act):
    # Explicit dispatch/instance interactions let rules depend on instance structure.
    n=len(act)-(len(TIES)+4+4)                                   # the dispatch block: 15 per rule slot
    return np.concatenate([inst,act,np.outer(act[:n],inst).ravel()]).astype(np.float32)

def pair_split(rows,plans):
    groups=defaultdict(list)
    for i,row in enumerate(rows):groups[(row['name'],dispatch_key(plans[row['action']]))].append(i)
    fit=[];dev=[]
    for group,indices in sorted(groups.items()):
        order=sorted(indices,key=lambda i:stable_key(('action-holdout-1',rows[i]['name'],rows[i]['action'])))
        n=max(1,int(len(order)*.2)) if len(order)>=2 else 0
        dev.extend(order[:n]);fit.extend(order[n:])
    return np.array(sorted(fit)),np.array(sorted(dev))

def instance_folds(names,data):
    strata=defaultdict(list)
    for n in names:
        x=data[n];strata[(len(x['duration_matrix']),len(x['duration_matrix'][0]),x['metadata']['family'])].append(n)
    folds={}
    for ns in strata.values():
        for i,n in enumerate(sorted(ns,key=lambda n:stable_key(('instance-fold-1',n)))):folds[n]=i%4
    return folds

def balanced_mse(rows,plans,pred):
    groups=defaultdict(list)
    for row,p in zip(rows,pred):groups[(row['name'],dispatch_key(plans[row['action']]))].append((float(p)-row['reward'])**2)
    return float(np.mean([np.mean(v) for v in groups.values()]))

class RewardModel(nn.Module):
    def __init__(self,dim,mean,std,prior,hidden=(128,64)):
        super().__init__();self.register_buffer('mean',torch.as_tensor(mean,dtype=torch.float32));self.register_buffer('std',torch.as_tensor(std,dtype=torch.float32))
        layers=[];width=dim
        for h in hidden:layers+=[nn.Linear(width,h),nn.Tanh()];width=h
        self.net=nn.Sequential(*layers,nn.Linear(width,1))
        nn.init.zeros_(self.net[-1].weight);nn.init.constant_(self.net[-1].bias,math.log(prior/(1-prior)))
    def forward(self,x):return self.net((x-self.mean)/self.std).squeeze(-1).sigmoid()


def fit_model(X,y,rows,plans,seed,steps,dev=None,hidden=(128,64)):
    torch.manual_seed(seed);rng=np.random.default_rng(seed);mean=X.mean(0);std=np.maximum(X.std(0),.05);prior=float(np.clip(y.mean(),.01,.99))
    model=RewardModel(X.shape[1],mean,std,prior,hidden);opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-3)
    xx=torch.from_numpy(X);yy=torch.from_numpy(y);groups=defaultdict(list)
    for i,row in enumerate(rows):groups[(row['name'],dispatch_key(plans[row['action']]))].append(i)
    groups=list(groups.values());best=None;best_mse=float('inf');best_step=steps;history=[]
    for step in range(1,steps+1):
        chosen=rng.integers(len(groups),size=256);ix=np.array([groups[g][rng.integers(len(groups[g]))] for g in chosen]);loss=(model(xx[ix])-yy[ix]).square().mean()
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite regression loss')
        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
        if dev is not None and (step%50==0 or step==steps):
            with torch.no_grad():prediction=model(torch.from_numpy(dev[0])).numpy()
            mse=balanced_mse(dev[2],plans,prediction);history.append(dict(step=step,balanced_dev_mse=mse,train_batch_mse=float(loss.detach())))
            if mse<best_mse:best_mse=mse;best_step=step;best={k:v.detach().clone() for k,v in model.state_dict().items()}
    if best is not None:model.load_state_dict(best)
    model.eval();return model,best_step,history


def mean_predictions(fit,targets):
    per=defaultdict(list)
    for r in fit:per[r['name']].append(r['reward'])
    fallback=float(np.mean([r['reward'] for r in fit]));return np.array([np.mean(per[r['name']]) if r['name'] in per else fallback for r in targets])


def train(cfg,pool_path,out,grammar_path=None):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():return
    if (out/'started.json').exists():raise RuntimeError('Partial model fit requires reconciliation, not hidden refitting')
    hidden=tuple(cfg.get('reward_model_hidden',(128,64)));steps_max=int(cfg.get('reward_model_steps',1000))
    rows=json.loads(Path(pool_path).read_text());data,split,_=experiment_data(cfg);grammar=json.loads(Path(grammar_path or ROOT/'cache/synthetic-row-v1/grammar.json').read_text());plans=grammar['plans']
    if any(r['name'] not in set(split['train']) for r in rows):raise ValueError('Non-training label in offline pool')
    if len({(r['name'],r['action']) for r in rows})!=len(rows):raise ValueError('Pool not deduplicated')
    write_json(out/'started.json',dict(pool_sha256=digest(rows),seed=431,steps=steps_max,hidden=list(hidden),batch=256,lr=.001,weight_decay=.001,selection='within-instance dispatch-balanced dev MSE; model must also beat global dispatch means in fixed-step grouped CV'))
    inst={n:instance_features(data[n]) for n in split['train']};acts=[action_features(p) for p in plans]
    X=np.stack([features(inst[r['name']],acts[r['action']]) for r in rows]);y=np.array([r['reward'] for r in rows],dtype=np.float32)
    fit,dev=pair_split(rows,plans);fitrows=[rows[i] for i in fit];devrows=[rows[i] for i in dev]
    write_json(out/'splits.json',dict(fit_indices=fit.tolist(),dev_indices=dev.tolist(),instance_folds=instance_folds(split['train'],data),pool_sha256=digest(rows)))
    model,steps,history=fit_model(X[fit],y[fit],fitrows,plans,431,steps_max,dev=(X[dev],y[dev],devrows),hidden=hidden)
    with torch.no_grad():pred=model(torch.from_numpy(X[dev])).numpy()
    mean_pred=mean_predictions(fitrows,devrows)
    metrics=dict(model_balanced_mse=balanced_mse(devrows,plans,pred),mean_balanced_mse=balanced_mse(devrows,plans,mean_pred),model_mse=float(np.mean((pred-y[dev])**2)),mean_mse=float(np.mean((mean_pred-y[dev])**2)),selected_steps=steps)
    write_json(out/'fit-history.json',history);write_json(out/'action-holdout.json',metrics);print(json.dumps(dict(stage='action_holdout',**metrics)),flush=True)
    # Fixed 600 steps, independent of action-holdout selection: do not let group-CV
    # held-out labels influence early stopping or training-length selection.
    folds=instance_folds(split['train'],data);cross=[]
    for fold in range(4):
        tr=np.array([i for i,r in enumerate(rows) if folds[r['name']]!=fold]);te=np.array([i for i,r in enumerate(rows) if folds[r['name']]==fold]);trrows=[rows[i] for i in tr];terows=[rows[i] for i in te]
        cv,_,_=fit_model(X[tr],y[tr],trrows,plans,500+fold,max(1,steps_max*3//5),hidden=hidden)
        with torch.no_grad():cp=cv(torch.from_numpy(X[te])).numpy()
        rule=defaultdict(list)
        for rr in trrows:rule[dispatch_key(plans[rr['action']])].append(rr['reward'])
        rp=np.array([np.mean(rule[dispatch_key(plans[rr['action']])]) if dispatch_key(plans[rr['action']]) in rule else float(np.mean([r['reward'] for r in trrows])) for rr in terows])
        cross.append(dict(fold=fold,train_instances=len({r['name'] for r in trrows}),test_instances=len({r['name'] for r in terows}),model_balanced_mse=balanced_mse(terows,plans,cp),global_dispatch_balanced_mse=balanced_mse(terows,plans,rp)))
        print(json.dumps(dict(stage='instance_cv',**cross[-1])),flush=True)
    selected='learned' if metrics['model_balanced_mse']<metrics['mean_balanced_mse'] and np.mean([x['model_balanced_mse'] for x in cross])<np.mean([x['global_dispatch_balanced_mse'] for x in cross]) else 'mean'
    write_json(out/'instance-cv.json',cross)
    final,_,_=fit_model(X,y,rows,plans,431,steps,hidden=hidden);torch.save(dict(state=final.state_dict(),dim=X.shape[1],steps=steps,hidden=list(hidden),pool_sha256=digest(rows)),out/'reward-model.pt')
    lookup=defaultdict(dict)
    for row in rows:lookup[row['name']][row['action']]=row['reward']
    tables={'mean':{},'learned':{}};means={}
    for name in split['train']:
        means[name]=float(np.mean(list(lookup[name].values())));mean=np.full(len(plans),means[name]);xx=np.stack([features(inst[name],a) for a in acts])
        with torch.no_grad():learned=final(torch.from_numpy(xx)).numpy().astype(float)
        for action,reward in lookup[name].items():mean[action]=reward;learned[action]=reward
        tables['mean'][name]=mean.tolist();tables['learned'][name]=learned.tolist()
    for name,value in tables.items():write_json(out/f'{name}-tables.json',value)
    write_json(out/'selected-tables.json',tables[selected]);write_json(out/'offline-means.json',means)
    write_json(out/'complete.json',dict(selected=selected,selection_frozen_before_audit=True,table_sha256=digest(tables[selected]),mean_table_sha256=digest(tables['mean']),learned_table_sha256=digest(tables['learned']),pool_sha256=digest(rows),unique_labels=len(rows),action_holdout=metrics,instance_cv=cross,known_rewards_exact=True,online_refitting=False))
    print((out/'complete.json').read_text(),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--pool',required=True);p.add_argument('--out',required=True);p.add_argument('--grammar');a=p.parse_args();torch.set_num_threads(2);train(json.loads(Path(a.config).read_text()),a.pool,a.out,a.grammar)
if __name__=='__main__':main()
