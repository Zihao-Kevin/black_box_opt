"""Matched-budget JobShop training with upstream fitted-table scalar/row Delta.
Rewards are revealed only for sampled candidates, table updated after actor step.
No new-policy samples, labels or task evaluation are used when building the CV.
"""
import argparse,json,time,os
from pathlib import Path
import numpy as np
import torch
from .data import ROOT,experiment_data,grouped_batches,write_json,digest
from .row_tree import RowTree
from .evaluator import Evaluator
from .actions import build_code
from .dispatch_entropy import dispatch_entropy
from .baseline_adapter import BaselineAdapter


def atomic_save(path,payload):
    tmp=path.with_suffix('.tmp');torch.save(payload,tmp);tmp.replace(path)

def load_tree(cache,name,grammar,common):
    features=torch.load(cache/f'{name}.pt',weights_only=False,map_location='cpu')
    if features['identity']!=common['identity']:raise RuntimeError('Feature identity mismatch')
    return RowTree(grammar,features,common)

def step_gradient(W,st,A,B,method,table,episodes,rewards,reference=None,beta=.01,diagnose=False):
    zero=lambda:(torch.zeros_like(A),torch.zeros_like(B))
    result=list(zero());diagnostic={}
    f=W.rewards_to_entries(table);Q,V=W.values(st,f)
    c=Q
    if method=='vbase':c=V[W.ent_node]                                    # state-value baseline: sum_path (f - V_node) s_e, since sum_v P_v s_v = 0
    if diagnose and method in ('qcv','row_delta','scalar_delta'):
        # Exact noise budget of this update, from the same algebra the estimators use: what
        # REINFORCE and Q leave behind, and how much of Q's residual a row-wise residual can
        # still cancel.  q_residual_share near zero means the reward is settled before the
        # branches Delta corrects, and no number of seeds will separate Q from Q+Delta.
        (var_reinforce,var_q),grad_sq=W.noise(st,f,[torch.zeros_like(f),Q])
        residual,row_gain=W.rowwise(st,f)
        diagnostic.update(true_gradient_sq=grad_sq,var_reinforce=var_reinforce,var_q=var_q,
                          q_residual_share=var_q/var_reinforce if var_reinforce>0 else 0.,
                          row_delta_share=row_gain/residual if residual>0 else 0.)
    if method=='scalar_delta':
        Q,D,gain=W.delta(st,f);c=Q+D;diagnostic['fitted_delta_gain']=gain
    if method=='grpo':
        mean=float(np.mean(rewards));std=float(np.std(rewards));advantages=(np.array(rewards)-mean)/(std+1e-4)
        diagnostic['zero_reward_std_group_fraction']=float(std==0)
    for i,((e,path),reward) in enumerate(zip(episodes,rewards)):
        if method=='grpo':
            ix=torch.tensor(path,device=A.device);lr=(reference['P'][ix].log()-st['P'][ix].log())
            # One on-policy update: current/old ratio=1; sampled reverse-KL penalty
            # follows the existing GRPO derivative. Forced syntax scores are zero.
            length=W.token_lengths[int(W.conf[e])]
            coeff=(advantages[i]+beta*(lr.exp()-1))/length
            pair=W.grad_w(st,ix,coeff)
        else:
            pair=W.episode_grad(st,c,reward,path)
            if method=='row_delta':
                correction=W.row_correction(st,f,path);pair=tuple(x-y for x,y in zip(pair,correction))
            elif method not in ('qcv','scalar_delta','vbase'):raise ValueError(method)
        for total,value in zip(result,pair):total.add_(value/len(episodes))
    return result,diagnostic


def evaluate(cfg,cache,grammar,common,A,B,data,split,ev,out,label,mismatch=False):
    target=out/label
    if (target/'summary.json').exists():return
    target.mkdir(parents=True,exist_ok=True)
    if (target/'pending.json').exists():raise RuntimeError('Partial evaluation needs reconciliation')
    records=json.loads((target/'records.json').read_text()) if (target/'records.json').exists() else []
    done={r['name'] for r in records};started=time.time()
    for ni,name in enumerate(split['val']):
        if name in done:continue
        prompt_name=name
        if mismatch:
            shape=(len(data[name]['duration_matrix']),len(data[name]['duration_matrix'][0]))
            prompt_name=next(n for n in sorted(split['train']) if (len(data[n]['duration_matrix']),len(data[n]['duration_matrix'][0]))==shape)
        W=load_tree(cache,prompt_name,grammar,common);st=W.forward(A,B);rng=np.random.default_rng(10000+1000*ni)
        episodes=[W.sample(st,rng) for _ in range(4)]
        write_json(target/'pending.json',dict(name=name,actions=[int(W.conf[e]) for e,p in episodes]))
        group=[]
        for e,path in episodes:
            action=int(W.conf[e]);res=ev.evaluate(build_code(grammar['plans'][action]),data[name])
            if not res['valid']:raise RuntimeError(f'Constrained constructor returned invalid schedule: {res}')
            group.append(dict(action=action,**res))
        records.append(dict(name=name,prompt_name=prompt_name,candidates=group));write_json(target/'records.json',records);(target/'pending.json').unlink()
        del W,st
    rewards=[r['reward'] for x in records for r in x['candidates']]
    summary=dict(mean_reward=float(np.mean(rewards)),best_of_B=float(np.mean([max(r['reward'] for r in x['candidates']) for x in records])),candidate_evaluations=len(rewards),plan_legal_rate=1.,feasible_rate=1.,wall_seconds=time.time()-started,partition='val',B=4,mismatch=mismatch,score_mode='synthetic_lb')
    write_json(target/'summary.json',summary);print(json.dumps(dict(stage=label,**summary)),flush=True)


def load_frozen_tables(path,names,n_actions):
    if path is None:return None,None
    raw=json.loads(Path(path).read_text())
    if set(raw)!=set(names):raise ValueError('Offline table instance set mismatch')
    tables={n:np.asarray(raw[n],dtype=float) for n in names}
    if any(v.shape!=(n_actions,) or not np.isfinite(v).all() or (v<0).any() or (v>1).any() for v in tables.values()):raise ValueError('Invalid offline reward table')
    return tables,digest(raw)


def run(cfg,cache,out,method,stop_after=0,train_only=False,frozen_tables_path=None,entropy_beta=.10):
    torch.set_num_threads(2)
    if method not in ("otb","rloo","relax"):raise ValueError("Upstream baselines only")
    if not np.isfinite(entropy_beta) or entropy_beta<0:raise ValueError("Invalid entropy coefficient")
    out=Path(out);out.mkdir(parents=True,exist_ok=True);cache=Path(cache)
    torch.backends.cuda.matmul.allow_tf32=False
    data,split,_=experiment_data(cfg);grammar=json.loads((cache/'grammar.json').read_text());common=torch.load(cache/'common.pt',weights_only=False,map_location='cpu')
    from .kings_val128_train import cache_relevant
    if cache_relevant(json.loads((cache/'identity.json').read_text())['config'])!=cache_relevant(cfg):raise RuntimeError('Cache config mismatch')
    A=common['A0'].cuda().clone().requires_grad_();B=common['B0'].cuda().clone().requires_grad_()
    optimizer=torch.optim.Adam([A,B],lr=cfg['learning_rate']);rng=np.random.default_rng(cfg['seed']);extra=BaselineAdapter(grammar)
    sums={n:np.zeros(len(grammar['plans'])) for n in split['train']};counts={n:np.zeros(len(grammar['plans']),dtype=np.int64) for n in split['train']}
    frozen_tables,table_hash=load_frozen_tables(frozen_tables_path,split['train'],len(grammar['plans']))
    identity_fields=dict(config=cfg,method=method,cache=common['identity'],dispatch_entropy_beta=entropy_beta,total_epochs=32,upstream_baselines='33ffc0ab4d82c86e84dac722f5c71662d8a06dfe',prefix_state='depth64_known_fields33',surrogate_lr=.01)
    if table_hash is not None:identity_fields['frozen_table_sha256']=table_hash
    checkpoint=out/'checkpoint.pt';identity=digest(identity_fields)
    state=dict(updates=0,candidates=0,wall_seconds=0.)
    def save():
        atomic_save(checkpoint,dict(identity=identity,A=A.detach().cpu(),B=B.detach().cpu(),optimizer=optimizer.state_dict(),rng=rng.bit_generator.state,sums=sums,counts=counts,state=state,config=cfg,method=method,frozen_table_sha256=table_hash,dispatch_entropy_beta=entropy_beta,total_epochs=32,baseline_extra=extra.state_dict()))
    if checkpoint.exists():
        old=torch.load(checkpoint,weights_only=False,map_location='cpu')
        if old['identity']!=identity:raise RuntimeError('Resume identity mismatch')
        with torch.no_grad():A.copy_(old['A']);B.copy_(old['B'])
        optimizer.load_state_dict(old['optimizer']);rng.bit_generator.state=old['rng'];sums=old['sums'];counts=old['counts'];state=old['state'];extra.load_state_dict(old['baseline_extra'])
    else:save();atomic_save(out/'initial.pt',torch.load(checkpoint,weights_only=False))
    if (out/'pending.json').exists():raise RuntimeError('Uncommitted batch: explicit reconciliation required; no silent retries')
    ev=Evaluator(cfg['image'],cfg['timeout'],score_mode='synthetic_lb')
    if not train_only:evaluate(cfg,cache,grammar,common,common['A0'].cuda(),common['B0'].cuda(),data,split,ev,out,'val-initial')
    if not train_only:
        for epoch in range(1,state['updates']//64+1):
            prior=torch.load(out/f'epoch-{epoch}.pt',weights_only=False,map_location='cpu')
            evaluate(cfg,cache,grammar,common,prior['A'].cuda(),prior['B'].cuda(),data,split,ev,out,f'val-epoch-{epoch}')
    schedule=grouped_batches(split['train'],32,cfg['seed'],cfg['group_size'],cfg['instances_per_batch']);write_json(out/'schedule.json',schedule)
    steps=0
    for batch in schedule[state['updates']:]:
        started=time.time();names=list(dict.fromkeys(batch));grad=[torch.zeros_like(A),torch.zeros_like(B)];records=[];diags=[]
        write_json(out/'pending.json',dict(update=state['updates'],names=names,method=method))
        for name in names:
            W=load_tree(cache,name,grammar,common);st=W.forward(A,B)
            episodes=[W.sample(st,rng) for _ in range(cfg['group_size'])]
            # Upstream Seen.f_hat(zero=True): unit prior pseudo-count, zero reward.
            table=frozen_tables[name] if frozen_tables is not None else sums[name]/(counts[name]+cfg['reward_table_prior_count'])
            rewards=[]
            for e,path in episodes:
                action=int(W.conf[e]);res=ev.evaluate(build_code(grammar['plans'][action]),data[name])
                if not res['valid']:raise RuntimeError(f'Invalid constrained schedule {res}')
                rewards.append(res['reward']);record=dict(name=name,action=action,update=state['updates']+1,**res);records.append(record)
                with (out/'journal.jsonl').open('a') as stream:stream.write(json.dumps(record)+'\n');stream.flush();os.fsync(stream.fileno())
            ref=W.forward(common['A0'].cuda(),common['B0'].cuda()) if method=='grpo' else None
            g,diag=extra.gradient(method,W,st,episodes,rewards,name)
            entropy_grad,entropy_diag=dispatch_entropy(W,st)
            diag.update(entropy_diag);diag['entropy_beta']=entropy_beta
            diag['entropy_gradient_norm']=float(sum(v.square().sum() for v in entropy_grad).sqrt())
            for dest,value in zip(g,entropy_grad):dest.add_(value,alpha=entropy_beta)
            diags.append(diag)
            for dest,value in zip(grad,g):dest.add_(value/len(names))
            del W,st,g,ref
        norm=float(sum(g.square().sum() for g in grad).sqrt())
        if not np.isfinite(norm):raise RuntimeError('Nonfinite gradient')
        optimizer.zero_grad(set_to_none=True);A.grad=-grad[0];B.grad=-grad[1];optimizer.step()
        if not torch.isfinite(A).all() or not torch.isfinite(B).all():raise RuntimeError('Nonfinite parameters')
        # Newly revealed labels become available only for the NEXT actor update.
        for r in records:sums[r['name']][r['action']]+=r['reward'];counts[r['name']][r['action']]+=1
        state['updates']+=1;state['candidates']+=len(records);state['wall_seconds']+=time.time()-started;save()
        (out/'pending.json').unlink();steps+=1
        summary=dict(method=method,counts=state,mean_batch_reward=float(np.mean([r['reward'] for r in records])),gradient_norm=norm,diagnostics=diags,unique_observed_actions=sum(int((c>0).sum()) for c in counts.values()),peak_gpu_bytes=torch.cuda.max_memory_allocated())
        with (out/'metrics.jsonl').open('a') as stream:stream.write(json.dumps(summary)+'\n')
        write_json(out/'summary.json',summary);print(json.dumps(summary),flush=True)
        if state['candidates']%1024==0:
            epoch=state['candidates']//1024;atomic_save(out/f'epoch-{epoch}.pt',torch.load(checkpoint,weights_only=False))
            if not train_only:evaluate(cfg,cache,grammar,common,A,B,data,split,ev,out,f'val-epoch-{epoch}')
        if stop_after and steps>=stop_after:return
    if not train_only:
        evaluate(cfg,cache,grammar,common,A,B,data,split,ev,out,'val-final-mismatch',True)
        write_json(out/'complete.json',dict(training_candidates=state['candidates'],updates=state['updates'],validation_candidates=8704,test_candidates=0,method=method,frozen_table_sha256=table_hash,online_reward_model_refitting=False,offline_q_used=False,dispatch_entropy_beta=entropy_beta,total_epochs=32))

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--cache',required=True);p.add_argument('--out',required=True);p.add_argument('--method',choices=['otb','rloo','relax'],required=True);p.add_argument('--stop-after',type=int,default=0);p.add_argument('--train-only',action='store_true');p.add_argument('--frozen-tables');p.add_argument('--entropy-beta',type=float,default=.10);a=p.parse_args()
    run(json.loads(Path(a.config).read_text()),a.cache,a.out,a.method,a.stop_after,a.train_only,a.frozen_tables,a.entropy_beta)
if __name__=='__main__':main()
