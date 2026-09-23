import numpy as np
import torch
from jobshop_rl.row_tree import RowTree,make_grammar
from jobshop_rl.actions import validate_plan

class CharTokenizer:
    eos_token_id=0
    def encode(self,s,**kwargs):return [ord(c) for c in s]

def tiny():
    torch.manual_seed(83)
    grammar=dict(prefixes=[[],[0],[1]],parents=[-1,0,1],depths=[0,1,1],entry_node=[0,0,1,1,2,2],entry_token=[0,1,0,1,0,1],entry_child=[1,2,-1,-1,-1,-1],conf=[-1,-1,0,1,2,3])
    cache=dict(X=torch.randn(3,5),V0=torch.randn(3,3))
    common=dict(norm=torch.rand(3)+.5,eps=1e-6,scaling=2.,tokens=[0,1],W=torch.randn(2,3))
    return RowTree(grammar,cache,common,'cpu'),torch.randn(2,5)*.2,torch.randn(3,2)*.2

def paths():return [[2,0],[3,0],[4,1],[5,1]]

def test_all_actions_reachable_once():
    g=make_grammar(CharTokenizer());assert len(g['plans'])==1500 and all(map(validate_plan,g['plans']))
    assert sorted(c for c in g['conf'] if c>=0)==list(range(1500))
    for i,p in enumerate(g['prefixes']):
        parent=g['parents'][i]
        if parent>=0:
            pre=g['prefixes'][g['entry_node'][parent]]+[g['entry_token'][parent]]
            assert p[:len(pre)]==pre

def test_scores_match_autograd():
    W,A,B=tiny();st=W.forward(A,B);aa=A.clone().requires_grad_();bb=B.clone().requires_grad_()
    v=W.V0+W.s*(W.X['instance']@aa.T)@bb.T
    h=W.g*v/(v.square().mean(-1,keepdim=True)+W.eps).sqrt()
    logits=h@W.W.T
    for e in range(W.E):
        n=W.en_np[e];token=int(W.ent_tok[e]);lp=logits[n].log_softmax(-1)[token]
        ga,gb=torch.autograd.grad(lp,(aa,bb),retain_graph=True)
        actual=W.grad_w(st,torch.tensor([e]),torch.ones(1))
        torch.testing.assert_close(actual[0],ga,atol=1e-6,rtol=1e-5);torch.testing.assert_close(actual[1],gb,atol=1e-6,rtol=1e-5)

def test_row_correction_unbiased_and_reduces_oracle_variance():
    W,A,B=tiny();st=W.forward(A,B);truth=W.rewards_to_entries([.2,.8,.1,.6]);fitted=W.rewards_to_entries([.4,.1,.7,.9])
    probs=(st['pn'][W.ent_node]*st['P'])[W.leaf]
    flatten=lambda pair:torch.cat([p.flatten() for p in pair]).double()
    target=flatten(W.true_grad(st,truth))
    for table in [truth,fitted]:
        Q,_=W.values(st,table);base=[];corrected=[];corrections=[]
        for path in paths():
            g=flatten(W.episode_grad(st,Q,truth[path[0]],path));c=flatten(W.row_correction(st,table,path))
            base.append(g);corrected.append(g-c);corrections.append(c)
        gb=torch.stack(base);gr=torch.stack(corrected);gc=torch.stack(corrections)
        torch.testing.assert_close(probs@gc,torch.zeros_like(target),atol=2e-6,rtol=1e-5)
        torch.testing.assert_close(probs@gr,target,atol=2e-6,rtol=1e-5)
        vb=(probs[:,None]*(gb-target).square()).sum();vr=(probs[:,None]*(gr-target).square()).sum()
        if table is truth:
            total,gain=W.rowwise(st,truth)
            torch.testing.assert_close(vb,torch.tensor(total,dtype=vb.dtype),atol=2e-6,rtol=1e-5)
            torch.testing.assert_close(vr,torch.tensor(total-gain,dtype=vr.dtype),atol=2e-6,rtol=1e-5)
            assert vr<=vb+1e-7

def test_scalar_and_q_unbiased_for_arbitrary_fitted_table():
    W,A,B=tiny();st=W.forward(A,B);truth=W.rewards_to_entries([.2,.8,.1,.6]);fitted=W.rewards_to_entries([.4,.1,.7,.9]);Q,D,_=W.delta(st,fitted)
    probs=(st['pn'][W.ent_node]*st['P'])[W.leaf];flat=lambda p:torch.cat([x.flatten() for x in p]).double()
    for c in [Q,Q+D]:
        estimates=torch.stack([flat(W.episode_grad(st,c,truth[p[0]],p)) for p in paths()])
        torch.testing.assert_close(probs@estimates,flat(W.true_grad(st,truth)),atol=2e-6,rtol=1e-5)

def test_table_prior_and_group_gradient_averaging():
    from jobshop_rl.baseline_train import step_gradient
    W,A,B=tiny();st=W.forward(A,B);W.token_lengths=[2]*4
    eps=[(p[0],p) for p in paths()];rewards=[.2,.8,.1,.6];table=np.array([.4,.1,.7,.9])
    for method in ['qcv','scalar_delta','row_delta']:
        batched,_=step_gradient(W,st,A,B,method,table,eps,rewards)
        singles=[step_gradient(W,st,A,B,method,table,[e],[r])[0] for e,r in zip(eps,rewards)]
        for k in [0,1]:torch.testing.assert_close(batched[k],sum(s[k] for s in singles)/4)
    grpo,diag=step_gradient(W,st,A,B,'grpo',table,eps,[.5]*4,st)
    assert diag['zero_reward_std_group_fraction']==1 and all(torch.count_nonzero(g)==0 for g in grpo)
