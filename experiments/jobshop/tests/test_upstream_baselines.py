import copy,itertools
import numpy as np
import torch
from test_row_tree import tiny,paths
from jobshop_rl.baseline_adapter import BaselineAdapter,padded_positions,relax_batch
from jobshop_rl.upstream_baselines import relax_weights


def fixture():
    W,A,B=tiny();W.grammar['plans']=[dict(dispatch=d,tie_break='job_id',postprocess='none',search_radius=0) for d in ['est_spt','spt','lpt','mwr']]
    return W,A,B,BaselineAdapter(W.grammar,'cpu')


def test_rloo_exact_expectation_matches_true_gradient():
    W,A,B,extra=fixture();st=W.forward(A,B);rewards=[.2,.8,.1,.6];p=(st['pn'][W.ent_node]*st['P'])[W.leaf];avg=[torch.zeros_like(A),torch.zeros_like(B)]
    for i,j in itertools.product(range(4),repeat=2):
        eps=[(paths()[k][0],paths()[k]) for k in [i,j]];g,_=extra.gradient('rloo',W,st,eps,[rewards[i],rewards[j]],'a')
        for acc,v in zip(avg,g):acc.add_(v*float(p[i]*p[j]))
    target=W.true_grad(st,W.rewards_to_entries(rewards))
    for actual,want in zip(avg,target):torch.testing.assert_close(actual,want,atol=2e-6,rtol=2e-5)


def test_otb_matches_inclusive_energy_formula_and_direction():
    W,A,B,extra=fixture();st=W.forward(A,B);eps=[(p[0],p) for p in paths()];rewards=torch.tensor([.2,.8,.1,.6],dtype=torch.float64);pos=padded_positions(eps,'cpu');energy=W.score_norms(st)[pos].cumsum(1);baseline=(energy*rewards[:,None]).sum(0)/energy.sum(0)
    target=[torch.zeros_like(A),torch.zeros_like(B)]
    for i in range(4):
        g=W.grad_w(st,pos[i],rewards[i]-baseline)
        for x,y in zip(target,g):x.add_(y/4)
    g,_=extra.gradient('otb',W,st,eps,rewards,'a')
    for x,y in zip(g,target):torch.testing.assert_close(x,y)
    assert pos[0].tolist()==[0,2]


def test_relax_save_restore_and_finite_surrogate_update():
    torch.set_num_threads(1);W,A,B,extra=fixture();st=W.forward(A,B);eps=[(p[0],p) for p in paths()];rewards=[.2,.8,.1,.6]
    extra.gradient('relax',W,st,eps,rewards,'a');saved=copy.deepcopy(extra.state_dict());g,d=extra.gradient('relax',W,st,eps,rewards,'a')
    other=BaselineAdapter(W.grammar,'cpu');other.load_state_dict(saved);g2,d2=other.gradient('relax',W,st,eps,rewards,'a')
    for x,y in zip(g,g2):torch.testing.assert_close(x,y,atol=0,rtol=0);assert torch.isfinite(x).all()
    assert d==d2 and d['relax_temperature']>0


def test_relax_frozen_surrogate_monte_carlo_mean():
    torch.set_num_threads(1);W,A,B,extra=fixture();st=W.forward(A,B);truth=[.2,.8,.1,.6];prob=(st['pn'][W.ent_node]*st['P'])[W.leaf].numpy();rng=np.random.default_rng(17);c=extra.surrogate('audit');samples=[]
    for _ in range(40):
        actions=rng.choice(4,size=256,p=prob);eps=[(paths()[i][0],paths()[i]) for i in actions];ent,b,lp,state=relax_batch(W,st,eps,extra.prefix)
        w=relax_weights(c,lp,ent,b,torch.tensor([truth[i] for i in actions],dtype=torch.float64),W.E,state,gen=extra.gen)
        g=W.grad_w(st,torch.arange(W.E),w);samples.append(torch.cat([x.detach().flatten() for x in g]))
    samples=torch.stack(samples);target=torch.cat([x.flatten() for x in W.true_grad(st,W.rewards_to_entries(truth))]);se=samples.std(0)/np.sqrt(len(samples))
    assert torch.all((samples.mean(0)-target).abs()<6*se+1e-4)
