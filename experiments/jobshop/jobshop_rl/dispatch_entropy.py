"""Exact marginal dispatch entropy and its gradient; no sampled entropy reward."""
import math
import torch

def dispatch_entropy(W,st):
    key=lambda p:'|'.join(p['dispatch']) if isinstance(p['dispatch'],list) else p['dispatch']
    labels=sorted({key(p) for p in W.grammar['plans']});lookup={s:i for i,s in enumerate(labels)}
    ids=torch.tensor([lookup[key(W.grammar['plans'][int(a)])] for a in W.conf[W.leaf_np]],device=W.device)
    leaf_probs=(st['pn'][W.ent_node]*st['P'])[W.leaf]
    marginal=torch.zeros(len(labels),dtype=leaf_probs.dtype,device=W.device).index_add_(0,ids,leaf_probs)
    logp=marginal.clamp_min(torch.finfo(marginal.dtype).tiny).log()
    H=-(marginal*logp).sum()
    f=torch.zeros(W.E,dtype=leaf_probs.dtype,device=W.device);f[W.leaf]=-logp[ids]
    # dH = sum_a dp(a) [-log p(dispatch(a))-1]; the -1 term sums to zero.
    grad=W.true_grad(st,f)
    return grad,dict(dispatch_entropy=float(H),dispatch_entropy_normalized=float(H)/math.log(len(labels)),dispatch_max_probability=float(marginal.max()),dispatch_effective_count=math.exp(float(H)),dispatch_probabilities=dict(zip(labels,marginal.cpu().tolist())))
