"""JobShop adapter for upstream baselines.py at 33ffc0ab4d82c86e84dac722f5c71662d8a06dfe."""
import hashlib
import numpy as np
import torch
from .actions import DISPATCH,TIES,POST
from .upstream_baselines import group_weights,relax_weights,Surrogate


def prefix_states(grammar):
    d0=grammar['plans'][0]['dispatch'];G=len(d0) if isinstance(d0,list) else 0
    fields=([(('dispatch',g),list(DISPATCH)) for g in range(G)] if G else [('dispatch',list(DISPATCH))])+[('tie_break',list(TIES)),('postprocess',list(POST)),('search_radius',[0,2,4,8])]
    get=lambda a,field:grammar['plans'][a][field[0]][field[1]] if isinstance(field,tuple) else grammar['plans'][a][field]
    nodes=len(grammar['parents']);desc=[set() for _ in range(nodes)]
    edges=[[] for _ in range(nodes)]
    for e,n in enumerate(grammar['entry_node']):edges[n].append(e)
    state=np.zeros((nodes,64+sum(len(values) for _,values in fields)),dtype=np.float64)
    for n in range(nodes-1,-1,-1):
        for e in edges[n]:
            child=grammar['entry_child'][e]
            if child<0:desc[n].add(grammar['conf'][e])
            else:desc[n].update(desc[child])
        state[n,min(grammar['depths'][n],63)]=1.;offset=64
        for field,values in fields:
            possible={get(a,field) for a in desc[n]}
            if len(possible)==1:state[n,offset+values.index(next(iter(possible)))]=1.
            offset+=len(values)
    return state


def padded_positions(episodes,device):
    T=max(len(p) for _,p in episodes);pos=torch.full((len(episodes),T),-1,dtype=torch.long,device=device)
    for i,(_,path) in enumerate(episodes):pos[i,:len(path)]=torch.tensor(path[::-1],device=device)
    return pos


def relax_batch(W,st,episodes,prefix):
    pos=padded_positions(episodes,W.device);K,T=pos.shape;branches=max(W.by_k)
    ent=torch.full((K,T,branches),-1,dtype=torch.long,device=W.device);b=torch.zeros((K,T),dtype=torch.long,device=W.device)
    state=torch.zeros((K,T,prefix.shape[-1]),dtype=torch.float64,device=W.device)
    for i,(_,path) in enumerate(episodes):
        for t,e in enumerate(path[::-1]):
            n=int(W.en_np[e]);start,end=int(W.start_np[n]),int(W.start_np[n+1]);ent[i,t,:end-start]=torch.arange(start,end,device=W.device);b[i,t]=e-start;state[i,t]=prefix[n]
    logp=st['P'].clamp_min(1e-300).log()[ent.clamp_min(0)].double()
    return ent,b,logp,state


class BaselineAdapter:
    def __init__(self,grammar,device='cuda'):
        self.device=device;self.prefix=torch.as_tensor(prefix_states(grammar),device=device);self.dim=self.prefix.shape[1]+max(np.bincount(grammar['entry_node']))
        self.surrogates={};self.gen=torch.Generator(device=device).manual_seed(20260921)

    def surrogate(self,name):
        if name not in self.surrogates:
            seed=int(hashlib.sha256(('relax-init:'+name).encode()).hexdigest()[:8],16)
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed);c=Surrogate(self.dim).to(self.device).double()
            c.opt=torch.optim.Adam(c.parameters(),lr=1e-2);self.surrogates[name]=c
        return self.surrogates[name]

    def gradient(self,method,W,st,episodes,rewards,name):
        f=torch.as_tensor(rewards,dtype=torch.float64,device=W.device)
        if method in ('otb','rloo'):
            pos=padded_positions(episodes,W.device);w=group_weights(method.upper(),f,(pos>=0).double(),pos,W.score_norms(st));nz=w.nonzero().flatten()
            return W.grad_w(st,nz,w[nz]),{}
        if method!='relax':raise ValueError(method)
        ent,b,lp,state=relax_batch(W,st,episodes,self.prefix);c=self.surrogate(name)
        w=relax_weights(c,lp,ent,b,f,W.E,state,gen=self.gen)
        if not torch.isfinite(w).all():raise RuntimeError('Nonfinite RELAX coefficients')
        nz=(w.detach()!=0).nonzero().flatten();ga,gb=W.grad_w(st,nz,w[nz]);loss=ga.square().sum()+gb.square().sum()
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite RELAX surrogate loss')
        # Compute this batch actor estimate before fitting phi for subsequent use.
        result=[ga.detach(),gb.detach()]
        c.opt.zero_grad();loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in c.parameters()):raise RuntimeError('Nonfinite RELAX surrogate gradient')
        c.opt.step()
        if any(not torch.isfinite(p).all() for p in c.parameters()):raise RuntimeError('Nonfinite RELAX surrogate parameters')
        return result,dict(relax_surrogate_loss=float(loss.detach()),relax_temperature=float(c.log_tau.detach().exp()))

    def state_dict(self):
        return dict(generator=self.gen.get_state(),surrogates={n:dict(model=c.state_dict(),optimizer=c.opt.state_dict()) for n,c in self.surrogates.items()})

    def load_state_dict(self,s):
        self.gen.set_state(s['generator'].cpu())
        for n,payload in s['surrogates'].items():
            c=self.surrogate(n);c.load_state_dict(payload['model']);c.opt.load_state_dict(payload['optimizer'])
