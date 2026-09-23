"""Canonical JSON grammar; exact last-layer LoRA policy and upstream row-wise CV.
Forced grammar tokens have probability one. Branch logits are normalized over the
allowed next tokens, NOT over the original unrestricted vocabulary. Every action
remains reachable. Frozen backbone prefix features are exact for last down_proj.
"""
import json
import numpy as np
import torch
from .actions import DISPATCH,TIES
from .row_math import TreeMath


def make_grammar(tok):
    plans=[];texts=[];root={}
    for dispatch in DISPATCH:
        for tie in TIES:
            for post,radius in [('none',0)]+[(p,r) for p in ('local_sort','local_insert','local_swap') for r in (2,4,8)]:
                plan=dict(dispatch=dispatch,tie_break=tie,postprocess=post,search_radius=radius)
                text=json.dumps(plan,separators=(',',':'));tokens=tok.encode(text,add_special_tokens=False)+[tok.eos_token_id]
                node=root
                for token in tokens:node=node.setdefault(token,{})
                if node:raise ValueError('Nonunique canonical token sequence')
                node['leaf']=len(plans);plans.append(plan);texts.append(text)
    def skip(node,prefix):
        while 'leaf' not in node and len(node)==1:
            token,child=next(iter(node.items()));prefix=prefix+(token,);node=child
        return node,prefix
    prefixes=[];parents=[];depths=[];en=[];et=[];ec=[];conf=[]
    def walk(node,prefix,parent,depth):
        node,prefix=skip(node,prefix)
        if 'leaf' in node:return -1,node['leaf']
        n=len(prefixes);prefixes.append(list(prefix));parents.append(parent);depths.append(depth)
        todo=[]
        for token,child in sorted(node.items()):
            e=len(en);en.append(n);et.append(token);ec.append(-1);conf.append(-1);todo.append((e,token,child))
        for e,token,child in todo:ec[e],conf[e]=walk(child,prefix+(token,),e,depth+1)
        return n,-1
    walk(root,(),-1,0)
    return dict(plans=plans,texts=texts,prefixes=prefixes,parents=parents,depths=depths,entry_node=en,entry_token=et,entry_child=ec,conf=conf)


class RowTree(TreeMath):
    def __init__(self,grammar,cache,common,device='cuda'):
        self.device=device;T=lambda x,dtype=torch.long:torch.tensor(x,dtype=dtype,device=device)
        self.grammar=grammar;self.token_lengths=common.get('token_lengths');self.N=len(grammar['prefixes']);self.E=len(grammar['entry_node'])
        self.ent_node=T(grammar['entry_node']);self.ent_tok=T(grammar['entry_token']);self.ent_child=T(grammar['entry_child']);self.node_pe=T(grammar['parents'])
        self.en_np=np.array(grammar['entry_node']);self.pe_np=np.array(grammar['parents']);self.depth=np.array(grammar['depths']);self.conf=np.array(grammar['conf'])
        # DFS recursion means entries were appended after all siblings of each parent;
        # nodes are also appended in DFS order, so entry blocks remain node-contiguous.
        counts=np.bincount(self.en_np,minlength=self.N);self.start_np=np.r_[0,np.cumsum(counts)];self.start=T(self.start_np)
        assert np.array_equal(self.en_np,np.repeat(np.arange(self.N),counts))
        self.levels=[T(np.flatnonzero(self.depth==d)) for d in range(self.depth.max()+1)]
        self.lev_ent=[T(np.flatnonzero(self.depth[self.en_np]==d)) for d in range(self.depth.max()+1)]
        self.pos=T(np.zeros(self.N,dtype=int))
        for lv in self.levels:self.pos[lv]=torch.arange(len(lv),device=device)
        self.by_k={int(k):T(np.flatnonzero(counts==k)) for k in np.unique(counts)}
        self.leaf_np=np.array(grammar['entry_child'])<0;self.leaf=T(self.leaf_np,torch.bool);self.leaf_idx=np.flatnonzero(self.leaf_np)
        self.X={'instance':cache['X'].to(device)};self.V0=cache['V0'].to(device)
        self.xx={'instance':self.X['instance'].double().square().sum(-1)}
        self.g=common['norm'].to(device);self.eps=common['eps'];self.s=common['scaling']
        # Only grammar token embedding rows are needed after conditioning.
        self.W=common['W'].to(device);self.token_index={t:i for i,t in enumerate(common['tokens'])}
        self.embidx=T([self.token_index[t] for t in grammar['entry_token']])

    @torch.no_grad()
    def forward(self,A,B):
        X=self.X['instance'];XA=X@A.T;v=self.V0+self.s*XA@B.T
        rms=(v.square().mean(-1,keepdim=True)+self.eps).sqrt();h=self.g*v/rms
        w=self.W[self.embidx];logits=(h[self.ent_node]*w).sum(-1)
        P=torch.empty(self.E,dtype=torch.float64,device=v.device)
        for k,nodes in self.by_k.items():
            ix=self.start[nodes,None]+torch.arange(k,device=v.device)
            P[ix]=logits[ix].double().softmax(-1)
        mean=torch.zeros_like(h).index_add_(0,self.ent_node,P.float()[:,None]*w)
        a=w-mean[self.ent_node];ga=self.g*a;ve=v[self.ent_node];rr=rms[self.ent_node]
        u=ga/rr-ve*(ve*ga).sum(-1,keepdim=True)/(v.shape[1]*rr**3)
        lpn=torch.zeros(self.N,dtype=torch.float64,device=v.device)
        for lv in self.levels[1:]:
            pe=self.node_pe[lv];lpn[lv]=lpn[self.ent_node[pe]]+P[pe].log()
        return dict(job='instance',P=P,pn=lpn.exp(),u=u,BU=u@B,XA=XA,useA=bool(B.abs().sum()>0))

    def rewards_to_entries(self,table):
        f=torch.zeros(self.E,dtype=torch.float64,device=self.device)
        f[self.leaf]=torch.as_tensor(table,dtype=torch.float64,device=self.device)[torch.as_tensor(self.conf[self.leaf_np],device=self.device)]
        return f
