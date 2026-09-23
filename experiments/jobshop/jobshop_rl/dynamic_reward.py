"""Train-only evolving terminal reward table, updated after actor commits."""
import time
import numpy as np
import torch
from .offline_reward import instance_features,action_features,features,fit_model

class DynamicReward:
    def __init__(self,rows,tables,data,plans,steps=1000):
        self.names=set(tables);self.plans=plans;self.data=data;self.steps=steps
        self.tables={n:np.asarray(t,dtype=np.float64).copy() for n,t in tables.items()}
        self.known={};self.refit_epoch=0
        self.observe(rows);self.initial_count=len(self.known)

    def observe(self,rows):
        # Validate entire block before mutation. No held-out instance can enter.
        pending={}
        for r in rows:
            n,a,reward=r['name'],r['action'],float(r['reward'])
            if n not in self.names or type(a) is not int or not 0<=a<len(self.plans) or not np.isfinite(reward) or not 0<=reward<=1:raise ValueError('Invalid train-only reward label')
            key=(n,a);prior=pending.get(key,self.known.get(key))
            if prior is not None and abs(prior-reward)>1e-8:raise ValueError('Conflicting deterministic reward')
            pending[key]=reward
        for (n,a),reward in pending.items():self.known[n,a]=reward;self.tables[n][a]=reward

    def prediction_errors(self,rows):
        all_errors=[];new_errors=[]
        for r in rows:
            key=(r['name'],r['action']);error=(self.tables[key[0]][key[1]]-r['reward'])**2
            all_errors.append(error)
            if key not in self.known:new_errors.append(error)
        result=dict(reward_prediction_mse=float(np.mean(all_errors)),unseen_sample_count=len(new_errors))
        if new_errors:result['unseen_reward_prediction_mse']=float(np.mean(new_errors))
        return result

    def state_dict(self):return dict(tables=self.tables,known=self.known,refit_epoch=self.refit_epoch,steps=self.steps,initial_count=self.initial_count)

    def load_state_dict(self,s):
        if s['steps']!=self.steps or set(s['tables'])!=self.names:raise ValueError('Dynamic Q state mismatch')
        for v in s['tables'].values():
            v=np.asarray(v)
            if v.shape!=(len(self.plans),) or not np.isfinite(v).all() or (v<0).any() or (v>1).any():raise ValueError('Invalid dynamic reward table')
        self.tables={n:np.asarray(v).copy() for n,v in s['tables'].items()};self.known={};self.refit_epoch=s['refit_epoch'];self.initial_count=s['initial_count']
        self.observe([dict(name=n,action=int(a),reward=r) for (n,a),r in s['known'].items()])

    def refit(self,epoch):
        if epoch!=self.refit_epoch+1:raise ValueError('Nonconsecutive Q refit')
        start=time.time();rows=[dict(name=n,action=a,reward=r) for (n,a),r in sorted(self.known.items())]
        inst={n:instance_features(self.data[n]) for n in sorted(self.names)};acts=[action_features(p) for p in self.plans]
        X=np.stack([features(inst[r['name']],acts[r['action']]) for r in rows]);y=np.array([r['reward'] for r in rows],dtype=np.float32)
        # Refit from scratch at fixed budget, not an adaptive validation-selected fit.
        # Preserve global torch RNG; actor sampling has its own saved NumPy RNG.
        with torch.random.fork_rng(devices=[]):
            model,_,_=fit_model(X,y,rows,self.plans,431+epoch,self.steps)
            tables={}
            with torch.no_grad():
                for n in sorted(self.names):
                    xx=np.stack([features(inst[n],a) for a in acts]);tables[n]=model(torch.from_numpy(xx)).numpy().astype(float)
        if any(not np.isfinite(v).all() for v in tables.values()):raise RuntimeError('Nonfinite fitted table')
        for (n,a),r in self.known.items():tables[n][a]=r
        self.tables=tables;self.refit_epoch=epoch
        return dict(epoch=epoch,unique_labels=len(rows),new_unique_labels=len(rows)-self.initial_count,fit_steps=self.steps,wall_seconds=time.time()-start),dict(state=model.state_dict(),dim=X.shape[1],epoch=epoch,steps=self.steps)
