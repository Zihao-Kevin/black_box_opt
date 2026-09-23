import json
import numpy as np
import torch
from test_row_tree import tiny,paths
from jobshop_rl.kings_val128_train import reinforce_gradient,actor_initialization,recover_committed_logs

def test_reinforce_enumerated_expectation():
 W,A,B=tiny();st=W.forward(A,B);reward=[.2,.8,.1,.6];probs=(st['pn'][W.ent_node]*st['P'])[W.leaf];avg=[torch.zeros_like(A),torch.zeros_like(B)]
 for i,path in enumerate(paths()):
  g,_=reinforce_gradient(W,st,[(path[0],path)],[reward[i]])
  for a,v in zip(avg,g):a.add_(v*probs[i])
 for a,b in zip(avg,W.true_grad(st,W.rewards_to_entries(reward))):torch.testing.assert_close(a,b,rtol=2e-5,atol=2e-6)

def test_independent_initializations_reproducible_preserve_global_rng():
 c={'A0':torch.randn(8,16),'B0':torch.zeros(7,8)};before=torch.get_rng_state().clone()
 a,b=actor_initialization(c,1);aa,bb=actor_initialization(c,1);a2,_=actor_initialization(c,2)
 assert torch.equal(a,aa) and torch.equal(b,bb) and not torch.equal(a,a2)
 assert torch.equal(torch.get_rng_state(),before)
 assert torch.equal(actor_initialization(c,0)[0],c['A0'])

def test_recovery_checkpoint_before_log_commit(tmp_path):
 previous=[{'update':1,'value':i} for i in range(16)];last=[{'update':2,'value':i} for i in range(16)]
 (tmp_path/'journal.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in previous+last[:4])+'{bad')
 (tmp_path/'metrics.jsonl').write_text(json.dumps({'counts':{'updates':1}})+'\n')
 (tmp_path/'pending.json').write_text('{"update":2}')
 metric={'counts':{'updates':2}};recover_committed_logs(tmp_path,{'updates':2,'candidates':32},last,metric)
 assert len((tmp_path/'journal.jsonl').read_text().splitlines())==32
 assert len((tmp_path/'metrics.jsonl').read_text().splitlines())==2
 assert not (tmp_path/'pending.json').exists()
 recover_committed_logs(tmp_path,{'updates':2,'candidates':32},last,metric)
 assert len((tmp_path/'journal.jsonl').read_text().splitlines())==32
