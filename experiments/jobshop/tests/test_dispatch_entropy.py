import torch
from test_row_tree import tiny,paths
from jobshop_rl.dispatch_entropy import dispatch_entropy

def test_exact_marginal_entropy_gradient_matches_autograd():
    W,A,B=tiny();W.grammar['plans']=[dict(dispatch=x) for x in ['a','b','a','b']]
    st=W.forward(A,B);grad,diag=dispatch_entropy(W,st)
    aa=A.clone().requires_grad_();bb=B.clone().requires_grad_()
    v=W.V0+W.s*(W.X['instance']@aa.T)@bb.T
    h=W.g*v/(v.square().mean(-1,keepdim=True)+W.eps).sqrt();logits=h@W.W.T
    probs=logits.softmax(-1)
    leaves=torch.stack([torch.stack([probs[W.en_np[e],int(W.ent_tok[e])] for e in p]).prod() for p in paths()])
    marginal=torch.stack([leaves[0]+leaves[2],leaves[1]+leaves[3]])
    entropy=-(marginal*marginal.log()).sum();expected=torch.autograd.grad(entropy,(aa,bb))
    for actual,want in zip(grad,expected):torch.testing.assert_close(actual,want,atol=2e-6,rtol=2e-5)
    assert abs(diag['dispatch_entropy']-float(entropy.detach()))<1e-6
    assert abs(sum(diag['dispatch_probabilities'].values())-1)<1e-10
