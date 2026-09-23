"""Cache exact frozen-backbone features for constrained last-layer LoRA policy."""
import argparse,copy,json,math,time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from .data import ROOT,experiment_data,full_instance_context,write_json,digest,grouped_batches
from .row_tree import make_grammar

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--out',required=True);p.add_argument('--limit',type=int,default=0);p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);a=p.parse_args()
    cfg=json.loads(Path(a.config).read_text());out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32=False
    data,split,_=experiment_data(cfg);pc=cfg['planner']
    tok=AutoTokenizer.from_pretrained(pc['model'],revision=pc['revision'],local_files_only=True)
    grammar=make_grammar(tok);write_json(out/'grammar.json',grammar)
    identity=digest(dict(config=cfg,grammar=grammar,dataset=split['dataset_sha256']))
    meta=out/'identity.json'
    if meta.exists() and json.loads(meta.read_text())['hash']!=identity:raise RuntimeError('Cache identity mismatch')
    write_json(meta,dict(hash=identity,config=cfg,dataset_sha256=split['dataset_sha256'],nodes=len(grammar['prefixes']),actions=len(grammar['plans'])))
    names=list(dict.fromkeys(n for b in grouped_batches(split['train'],cfg['epochs'],cfg['seed'],cfg['group_size'],cfg['instances_per_batch']) for n in b))+split['val'] # no test features or labels accessed
    assert 0<=a.shard<a.shards
    names=names[a.shard::a.shards]
    if a.limit:names=names[:a.limit]
    missing=[n for n in names if not (out/f'{n}.pt').exists()]
    print(json.dumps(dict(stage='prepare',nodes=len(grammar['prefixes']),instances=len(names),missing=len(missing))),flush=True)
    if not missing:return
    model=AutoModelForCausalLM.from_pretrained(pc['model'],revision=pc['revision'],torch_dtype=torch.float32,attn_implementation='eager',local_files_only=True).cuda().eval().requires_grad_(False)
    lm=model.model;last=lm.layers[-1];saved={}
    last.mlp.down_proj.register_forward_pre_hook(lambda m,args:saved.update(X=args[0]))
    lm.norm.register_forward_pre_hook(lambda m,args:saved.update(V0=args[0]))
    torch.manual_seed(pc['init_seed']);A=torch.empty(pc['lora_rank'],last.mlp.down_proj.in_features);torch.nn.init.kaiming_uniform_(A,a=math.sqrt(5))
    tokens=sorted(set(grammar['entry_token']))
    common=dict(token_lengths=[len(tok.encode(s,add_special_tokens=False))+1 for s in grammar['texts']],W=model.lm_head.weight[tokens].cpu(),tokens=tokens,norm=lm.norm.weight.cpu(),eps=lm.norm.variance_epsilon,scaling=16/pc['lora_rank'],A0=A,B0=torch.zeros(model.config.hidden_size,pc['lora_rank']),identity=identity)
    torch.save(common,out/'common.pt')
    paths=[tuple(x) for x in grammar['prefixes']];ancestors={p[:i] for p in paths for i in range(len(p))};maximal=[p for p in paths if p not in ancestors];where={}
    for i,path in enumerate(maximal):
        for j in range(len(path)+1):where.setdefault(path[:j],(i,j))
    owner=[where[p] for p in paths];width=max(map(len,maximal));batch=cfg.get('cache_microbatch',4)
    for ni,n in enumerate(missing):
        started=time.time();prompt=tok.apply_chat_template([{'role':'system','content':pc['system_prompt']},{'role':'user','content':full_instance_context(data[n])}],add_generation_prompt=True)
        if len(prompt)>pc['max_prompt_tokens']:raise ValueError('Prompt overflow')
        X=torch.empty(len(paths),model.config.intermediate_size);V0=torch.empty(len(paths),model.config.hidden_size)
        checks=[]
        with torch.no_grad():
            prefix=lm(input_ids=torch.tensor([prompt[:-1]],device='cuda'),use_cache=True).past_key_values
            for offset in range(0,len(maximal),batch):
                chunk=maximal[offset:offset+batch];kv=copy.deepcopy(prefix);kv.batch_repeat_interleave(len(chunk))
                ids=torch.tensor([prompt[-1:]+list(path)+[tok.pad_token_id]*(width-len(path)) for path in chunk],device='cuda')
                lm(input_ids=ids,past_key_values=kv,use_cache=True)
                ix=[i for i,(s,k) in enumerate(owner) if offset<=s<offset+len(chunk)]
                ss=[owner[i][0]-offset for i in ix];kk=[owner[i][1] for i in ix]
                X[ix]=saved['X'][ss,kk].cpu();V0[ix]=saved['V0'][ss,kk].cpu()
                del kv
            # Compare cached features against independent, unpadded full forwards.
            for i in sorted(set((0,len(paths)//2,len(paths)-1))):
                lm(input_ids=torch.tensor([prompt+list(paths[i])],device='cuda'),use_cache=False)
                error=max(float((saved['X'][0,-1].cpu()-X[i]).abs().max()),float((saved['V0'][0,-1].cpu()-V0[i]).abs().max()))
                checks.append(error)
            if max(checks)>.003:raise RuntimeError(f'Feature cache mismatch {checks}')
        tmp=out/f'{n}.tmp';torch.save(dict(X=X,V0=V0,identity=identity,name=n,feature_check_max=max(checks)),tmp);tmp.replace(out/f'{n}.pt')
        print(json.dumps(dict(stage='cached',name=n,completed=ni+1,total=len(missing),seconds=time.time()-started,feature_error=max(checks))),flush=True)
if __name__=='__main__':main()
