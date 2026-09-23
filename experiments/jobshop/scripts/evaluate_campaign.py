"""Frozen validation-selected Kings actors, shared streamed test features."""
import concurrent.futures as cf
import fcntl
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import statistics
import sys
import time

ROOT = Path('runs/kings-beta010-val128-n10/test-selected')
ACTORS = {}
POOL = None
EV = None

def solve_group(task):
    global EV
    from jobshop_rl.persistent_evaluator import PersistentEvaluator
    from jobshop_rl.actions import build_code
    cfg, plans, instance, actions = task
    if EV is None:
        EV = PersistentEvaluator(cfg['image'], cfg['timeout'], score_mode='synthetic_lb')
    records = []
    for action in actions:
        result = EV.evaluate(build_code(plans[action]), instance)
        if not result['valid']:
            raise RuntimeError(str(result))
        records.append(dict(action=action, **result))
    return records

def evaluate_instance(name, index, features, grammar, common, cfg, instance):
    import numpy as np
    import torch
    from jobshop_rl.row_tree import RowTree
    from jobshop_rl.data import write_json
    selection = json.loads((ROOT/'selection.json').read_text())['selected']
    W = RowTree(grammar, features, common)
    pending = []
    for key, choice in selection.items():
        target = ROOT/key
        target.mkdir(parents=True, exist_ok=True)
        dest = target/(name+'.json')
        if dest.exists():
            continue
        if key not in ACTORS:
            path = Path(choice['checkpoint'])
            assert hashlib.sha256(path.read_bytes()).hexdigest() == choice['sha256']
            ck = torch.load(path, map_location='cpu', weights_only=False)
            ACTORS[key] = ck['A'].cuda(), ck['B'].cuda()
        with torch.no_grad():
            st = W.forward(*ACTORS[key])
        rng = np.random.default_rng(10000+1000*index)
        actions = [int(W.conf[e]) for e, _ in [W.sample(st, rng) for _ in range(4)]]
        pending.append((dest, POOL.submit(solve_group, (cfg, grammar['plans'], instance, actions))))
    for dest, future in pending:
        write_json(dest, dict(name=name, candidates=future.result()))
    write_json(ROOT/'progress.json', dict(last_instance=name, index=index, instances_completed=index+1, time=time.time()))
    print(json.dumps(dict(stage='test_instance_complete', name=name, index=index)), flush=True)

def finish(names):
    from jobshop_rl.data import write_json
    selection = json.loads((ROOT/'selection.json').read_text())['selected']
    summaries = {}
    for key, choice in selection.items():
        groups = [json.loads((ROOT/key/(n+'.json')).read_text())['candidates'] for n in names]
        assert all(len(g)==4 and all(c['valid'] for c in g) for g in groups)
        summaries[key] = dict(epoch=choice['epoch'], validation=choice['validation'], partition='test', instances=len(names), B=4, candidate_evaluations=4*len(names), mean_reward=statistics.mean(c['reward'] for g in groups for c in g), best_of_B=statistics.mean(max(c['reward'] for c in g) for g in groups), feasible_rate=1.)
    aggregate = {}
    for method in sorted({k.split('/')[-1] for k in summaries}):
        rows = [v for k,v in summaries.items() if k.endswith('/'+method)]
        aggregate[method] = {'seeds':len(rows)}
        for metric in ('mean_reward','best_of_B'):
            values = [r[metric] for r in rows]
            aggregate[method][metric] = dict(mean=statistics.mean(values), seed_std=statistics.stdev(values))
    write_json(ROOT/'summary.json', dict(runs=summaries, aggregate=aggregate))
    write_json(ROOT/'complete.json', dict(time=time.time(), runs=len(summaries)))

def main():
    global POOL, ROOT
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--run-root',required=True);p.add_argument('--cache',required=True);p.add_argument('--stream-cache',required=True);p.add_argument('--config',default='configs/synthetic-val128-v1.json');args=p.parse_args()
    ROOT=Path(args.run_root)/'test-selected';os.environ['JOBSHOP_FEATURE_CACHE']=str(Path(args.cache).resolve())
    import torch
    from jobshop_rl.data import write_json
    from jobshop_rl import test_prepare_stream_val128 as stream
    torch.set_num_threads(1)
    ROOT.mkdir(parents=True, exist_ok=True)
    lock = (ROOT/'test.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    gpu_lock = (ROOT.parent.parent/('gpu-'+os.environ['CUDA_VISIBLE_DEVICES']+'.lock')).open('a')
    fcntl.flock(gpu_lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    selected = {}
    for seed in range(10,20):
        for method in ('qcv','row_delta','grpo','relax','rloo','otb','reinforce'):
            run = ROOT.parent/f'seed-{seed}'/method
            assert (run/'complete.json').exists()
            best = json.loads((run/'best.json').read_text())
            ck = run/'best.pt'
            selected[f'seed-{seed}/{method}'] = dict(checkpoint=str(ck), sha256=hashlib.sha256(ck.read_bytes()).hexdigest(), epoch=best['epoch'], validation=best['validation'])
    selection = dict(rule='Maximum validation mean reward; earliest epoch on ties; frozen before test', selected=selected, sampling_seed='10000 + 1000 * test_instance_index', B=4)
    path = ROOT/'selection.json'
    if path.exists():
        assert json.loads(path.read_text()) == selection
    else:
        write_json(path, selection)
    stream.TEST_ROOT = ROOT
    stream.evaluate_instance = evaluate_instance
    stream.finish = finish
    sys.argv = ['test_prepare_stream','--config',args.config,'--out',args.stream_cache]
    with cf.ProcessPoolExecutor(max_workers=32, mp_context=mp.get_context('spawn')) as pool:
        POOL = pool
        stream.main()

if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        ROOT.mkdir(parents=True, exist_ok=True)
        (ROOT/'failure.json').write_text(json.dumps(dict(error=str(exc), time=time.time())))
        raise
