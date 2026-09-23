import hashlib, importlib.util, json, random, sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FRONTIER = ROOT / 'vendor/frontier'
JOBSHOP = FRONTIER / 'benchmarks/JobShop'
VERSIONS = {'frontier': 'e3fa29c193356af2ce1ec8b3d23ab1a2e2410071', 'black_box_opt': '1289992a063410ff6b2ce45da94dfdb82a4c956a'}

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

def database(path=None):
    return json.loads(((ROOT / path) if path else (JOBSHOP / 'data/benchmark_instances.json')).read_text())

def experiment_data(cfg):
    data=database(cfg.get('dataset'))
    split=json.loads((ROOT/cfg['split']).read_text())
    if split.get('dataset_sha256') != digest(data):raise ValueError('Dataset/split hash mismatch')
    if cfg.get('dataset') and cfg.get('score_mode')!='synthetic_lb':raise ValueError('Custom dataset requires explicit scoring')
    if cfg.get('baseline'):
        baseline=json.loads((ROOT/cfg['baseline']).read_text())
    elif cfg.get('input_mode')=='full_instance':
        baseline={name:{} for name in data}
    else:raise ValueError('Baseline required for summary context')
    return data,split,baseline

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()

def make_split(seed=20260914):
    data = database(); groups = defaultdict(list)
    for name, x in data.items():
        if name.startswith('la'):
            groups[(len(x['duration_matrix']), max(max(r) for r in x['machines_matrix'])+1)].append(name)
    rng = random.Random(seed)
    result = dict(seed=seed, versions=VERSIONS, dataset_sha256=digest(data), train=[], val=[], test=[], strata={})
    for shape, names in sorted(groups.items()):
        names.sort(); rng.shuffle(names)
        assert len(names) == 5
        result['train'] += names[:3]; result['val'] += names[3:4]; result['test'] += names[4:]
        result['strata'][f'{shape[0]}x{shape[1]}'] = names
    assert [len(result[s]) for s in ('train','val','test')] == [24,8,8]
    return result

def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n'); temp.replace(path)

def context(instance, baseline_metrics, timeout, source_path=None):
    # Full task and initial code; only scale and actual baseline feedback vary with x.
    task = (JOBSHOP/'la/Task.md').read_text()
    return task + '\nConstraints:\n' + (JOBSHOP/'la/frontier_eval/constraints.txt').read_text() + '\n' + json.dumps({
        'jobs':len(instance['duration_matrix']), 'machines':max(max(r) for r in instance['machines_matrix'])+1,
        'baseline_feedback':{k:baseline_metrics[k] for k in ('combined_score','valid','actual_makespan')},
        'execution_budget':{'wall_seconds':timeout,'cpu_cores':1,'memory_mb':512},
        'interface':'solve_instance(instance) -> dict(name, makespan, machine_schedules). Each machine_schedules entry is a list of dict(job_id, operation_index, start_time, end_time, duration). Preserve every operation, duration, machine assignment, precedence, and non-overlap. No files, network, external solver, test data, or benchmark lookup; use the supplied matrices.',
    }) + '\nFull initial baseline source:\n' + ((ROOT/source_path) if source_path else (JOBSHOP/'la/baseline/init.py')).read_text()


def policy_context(instance, baseline_metrics, timeout, source_path=None):
    ds=instance['duration_matrix']; ms=instance['machines_matrix']
    nm=max(max(r) for r in ms)+1
    loads=[sum(d for j,row in enumerate(ds) for o,d in enumerate(row) if ms[j][o]==m) for m in range(nm)]
    return 'Choose one plan from the system action dictionary for this job-shop instance. Minimize makespan. '+json.dumps(dict(jobs=len(ds),machines=nm,job_work=list(map(sum,ds)),machine_work=loads,baseline_makespan=baseline_metrics['actual_makespan'],neighbor_budget=64,timeout_seconds=timeout))


def full_instance_context(instance, baseline_metrics=None, timeout=10, source_path=None):
    """Lossless operation information, with no benchmark identity or score leakage."""
    ds=instance['duration_matrix'];ms=instance['machines_matrix']
    return ('Choose the best plan for THIS instance to minimize makespan. '
            'Each jobs row is one job in precedence order; each pair is [machine_id, processing_time]. '
            'Job and machine indices start at zero. The constructor will solve these exact operations. '
            'Use the instance structure when choosing the algorithm.\n'+json.dumps(dict(
                num_jobs=len(ds),num_machines=max(max(r) for r in ms)+1,
                jobs=[[[m,d] for m,d in zip(machines,durations)] for machines,durations in zip(ms,ds)],
                neighbor_budget=64,timeout_seconds=timeout),separators=(',',':')))


def context_function(cfg):
    if cfg.get('input_mode')=='full_instance':return full_instance_context
    return policy_context if cfg.get('planner',{}).get('output_mode')=='policy_plan' else context


def balanced_batches(names,epochs,seed):
    """Two shuffled passes/epoch; each instance supplies 2 plans/pass."""
    if not names or len(names)%2 or len(set(names))!=len(names):raise ValueError('Need even, unique instance set')
    rng=random.Random(seed);batches=[]
    for epoch in range(epochs):
        for sweep in range(2):
            order=list(names);rng.shuffle(order)
            for i in range(0,len(order),2):batches.append([order[i]]*2+[order[i+1]]*2)
    return batches


def grouped_batches(names, epochs, seed, group_size=8, instances_per_batch=2):
    """One independent shuffled pass per epoch, G adjacent outputs per instance."""
    if group_size < 2 or instances_per_batch < 1 or epochs < 1:
        raise ValueError('Invalid grouped schedule dimensions')
    if not names or len(set(names)) != len(names) or len(names) % instances_per_batch:
        raise ValueError('Need unique instances divisible by instances_per_batch')
    rng=random.Random(seed); batches=[]
    for _ in range(epochs):
        order=list(names);rng.shuffle(order)
        for i in range(0,len(order),instances_per_batch):
            batches.append([n for n in order[i:i+instances_per_batch] for _ in range(group_size)])
    return batches
