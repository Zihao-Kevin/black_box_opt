"""Predeclared, independently seeded synthetic JobShop dataset; no score filtering."""
import argparse
import hashlib
import math
import random
from functools import reduce
from pathlib import Path
from .data import digest, write_json

SIZES = ((10,5),(10,10),(15,5),(15,10),(15,15),(20,5),(20,10),(30,10))
FAMILIES = ('uniform','bottleneck','heterogeneous','correlated_routes')
COUNTS = {'train':4,'val':2,'test':4}

def lower_bound(x):
    ds,ms=x['duration_matrix'],x['machines_matrix']
    loads=[0]*(max(map(max,ms))+1)
    for dr,mr in zip(ds,ms):
        for d,m in zip(dr,mr): loads[m]+=d
    return max(max(map(sum,ds)),max(loads))

def invariant_fingerprint(x):
    # Necessary invariant under arbitrary job/machine renaming and global scale.
    # Equal signatures flag possible equivalence, not a proof of isomorphism.
    ds,ms=x['duration_matrix'],x['machines_matrix']
    gcd=reduce(math.gcd,(d for row in ds for d in row))
    jobs=sorted(tuple(d//gcd for d in row) for row in ds)
    machines=sorted(tuple(sorted((o,d//gcd) for dr,mr in zip(ds,ms)
                    for o,(d,m) in enumerate(zip(dr,mr)) if m==machine))
                    for machine in range(len(ms[0])))
    return digest([jobs,machines])

def validate_instance(x):
    ds,ms=x['duration_matrix'],x['machines_matrix']; n=len(ds); m=len(ms[0])
    if not n or len(ms)!=n: raise ValueError('invalid jobs')
    for dr,mr in zip(ds,ms):
        if len(dr)!=m or sorted(mr)!=list(range(m)): raise ValueError('invalid routing')
        if any(type(d) is not int or d<=0 for d in dr): raise ValueError('invalid duration')
    return lower_bound(x)

def generate(seed=20260916):
    data={}; split={s:[] for s in COUNTS}; strata={}
    for n,m in SIZES:
        for family in FAMILIES:
            stratum=f'{n}x{m}/{family}'; strata[stratum]={}
            for partition,count in COUNTS.items():
                strata[stratum][partition]=[]
                for k in range(count):
                    name=f'syn1_{n}x{m}_{family}_{partition}_{k:02d}'
                    instance_seed=int.from_bytes(hashlib.sha256(f'{seed}/{name}'.encode()).digest()[:8],'big')
                    rng=random.Random(instance_seed); template=rng.sample(range(m),m); bottleneck=rng.randrange(m)
                    ds=[];ms=[]
                    for j in range(n):
                        route=rng.sample(range(m),m)
                        if family=='correlated_routes':
                            route=template.copy()
                            for _ in range(max(1,m//5)):
                                a,b=rng.sample(range(m),2);route[a],route[b]=route[b],route[a]
                        factor=rng.choice((1,2,4)) if family=='heterogeneous' else 1
                        durations=[]
                        for machine in route:
                            d=rng.randint(1,99)
                            if family=='bottleneck' and machine==bottleneck:d*=4
                            if family=='heterogeneous':d=rng.randint(1,25)*factor*(4 if rng.random()<.15 else 1)
                            durations.append(d)
                        ds.append(durations);ms.append(route)
                    x=dict(name=name,duration_matrix=ds,machines_matrix=ms,metadata=dict(
                        source='synthetic-v1',family=family,seed=instance_seed,
                        generator_parameters=dict(bottleneck_machine=bottleneck if family=='bottleneck' else None,
                        route_template=template if family=='correlated_routes' else None)))
                    x['metadata']['lower_bound']=validate_instance(x)
                    data[name]=x;split[partition].append(name);strata[stratum][partition].append(name)
    signatures=[invariant_fingerprint(x) for x in data.values()]
    if len(set(signatures))!=len(data):raise ValueError('possible equivalent instances; investigate without resampling')
    split.update(seed=seed,dataset_sha256=digest(data),strata=strata)
    manifest=dict(version=1,seed=seed,counts={s:len(split[s]) for s in COUNTS},sizes=SIZES,families=FAMILIES,
        dataset_sha256=digest(data),split_sha256=digest(split),generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        reward_mode='synthetic_lb',reward='max(max_job_work,max_machine_work)/validated_makespan',
        protocol='Independent hash-derived seeds; no rejection based on solver performance; 4/2/4 per size-family stratum.',
        distributions={'uniform':'independent uniform machine permutations; durations uniform integers 1..99',
        'bottleneck':'uniform routes and durations; one independently chosen machine durations multiplied by 4',
        'heterogeneous':'uniform routes; per-job factor uniformly 1/2/4; per-operation uniform 1..25 times factor, with independent 15% 4x outliers',
        'correlated_routes':'shared uniform machine permutation, max(1,m//5) random pair swaps per job; durations uniform 1..99'},
        equivalence_check='All normalized duration/routing signatures differ; excludes job/machine relabeling and global scaling equivalences, without claiming general graph canonicalization.')
    return data,split,manifest

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--seed',type=int,default=20260916);a=p.parse_args()
    out=Path(a.out)
    if out.exists():raise FileExistsError('Refusing to replace frozen dataset')
    data,split,manifest=generate(a.seed)
    for name,value in [('instances',data),('split',split),('manifest',manifest)]:write_json(out/f'{name}.json',value)
    print(manifest['counts'],manifest['dataset_sha256'])
if __name__=='__main__':main()
