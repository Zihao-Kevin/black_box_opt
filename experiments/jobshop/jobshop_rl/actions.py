"""Version 2: strict finite plans and deterministic feasible sequence search."""
import json
import re

DISPATCH = {
    'est_spt': 'earliest start, then shortest processing time',
    'spt': 'shortest processing time',
    'lpt': 'longest processing time',
    'mwr': 'most remaining job work',
    'remaining_work': 'least remaining job work',
    'remaining_ops': 'fewest remaining job operations',
    'machine_ready': 'earliest machine availability',
    'job_ready': 'earliest job availability',
    'machine_load': 'largest remaining machine workload',
    'least_machine_load': 'smallest remaining machine workload',
    'shortest_next': 'shortest next operation duration (zero at job end)',
    'fifo': 'lowest next operation index',
    'ratio': 'smallest (earliest start + 1)/(remaining job work + 1)',
    'completion': 'earliest operation completion',
    'lookahead': 'earliest projected completion through this and next 2 job operations using current machine availability',
}
TIES = {
    'job_id': 'lowest job index', 'duration': 'shortest duration',
    'remaining_ops': 'fewest remaining operations', 'remaining_work': 'least remaining job work',
    'machine_id': 'lowest machine index', 'machine_time': 'earliest machine availability',
    'machine_load': 'least remaining machine load', 'est': 'earliest start',
    'precedence_gap': 'least wait after job predecessor',
    'random_tiebreak': 'fixed reproducible integer hash of job and operation',
}
POST = ('none', 'local_sort', 'local_insert', 'local_swap')
FIELDS = {'dispatch','tie_break','postprocess','search_radius'}
PLAN_SYSTEM = '''Choose a job-shop scheduling algorithm to minimize makespan. Return ONLY a short JSON object with exactly these fields:
{"dispatch":"est_spt","tie_break":"job_id","postprocess":"none","search_radius":0}
Dispatch rules:\n''' + '\n'.join(k+': '+v for k,v in DISPATCH.items()) + '\nTie breakers (used after dispatch score):\n' + '\n'.join(k+': '+v for k,v in TIES.items()) + '''
Final ties use job index. A deterministic constructor schedules every job operation in precedence order.
postprocess: none, local_sort, local_insert, local_swap.
none requires search_radius=0. Other modes require search_radius 2, 4 or 8.
Search edits the existing job sequence, then fully decodes a feasible schedule; only strictly lower makespan is accepted.
local_insert moves one job occurrence by up to radius positions; local_swap exchanges two within radius; local_sort sorts a window of radius+1 occurrences by their operation durations.
All searches examine at most 64 neighbors, one deterministic pass. Radius is distance/window size, not extra evaluations. No extra keys, explanations, Python, or Markdown.'''


def validate_plan(plan):
    return (isinstance(plan, dict) and set(plan)==FIELDS
            and isinstance(plan['dispatch'],str) and plan['dispatch'] in DISPATCH
            and isinstance(plan['tie_break'],str) and plan['tie_break'] in TIES
            and isinstance(plan['postprocess'],str) and plan['postprocess'] in POST
            and type(plan['search_radius']) is int
            and ((plan['postprocess']=='none' and plan['search_radius']==0)
                 or (plan['postprocess']!='none' and plan['search_radius'] in (2,4,8))))


def parse_action(text):
    try:
        # Fences are formatting, never repairs to field values.
        match=re.fullmatch(r'\s*```(?:json)?\s*(.*?)\s*```\s*',text,re.S)
        def unique(pairs):
            d={}
            for k,v in pairs:
                if k in d: raise ValueError('duplicate key')
                d[k]=v
            return d
        plan=json.loads(match.group(1) if match else text,object_pairs_hook=unique)
        return plan if validate_plan(plan) else None
    except (ValueError,TypeError):
        return None


def solve_plan(instance, plan):
    """Each occurrence of job j denotes its next operation, preserving precedence."""
    ds=instance['duration_matrix']; ms=instance['machines_matrix']
    nj=len(ds); nm=max((m for row in ms for m in row),default=-1)+1
    total=sum(map(len,ds))
    def decode(seq):
        jt=[0]*nj; mt=[0]*nm; ix=[0]*nj; schedules=[[] for _ in range(nm)]
        for j in seq:
            o=ix[j]; m=ms[j][o]; d=ds[j][o]; st=max(jt[j],mt[m]); end=st+d
            schedules[m].append(dict(job_id=j,operation_index=o,start_time=st,end_time=end,duration=d))
            ix[j]+=1; jt[j]=mt[m]=end
        assert ix==list(map(len,ds))
        return dict(name=instance['name'],makespan=max(jt,default=0),machine_schedules=schedules)
    jt=[0]*nj; mt=[0]*nm; ix=[0]*nj; rw=list(map(sum,ds)); load=[0]*nm; seq=[]
    for j,row in enumerate(ds):
        for o,d in enumerate(row): load[ms[j][o]]+=d
    for _ in range(total):
        choices=[]
        for j in range(nj):
            o=ix[j]
            if o==len(ds[j]): continue
            m=ms[j][o]; d=ds[j][o]; est=max(jt[j],mt[m]); ops=len(ds[j])-o
            future=est+d
            for k in range(o+1,min(o+3,len(ds[j]))): future=max(future,mt[ms[j][k]])+ds[j][k]
            keys={'est_spt':(est,d),'spt':(d,), 'lpt':(-d,), 'mwr':(-rw[j],),
                  'remaining_work':(rw[j],),'remaining_ops':(ops,),
                  'machine_ready':(mt[m],),'job_ready':(jt[j],),'machine_load':(-load[m],),
                  'least_machine_load':(load[m],),'shortest_next':(ds[j][o+1] if o+1<len(ds[j]) else 0,),
                  'fifo':(o,), 'ratio':((est+1)/(rw[j]+1),),'completion':(est+d,), 'lookahead':(future,)}
            ties={'job_id':j,'duration':d,'remaining_ops':ops,'remaining_work':rw[j], 'machine_id':m,
                  'machine_time':mt[m],'machine_load':load[m],'est':est,'precedence_gap':est-jt[j],
                  'random_tiebreak':((j+1)*1103515245+(o+1)*12345)%2147483647}
            choices.append((keys[plan['dispatch']]+(ties[plan['tie_break']],j),j))
        j=min(choices)[1]; o=ix[j]; m=ms[j][o]; d=ds[j][o]
        jt[j]=mt[m]=max(jt[j],mt[m])+d; ix[j]+=1; rw[j]-=d; load[m]-=d; seq.append(j)
    best=decode(seq); evaluations=accepted=0; radius=plan['search_radius']; kind=plan['postprocess']
    if kind!='none' and total>1:
        # Same 64-proposal budget for all enabled modes. Spread anchors over the sequence.
        for step in range(min(64,total*(total-1))):
            a=(step*37)%total; distance=1+(step//total+step)%radius
            b=min(total-1,a+distance) if step%2==0 else max(0,a-distance)
            candidate=seq.copy()
            if kind=='local_swap': candidate[a],candidate[b]=candidate[b],candidate[a]
            elif kind=='local_insert': candidate.insert(b,candidate.pop(a))
            else:
                lo=max(0,min(a,total-radius-1)); hi=min(total,lo+radius+1)
                seen=[0]*nj; decorated=[]
                for p,j in enumerate(seq):
                    if lo<=p<hi: decorated.append((ds[j][seen[j]],p,j))
                    seen[j]+=1
                candidate[lo:hi]=[j for _,_,j in sorted(decorated)]
            evaluated=decode(candidate); evaluations+=1
            if evaluated['makespan']<best['makespan']:
                best=evaluated;seq=candidate;accepted+=1
    best['search_stats']=dict(evaluations=evaluations,accepted=accepted)
    return best


def build_code(plan):
    import inspect
    if not validate_plan(plan): raise ValueError('Invalid action plan')
    return inspect.getsource(solve_plan)+'\nPLAN = '+repr(plan)+'\ndef solve_instance(instance):\n    return solve_plan(instance, PLAN)\n'
