import json, math, os, subprocess, tempfile, time, uuid
from pathlib import Path
from .data import JOBSHOP, ROOT, load_module

class InfrastructureError(RuntimeError): pass

class Evaluator:
    def __init__(self, image, timeout=10, score_mode="official"):
        if score_mode not in ("official","synthetic_lb"):raise ValueError("Unknown score mode")
        self.score_mode=score_mode
        self.image=image; self.timeout=timeout
        self.validator=load_module('official_la_validator',JOBSHOP/'la/verification/evaluate.py')
        self.unified=load_module('official_jobshop_metrics',JOBSHOP/'frontier_eval/evaluate_unified.py')
        result=subprocess.run(['docker','image','inspect',image],capture_output=True,text=True)
        if result.returncode: raise InfrastructureError('Candidate image missing; provision explicitly')
        self.image_id=json.loads(result.stdout)[0]['Id']
        self.instance_executions=0; self.candidate_evaluations=0
    def evaluate(self, code, instance):
        start=time.monotonic(); self.candidate_evaluations+=1
        with tempfile.TemporaryDirectory(prefix='candidate-',dir='/tmp') as tmp:
            stage=Path(tmp); os.chmod(stage,0o755)
            (stage/'candidate.py').write_text(code)
            (stage/'worker.py').write_text((ROOT/'jobshop_rl/worker.py').read_text())
            # Runtime can see exactly one input. Metadata/IDs are unnecessary for solving.
            public={k:instance[k] for k in ('duration_matrix','machines_matrix')}; public.update(name='instance',metadata={})
            (stage/'instance.json').write_text(json.dumps(public))
            for p in stage.iterdir():os.chmod(p,0o444)
            name='jobshop-'+uuid.uuid4().hex
            cmd=['docker','run','--rm','--name',name,'--network=none','--read-only','--cap-drop=ALL','--security-opt=no-new-privileges','--pids-limit=32','--memory=512m','--memory-swap=512m','--cpus=1','--user=65534:65534','--ulimit','fsize=2048:2048','--tmpfs','/tmp:rw,noexec,nosuid,size=16m','-v',f'{stage}:/input:ro',self.image_id,'python','-I','-B','/input/worker.py']
            self.instance_executions+=1
            try:
                # Bound logs on disk, not an unbounded PIPE in the parent.
                with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
                    p=subprocess.Popen(cmd,stdout=out,stderr=err)
                    try: rc=p.wait(timeout=self.timeout+3)
                    except subprocess.TimeoutExpired:
                        subprocess.run(['docker','rm','-f',name],capture_output=True,timeout=15)
                        p.wait(timeout=15); return self.invalid('candidate_timeout',start)
                    out.seek(0); raw=out.read(2_000_001); err.seek(0); error=err.read(4096).decode(errors='replace')
                if rc in (125,126,127):raise InfrastructureError(f'Docker runtime failed: {error}')
                if rc: return self.invalid('candidate_process_error',start,error)
                if len(raw)>2_000_000:return self.invalid('candidate_output_limit',start)
                try:
                    result=json.loads(raw); validation=self.validator._validate_baseline_schedule(instance,result)
                except (ValueError,TypeError,KeyError,OverflowError) as exc:
                    return self.invalid('candidate_invalid',start,str(exc))
                return self.metrics(instance,validation.actual_makespan,time.monotonic()-start)
            except OSError as exc:raise InfrastructureError(str(exc)) from exc
    def invalid(self,status,start,error=''):
        return dict(combined_score=0.,valid=False,reward=0.,loss=1.,actual_makespan=None,status=status,error=error[:4096],wall_seconds=time.monotonic()-start)
    def metrics(self,instance,makespan,elapsed):
        if getattr(self,'score_mode','official')=='synthetic_lb':
            from .synthetic import lower_bound
            lb=lower_bound(instance)
            if not math.isfinite(makespan) or makespan<lb:raise InfrastructureError('Validated makespan violates lower bound')
            reward=lb/makespan
            return dict(combined_score=100*reward,valid=True,reward=reward,loss=1-reward,actual_makespan=makespan,status='ok',wall_seconds=elapsed,score_mode='synthetic_lb',lower_bound=lb)
        m=instance['metadata']
        row=self.validator.InstanceResult(instance['name'],m.get('optimum'),m.get('lower_bound'),m.get('upper_bound'),makespan,True,None,elapsed,None,None,'reference intentionally omitted in training')
        metrics=self.unified._compute_metrics([row])
        score=metrics['combined_score']; valid=bool(metrics['valid'])
        if not math.isfinite(score) or not 0<=score<=100:raise InfrastructureError('Unexpected official score range')
        return dict(combined_score=score,valid=valid,reward=score/100 if valid else 0.,loss=1-score/100 if valid else 1.,actual_makespan=makespan,status='ok',wall_seconds=elapsed,official_metrics=metrics)
