"""Persistent restricted Docker; fresh Python process + clean tmp per candidate.
Only used for the trusted canonical action constructor, not arbitrary LLM code.
"""
import atexit,base64,json,os,select,shutil,subprocess,tempfile,time,uuid
from pathlib import Path
from .evaluator import Evaluator,InfrastructureError
from .data import ROOT

class PersistentEvaluator(Evaluator):
    def __init__(self,image,timeout=10,score_mode='synthetic_lb'):
        super().__init__(image,timeout,score_mode)
        self.name='jobshop-persistent-'+uuid.uuid4().hex
        self.folder=Path(tempfile.mkdtemp(prefix='jobshop-service-',dir='/tmp'));self.folder.chmod(0o755)
        shutil.copy2(ROOT/'jobshop_rl/persistent_server.py',self.folder/'server.py');(self.folder/'server.py').chmod(0o444)
        self.stderr=tempfile.TemporaryFile();self.buffer=b'';self.request_id=0;self.closed=False
        cmd=['docker','run','--rm','-i','--init','--name',self.name,'--label','jobshop.backend=persistent-v1','--network=none','--read-only','--cap-drop=ALL','--security-opt=no-new-privileges','--pids-limit=32','--memory=512m','--memory-swap=512m','--cpus=1','--user=65534:65534','--ulimit','fsize=2097152:2097152','--tmpfs','/tmp:rw,noexec,nosuid,size=16m','-v',f'{self.folder}:/service:ro',self.image_id,'python','-I','-B','/service/server.py']
        self.process=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.stderr,bufsize=0)
        atexit.register(self.close)
    def close(self):
        if self.closed:return
        self.closed=True
        if self.process.stdin:self.process.stdin.close()
        try:self.process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            subprocess.run(['docker','rm','-f',self.name],capture_output=True,timeout=20)
            try:self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:self.process.kill();self.process.wait(timeout=5)
        self.process.stdout.close();self.stderr.close();shutil.rmtree(self.folder,ignore_errors=True)
    def read_response(self,deadline):
        while b'\n' not in self.buffer:
            remaining=deadline-time.monotonic()
            if remaining<=0:raise InfrastructureError('Persistent worker response timeout')
            if not select.select([self.process.stdout],[],[],remaining)[0]:raise InfrastructureError('Persistent worker response timeout')
            chunk=os.read(self.process.stdout.fileno(),65536)
            if not chunk:
                self.stderr.seek(0);error=self.stderr.read(4096).decode(errors='replace');raise InfrastructureError('Persistent worker exited: '+error)
            self.buffer+=chunk
            if len(self.buffer)>3_000_000:raise InfrastructureError('Persistent protocol output limit')
        line,self.buffer=self.buffer.split(b'\n',1)
        return json.loads(line)
    def evaluate(self,code,instance):
        if self.closed:raise InfrastructureError('Persistent evaluator is closed')
        start=time.monotonic();self.candidate_evaluations+=1;self.instance_executions+=1;self.request_id+=1
        public={k:instance[k] for k in ('duration_matrix','machines_matrix')};public.update(name='instance',metadata={})
        req=dict(id=self.request_id,code=code,instance=public,timeout=self.timeout+3)
        try:
            raw=(json.dumps(req)+'\n').encode();view=memoryview(raw)
            while view:
                n=self.process.stdin.write(view)
                if not n:raise InfrastructureError('Persistent worker stdin closed')
                view=view[n:]
            r=self.read_response(time.monotonic()+self.timeout+33)
            if r.get('id')!=self.request_id:raise InfrastructureError('Persistent response ID mismatch')
            if r['status']=='infrastructure_error':raise InfrastructureError(r.get('error','Persistent server failure'))
            if r['status']!='ok':return self.invalid(r['status'],start,r.get('error',''))
            raw=base64.b64decode(r['output'],validate=True)
            if len(raw)>2_000_000:return self.invalid('candidate_output_limit',start)
            try:
                result=json.loads(raw);validation=self.validator._validate_baseline_schedule(instance,result)
            except (ValueError,TypeError,KeyError,OverflowError) as e:return self.invalid('candidate_invalid',start,str(e))
            return self.metrics(instance,validation.actual_makespan,time.monotonic()-start)
        except (OSError,BrokenPipeError,ValueError) as e:raise InfrastructureError(str(e)) from e
