"""Container-only single-request server. No evaluator, dataset or host socket mounts."""
import base64,json,os,shutil,signal,subprocess,sys,tempfile,selectors,time,resource
from pathlib import Path
LIMIT=2_000_000
CHILD='''import contextlib,importlib.util,json,sys
from pathlib import Path
root=Path(sys.argv[1])
with (root/'instance.json').open() as f: instance=json.load(f)
with open('/dev/null','w') as sink,contextlib.redirect_stdout(sink):
 spec=importlib.util.spec_from_file_location('candidate',root/'candidate.py')
 mod=importlib.util.module_from_spec(spec);sys.modules['candidate']=mod
 spec.loader.exec_module(mod)
 result=mod.solve_instance(instance)
print(json.dumps(result,allow_nan=False))
'''
def clear_tmp():
    # The entire writable filesystem is reset between requests, not only cwd.
    for p in Path('/tmp').iterdir():
        if p.is_dir() and not p.is_symlink():shutil.rmtree(p)
        else:p.unlink()
def execute(req):
    try:
        with tempfile.TemporaryDirectory(prefix='request-',dir='/tmp') as tmp:
            root=Path(tmp);(root/'candidate.py').write_text(req['code']);(root/'instance.json').write_text(json.dumps(req['instance']))
            for f in root.iterdir():f.chmod(0o444)
            def child_limits():resource.setrlimit(resource.RLIMIT_FSIZE,(2048,2048))
            child=subprocess.Popen([sys.executable,'-I','-B','-c',CHILD,tmp],cwd=tmp,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True,preexec_fn=child_limits)
            streams=selectors.DefaultSelector();streams.register(child.stdout,selectors.EVENT_READ,'out');streams.register(child.stderr,selectors.EVENT_READ,'err')
            captured={'out':bytearray(),'err':bytearray()};deadline=time.monotonic()+req['timeout'];timed_out=False;overflow=False
            try:
                while streams.get_map():
                    remaining=deadline-time.monotonic()
                    if remaining<=0:timed_out=True;break
                    ready=streams.select(remaining)
                    if not ready:timed_out=True;break
                    for key,_ in ready:
                        chunk=os.read(key.fileobj.fileno(),65536)
                        if not chunk:streams.unregister(key.fileobj);continue
                        captured[key.data].extend(chunk)
                        if len(captured[key.data])>LIMIT:overflow=True;break
                    if overflow:break
                if timed_out or overflow:
                    os.killpg(child.pid,signal.SIGKILL);rc=child.wait(timeout=5)
                else:
                    try:rc=child.wait(timeout=max(.01,deadline-time.monotonic()))
                    except subprocess.TimeoutExpired:
                        timed_out=True;os.killpg(child.pid,signal.SIGKILL);rc=child.wait(timeout=5)
            finally:
                try:os.killpg(child.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                streams.close();child.stdout.close();child.stderr.close()
            raw=bytes(captured['out']);error=bytes(captured['err'][:4096]).decode(errors='replace')
            if timed_out:return dict(status='candidate_timeout',error='')
            if overflow:return dict(status='candidate_output_limit',error='')
            if rc:return dict(status='candidate_process_error',error=error)
            return dict(status='ok',output=base64.b64encode(raw).decode())
    finally:clear_tmp()
for line in sys.stdin.buffer:
    req=json.loads(line)
    try:response=execute(req)
    except Exception as exc:response=dict(status='infrastructure_error',error=str(exc))
    response['id']=req['id']
    sys.stdout.write(json.dumps(response)+'\n');sys.stdout.flush()
