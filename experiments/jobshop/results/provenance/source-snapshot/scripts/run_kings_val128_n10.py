"""Separate 70-run campaign: sharded feature preparation, training, then frozen test."""
import os,json,time,subprocess,fcntl,socket,shutil,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];os.chdir(ROOT)
OUT=ROOT/'runs/kings-beta010-val128-n10'
LOCAL=Path('/tmp/szhang3007-jobshop-kings/val128-n10')
CACHE=LOCAL/'cache';SLOTS=LOCAL/'slots'
METHODS=['qcv','row_delta','grpo','relax','rloo','otb','reinforce']
CONFIG='configs/synthetic-val128-v1.json'
def write(name,value):
 p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(p)
def launch(cmd,log,gpu):
 env={**os.environ,'CUDA_VISIBLE_DEVICES':gpu,'JOBSHOP_EVALUATOR':'persistent-v1','PYTHONPATH':'.','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','TOKENIZERS_PARALLELISM':'false','HF_HOME':str(ROOT/'cache/huggingface'),'XDG_CACHE_HOME':str(ROOT/'cache')}
 with Path(log).open('ab',buffering=0) as f:
  return subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,env=env,start_new_session=True)
def main():
 assert socket.gethostname().split('.')[0]=='kings'
 OUT.mkdir(exist_ok=True);LOCAL.mkdir(parents=True,exist_ok=True);SLOTS.mkdir(exist_ok=True)
 lock=(OUT/'manager.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not (OUT/'launches.json').exists(),'Inspect existing campaign before restarting'
 assert shutil.disk_usage(OUT).free>6*1024**3
 assert shutil.disk_usage(LOCAL).free>70*1024**3
 rows=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
 gpus=[];locks=[]
 for row in rows.splitlines():
  gpu,mem,util=[v.strip() for v in row.split(',')]
  if int(mem)>=100 or int(util):continue
  f=(ROOT/'runs'/f'gpu-{gpu}.lock').open('a')
  try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:f.close();continue
  gpus.append(gpu);locks.append(f)
  if len(gpus)==7:break
 assert len(gpus)==7,f'Need 7 idle unlocked GPUs, found {len(gpus)}'
 (SLOTS/'limit.json').write_text(json.dumps({'slots':70}))
 write('protocol.json',dict(seeds=list(range(10,20)),methods=METHODS,epochs=32,entropy_beta=.1,train_instances=128,val_instances=128,test_instances=128,validation_B=4,selection='Maximum validation mean reward, earliest epoch on ties; initial eligible; test never selects',checkpoint_retention='initial and every epoch actor, plus latest full resume state',training_evaluations=70*32768,validation_evaluations=70*34*512,gpus=gpus,config=CONFIG,cache=str(CACHE),created_at=time.time()))
 children=[]
 for i,gpu in enumerate(gpus):
  p=launch([str(ROOT/'.venv-train/bin/python'),'-u','-m','jobshop_rl.row_prepare_sharded','--config',CONFIG,'--out',str(LOCAL/f'shard-{i}'),'--shard',str(i),'--shards','7'],OUT/f'prepare-{i}.log',gpu);children.append(p)
 write('prepare-launches.json',[dict(pid=p.pid,gpu=gpus[i]) for i,p in enumerate(children)])
 while any(p.poll() is None for p in children):
  bad=[p.pid for p in children if p.poll() not in (None,0)]
  if bad:raise RuntimeError(f'Feature preparation failed: {bad}')
  write('status.json',dict(phase='preparing',cached=sum(len(list((LOCAL/f'shard-{i}').glob('syn*.pt'))) for i in range(7)),target=256,updated_at=time.time()))
  time.sleep(15)
 assert all(p.returncode==0 for p in children)
 CACHE.mkdir(exist_ok=True)
 for i in range(7):
  shard=LOCAL/f'shard-{i}'
  for name in ('identity.json','grammar.json'):
   assert json.loads((shard/name).read_text())==json.loads((LOCAL/'shard-0'/name).read_text())
  for src in shard.glob('syn*.pt'):
   dst=CACHE/src.name
   if not dst.exists():dst.symlink_to(src)
 for name in ('identity.json','grammar.json','common.pt'):
  dst=CACHE/name
  if not dst.exists():dst.symlink_to(LOCAL/'shard-0'/name)
 assert len(list(CACHE.glob('syn*.pt')))==256
 registry={};children={};monitors={}
 for seed in range(10,20):
  for i,method in enumerate(METHODS):
   out=OUT/f'seed-{seed}'/method;out.mkdir(parents=True,exist_ok=True)
   cmd=[str(ROOT/'.venv-train/bin/python'),'-u','-m','jobshop_rl.kings_val128_train','--config',CONFIG,'--cache',str(CACHE),'--out',str(out),'--method',method,'--seed',str(seed),'--slot-dir',str(SLOTS),'--docker-slots','70']
   key=f'{seed}/{method}';p=launch(cmd,out/'trainer.log',gpus[i]);children[key]=p
   registry[key]=dict(pid=p.pid,gpu=gpus[i],seed=seed,method=method);write('launches.json',registry)
   time.sleep(.5)
   if any(p.poll() not in (None,0) for p in children.values()):raise RuntimeError('Trainer failed during scale-up')
 for key,r in registry.items():
  out=OUT/f"seed-{r['seed']}"/r['method']
  p=launch([str(ROOT/'.venv-monitor/bin/python'),'-u','scripts/wandb_kings_val128_monitor.py','--seed',str(r['seed']),'--beta','0.10','--method',r['method'],'--entity','szhang3007-georgia-institute-of-technology','--project','jobshop-planner-rl'],out/'wandb.log',r['gpu']);monitors[key]=p.pid;time.sleep(.2)
 write('monitors.json',monitors)
 while True:
  active=[];complete=[];failed=[]
  for key,p in children.items():
   seed,method=key.split('/');out=OUT/f'seed-{seed}'/method
   if (out/'complete.json').exists():complete.append(key)
   elif p.poll() is None:active.append(key)
   else:failed.append(dict(run=key,returncode=p.returncode))
  write('status.json',dict(phase='training' if active else 'training_finished',active=active,complete=complete,failed=failed,updated_at=time.time(),home_free_gib=shutil.disk_usage(OUT).free/1024**3))
  if not active:break
  time.sleep(30)
 if failed:raise RuntimeError('Some training runs failed; test not launched')
 for f in locks:fcntl.flock(f,fcntl.LOCK_UN);f.close()
 p=launch([str(ROOT/'.venv-train/bin/python'),'-u','scripts/test_kings_val128_selected.py'],OUT/'test.log',gpus[0]);write('test-launch.json',dict(pid=p.pid,gpu=gpus[0],time=time.time()))
if __name__=='__main__':
 try:main()
 except Exception as e:write('manager-error.json',dict(error=str(e),time=time.time()));raise
