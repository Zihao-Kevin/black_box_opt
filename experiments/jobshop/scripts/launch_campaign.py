"""Portable explicit-GPU launcher; resumes each run using its committed checkpoint."""
import argparse,fcntl,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser();p.add_argument('--gpus',required=True,help='Comma-separated allocated GPU UUIDs');p.add_argument('--cache',required=True);p.add_argument('--out',required=True);p.add_argument('--slots',type=int,default=70);p.add_argument('--seed-start',type=int,default=10);p.add_argument('--seed-end',type=int,default=20);a=p.parse_args()
 os.chdir(ROOT);out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=True);cache=Path(a.cache).resolve();gpus=a.gpus.split(',');slots=out/'solver-slots';slots.mkdir(exist_ok=True)
 lock=(out/'manager.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 (slots/'limit.json').write_text(json.dumps({'slots':a.slots}));children={}
 methods=['qcv','row_delta','grpo','relax','rloo','otb','reinforce']
 for seed in range(a.seed_start,a.seed_end):
  for i,m in enumerate(methods):
   dest=out/f'seed-{seed}'/m;dest.mkdir(parents=True,exist_ok=True)
   if (dest/'complete.json').exists():continue
   env={**os.environ,'CUDA_VISIBLE_DEVICES':gpus[i%len(gpus)],'JOBSHOP_EVALUATOR':'persistent-v1','PYTHONPATH':str(ROOT),'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
   cmd=[sys.executable,'-u','-m','jobshop_rl.kings_val128_train','--config','configs/synthetic-val128-v1.json','--cache',str(cache),'--out',str(dest),'--method',m,'--seed',str(seed),'--slot-dir',str(slots),'--docker-slots',str(a.slots)]
   with (dest/'trainer.log').open('ab',buffering=0) as f:children[f'{seed}/{m}']=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,env=env)
   time.sleep(.5)
 while True:
  codes={k:v.poll() for k,v in children.items()};tmp=out/'launcher-status.tmp';tmp.write_text(json.dumps(codes,indent=2));tmp.replace(out/'launcher-status.json')
  if all(v is not None for v in codes.values()):break
  time.sleep(20)
 if any(v for v in codes.values()):raise SystemExit('One or more runs failed; inspect logs before resuming')
if __name__=='__main__':main()
