"""Generic launcher: python launch.py OUT CONFIG_DIR METHODS SEEDS STOP_AFTER SLOTS
methods comma-separated; seeds as a-b inclusive. One run per (config, seed, method), GPUs round-robin."""
import json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];DATA=os.environ.get('JOBSHOP_DATA',str(ROOT/'data'))
PY=sys.executable
out,cfgdir,methods,seeds,stop,slots=Path(sys.argv[1]),Path(sys.argv[2]),sys.argv[3].split(','),sys.argv[4],sys.argv[5],sys.argv[6]
a,b=map(int,seeds.split('-'));seeds=list(range(a,b+1))
out.mkdir(parents=True,exist_ok=True);sl=out/'solver-slots';sl.mkdir(exist_ok=True);(sl/'limit.json').write_text(json.dumps({'slots':int(slots)}))
cfgs=sorted(cfgdir.glob('*.json'));jobs=[(c,s,m) for c in cfgs for s in seeds for m in methods]
os.chdir(ROOT);children={}
for i,(cfg,seed,m) in enumerate(jobs):
    tag=f'{cfg.stem}/seed-{seed}/{m}';dest=out/tag;dest.mkdir(parents=True,exist_ok=True)
    if (dest/'complete.json').exists():continue
    env={**os.environ,'CUDA_VISIBLE_DEVICES':str(i%8),'JOBSHOP_EVALUATOR':'persistent-v1','PYTHONPATH':str(ROOT),
         'HF_HOME':os.environ.get('HF_HOME',os.path.join(DATA,'hf')),'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
    cmd=[PY,'-u','-m','jobshop_rl.kings_val128_train','--config',str(cfg),'--cache',os.environ.get('JOBSHOP_CACHE',os.path.join(DATA,'jobshop-v2')),'--out',str(dest),
         '--method',m,'--seed',str(seed),'--slot-dir',str(sl),'--docker-slots',slots,'--stop-after',stop]
    f=(dest/'trainer.log').open('ab',buffering=0)
    children[tag]=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,env=env);time.sleep(.3)
print(json.dumps({'launched':len(children)}),flush=True)
while True:
    codes={k:v.poll() for k,v in children.items()};(out/'status.json').write_text(json.dumps(codes,indent=2))
    if all(v is not None for v in codes.values()):break
    time.sleep(30)
print(json.dumps({'done':len(codes),'failed':[k for k,v in codes.items() if v]}),flush=True)
