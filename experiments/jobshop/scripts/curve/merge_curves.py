"""Merge expected_curve.py shard files INTO an existing curves json (adding or replacing runs, keeping the rest), after
checking every added run was scored on the same update grid as the runs already there.
usage: merge_curves.py RESULT_JSON SHARD_PREFIX      (reads SHARD_PREFIX*.json)"""
import json,glob,sys
result,prefix=sys.argv[1],sys.argv[2]
have=json.load(open(result)) if glob.glob(result) else {}
new={}
for f in sorted(glob.glob(prefix+'*.json')):new.update(json.load(open(f)))
if not new:sys.exit('no shard files at '+prefix)
grid=next(iter(have.values()))['updates'] if have else None
bad=[k for k,v in new.items() if grid is not None and v['updates']!=grid]
if bad:sys.exit(f'update grid mismatch for {bad[:3]}: {new[bad[0]]["updates"][:5]}... vs {grid[:5]}...')
have.update(new);json.dump(have,open(result,'w'))
methods=sorted({k.split('/')[-1] for k in new});print(f'merged {len(new)} runs ({", ".join(methods)}) -> {result}: {len(have)} runs total')
