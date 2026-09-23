# Runs ONLY inside restricted candidate container. No evaluator/data are mounted.
import contextlib, importlib.util, json, sys
with open('/input/instance.json') as f: instance = json.load(f)
with open('/dev/null','w') as sink, contextlib.redirect_stdout(sink):
    spec = importlib.util.spec_from_file_location('candidate','/input/candidate.py')
    mod = importlib.util.module_from_spec(spec); sys.modules['candidate'] = mod
    spec.loader.exec_module(mod)
    result = mod.solve_instance(instance)
print(json.dumps(result, allow_nan=False))
