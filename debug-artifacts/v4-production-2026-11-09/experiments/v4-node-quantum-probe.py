import inspect,json,textwrap
from solver.push_solver.benchmark_v4 import corpus,run
from solver.push_solver.v4 import Session
original=Session.coverage_beam_search
rows={r['id']:r for r in corpus()}
for quantum in [32,8,128]:
 src=textwrap.dedent(inspect.getsource(original)).replace('self.nodes - before >= 32 or spent >= 0.025',f'self.nodes - before >= {quantum}')
 src=src.replace('spent = 0.0','').replace('started = time.perf_counter()','').replace('spent += time.perf_counter() - started','')
 ns=dict(original.__globals__)
 exec(src,ns)
 Session.coverage_beam_search=ns['coverage_beam_search']
 for key in ['20260301','20260314','20260913']:
  r=run(rows[key],schedule='coverage',quality_seconds=30)
  print(json.dumps({'experiment':'node_quantum_'+str(quantum),**r}),flush=True)
