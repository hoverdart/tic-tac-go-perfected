import inspect,json,textwrap
from solver.push_solver.benchmark_v4 import corpus,run
from solver.push_solver.v4 import Context
original_put=Context.put
original_successors=Context.successors
original_checkpoint=Context.checkpoint
rows={r['id']:r for r in corpus()}
for variant,capacity,batch in [('capacity65536',65536,False),('capacity262144',262144,False),('capacity65536_batch16',65536,True)]:
 source=textwrap.dedent(inspect.getsource(original_successors)).replace('8192',str(capacity))
 ns={}
 exec(source,ns)
 Context.successors=ns['successors']
 def put(self,cache,key,value,limit=8192):
  if limit>=8192: limit=max(limit,capacity)
  return original_put(cache,key,value,limit)
 def checkpoint(self):
  self._ticks=getattr(self,'_ticks',0)+1
  if self._ticks%16==0:
   self.check()
   yield None
 Context.put=put
 Context.checkpoint=checkpoint if batch else original_checkpoint
 for key in ['20260301','20260314','20260913']:
  r=run(rows[key],schedule='coverage',quality_seconds=30)
  print(json.dumps({'experiment':variant,**r}),flush=True)
