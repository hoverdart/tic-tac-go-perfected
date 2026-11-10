import json
from collections import OrderedDict
from solver.push_solver.benchmark_v4 import corpus,run
from solver.push_solver.v4 import Context
original_children=Context.children
original_put=Context.put
original_checkpoint=Context.checkpoint
rows={r['id']:r for r in corpus()}
def children(self,state,macros=False):
 if self.ranker is not None:
  return (yield from original_children(self,state,macros))
 cache=getattr(self,'_children_complete',None)
 if cache is None: cache=self._children_complete=OrderedDict()
 key=state,macros,self.macro_mode
 if key in cache:
  self.metrics['complete_children_hits']+=1
  return list(cache[key])
 result=yield from original_children(self,state,macros)
 self.put(cache,key,tuple(result),16384)
 return result
Context.children=children
for variant,batch in [('child_cache',False),('child_cache_batch16',True)]:
 def checkpoint(self):
  self._ticks=getattr(self,'_ticks',0)+1
  if self._ticks%16==0:
   self.check()
   yield None
 Context.checkpoint=checkpoint if batch else original_checkpoint
 for key in ['20260301','20260314','20260913']:
  r=run(rows[key],schedule='coverage',quality_seconds=30)
  print(json.dumps({'experiment':variant,**r}),flush=True)
