import json
from solver.push_solver.benchmark_v4 import corpus,run
from solver.push_solver.v4 import Context
original_checkpoint=Context.checkpoint
original_put=Context.put
rows={r['id']:r for r in corpus()}
for variant in ['batch16','score32768','both']:
 def checkpoint(self):
  self._ticks=getattr(self,'_ticks',0)+1
  if self._ticks%16==0:
   self.check()
   yield None
 def put(self,cache,key,value,limit=8192):
  if cache is self.score_cache: limit=32768
  return original_put(cache,key,value,limit)
 Context.checkpoint=checkpoint if variant in ('batch16','both') else original_checkpoint
 Context.put=put if variant in ('score32768','both') else staticmethod(original_put)
 for key in ['20260301','20260314','20260913']:
  result=run(rows[key],schedule='coverage',quality_seconds=30)
  print(json.dumps({'experiment':variant,**result}),flush=True)
