import ast,json,textwrap
from collections import OrderedDict
from pathlib import Path
from solver.push_solver.benchmark_v4 import corpus,run
from solver.push_solver.v4 import Context,Session
import solver.push_solver.v4 as module
source=Path('/app/solver/push_solver/v4.py').read_text()
tree=ast.parse(source)
def method(classname,name):
 cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==classname)
 node=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name==name)
 return textwrap.dedent('\n'.join(source.splitlines()[node.lineno-1:node.end_lineno]))
original_put=Context.put
original_children=Context.children
ns=dict(module.__dict__)
exec(method('Context','successors').replace('8192','65536'),ns)
Context.successors=ns['successors']
s=method('Session','coverage_beam_search').replace('self.nodes - before >= 32 or spent >= 0.025','self.nodes - before >= 32').replace('spent = 0.0','').replace('started = time.perf_counter()','').replace('spent += time.perf_counter() - started','')
exec(s,ns)
Session.coverage_beam_search=ns['coverage_beam_search']
def put(self,cache,key,value,limit=8192):
 return original_put(cache,key,value,max(limit,65536) if limit>=8192 else limit)
Context.put=put
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
rows={r['id']:r for r in corpus()}
def checkpoint(self):
 self._ticks=getattr(self,'_ticks',0)+1
 if self._ticks%16==0:
  self.check()
  yield None
Context.checkpoint=checkpoint
for key in ['20260301','20260314','20260913']:
 r=run(rows[key],schedule='coverage',quality_seconds=30)
 print(json.dumps({'experiment':'combined_cython_node32_cache65536_children',**r}),flush=True)
