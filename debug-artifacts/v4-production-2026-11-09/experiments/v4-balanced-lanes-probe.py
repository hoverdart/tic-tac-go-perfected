import ast,json,textwrap
from pathlib import Path
from solver.push_solver.benchmark_v4 import corpus,run
from solver.push_solver.v4 import Session,GoalSession,Found
import solver.push_solver.v4 as module
original_search=GoalSession.search
def search(self):
 for item in original_search(self):
  if isinstance(item,Found):
   m=self.parent.context.metrics
   m['winner_width']=self.widths[0][0]
   m['winner_goal_objective']=int(self.objective=='goal')
   m['winner_goal_index']=self.index
  yield item
GoalSession.search=search
source=Path('/app/solver/push_solver/v4.py').read_text(); tree=ast.parse(source)
cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Session')
n=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='coverage_beam_search')
original=textwrap.dedent('\n'.join(source.splitlines()[n.lineno-1:n.end_lineno]))
rows={r['id']:r for r in corpus()}
for variant,cycle in [('balanced11','(0, 5, 1, 2, 3, 4, 5, 1, 2, 3, 5)')]:
 ns=dict(module.__dict__);exec(original.replace('for index in range(len(lanes)):', 'for index in '+cycle+':'),ns);Session.coverage_beam_search=ns['coverage_beam_search']
 for key in ['20261024','20251212','20260705','20260701','20250928','20251207','20260301','20260913']:
  r=run(rows[key],schedule='coverage',routes=2,quality_seconds=30)
  print(json.dumps({'experiment':'lane_allocation_'+variant,**r}),flush=True)
