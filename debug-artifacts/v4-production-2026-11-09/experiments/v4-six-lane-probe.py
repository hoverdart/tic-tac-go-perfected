import ast,json,textwrap
from pathlib import Path
from solver.push_solver.benchmark_v4 import corpus,run
from solver.push_solver.v4 import Session
import solver.push_solver.v4 as module
source=Path('/app/solver/push_solver/v4.py').read_text(); tree=ast.parse(source)
cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Session')
n=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='coverage_beam_search')
original=textwrap.dedent('\n'.join(source.splitlines()[n.lineno-1:n.end_lineno]))
rows={r['id']:r for r in corpus()}
for variant,lane in [('global48','self.serial_beam_search()'),('goal16','self.serial_beam_search(objective="goal", narrow_width=16)'),('goal24','self.serial_beam_search(objective="goal", narrow_width=24)'),('goal32','self.serial_beam_search(objective="goal", narrow_width=32)')]:
 s=original.replace('self.serial_beam_search(narrow_width=24),','self.serial_beam_search(narrow_width=24),\n        '+lane+',')
 ns=dict(module.__dict__);exec(s,ns);Session.coverage_beam_search=ns['coverage_beam_search']
 for key in ['20261024','20260301','20260913']:
  r=run(rows[key],schedule='coverage',routes=2,quality_seconds=30)
  print(json.dumps({'experiment':'six_lanes_extra_'+variant,**r}),flush=True)
