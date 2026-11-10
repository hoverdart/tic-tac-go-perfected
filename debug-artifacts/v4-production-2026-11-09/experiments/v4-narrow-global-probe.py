import ast,json,textwrap
from pathlib import Path
from solver.push_solver.benchmark_v4 import corpus,run
from solver.push_solver.v4 import Session
import solver.push_solver.v4 as module
source=Path('/app/solver/push_solver/v4.py').read_text(); tree=ast.parse(source)
cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Session')
node=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='serial_beam_search')
method=textwrap.dedent('\n'.join(source.splitlines()[node.lineno-1:node.end_lineno]))
rows={r['id']:r for r in corpus()}
for width in [16,32,24]:
 s=method.replace('widths = ((48, 60), (96, 90), (160, 120), (256, 160))',f'widths = (({width}, 60), (48, 90), (96, 120), (160, 160)) if objective == "global" and not wide else ((48, 60), (96, 90), (160, 120), (256, 160))')
 ns=dict(module.__dict__); exec(s,ns); Session.serial_beam_search=ns['serial_beam_search']
 for key in ['20260301','20260314','20260913','20260802']:
  r=run(rows[key],schedule='coverage',quality_seconds=30)
  print(json.dumps({'experiment':'narrow_global_'+str(width),**r}),flush=True)
