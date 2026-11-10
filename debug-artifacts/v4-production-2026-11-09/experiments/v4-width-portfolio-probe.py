import ast,json,textwrap
from pathlib import Path
from solver.push_solver.benchmark_v4 import corpus,run
from solver.push_solver.v4 import Session
import solver.push_solver.v4 as module
source=Path('/app/solver/push_solver/v4.py').read_text(); tree=ast.parse(source)
cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Session')
def method(name):
 n=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name==name)
 return textwrap.dedent('\n'.join(source.splitlines()[n.lineno-1:n.end_lineno]))
s=method('serial_beam_search').replace('wide=False):','wide=False, narrow_width=None):').replace('if wide:\n','if narrow_width is not None:\n        widths = ((narrow_width, 60), (48, 90), (96, 120), (160, 160))\n    if wide:\n')
ns=dict(module.__dict__);exec(s,ns);Session.serial_beam_search=ns['serial_beam_search']
s=method('coverage_beam_search').replace('self.serial_beam_search(),','self.serial_beam_search(narrow_width=16),\n        self.serial_beam_search(narrow_width=32),').replace('self.serial_beam_search(wide=True),','self.serial_beam_search(wide=True),\n        self.serial_beam_search(narrow_width=24),').replace('for index in (0, 1, 0, 1, 2):','for index in range(len(lanes)):')
exec(s,ns);Session.coverage_beam_search=ns['coverage_beam_search']
rows={r['id']:r for r in corpus()}
for key in ['20260301','20260314','20260913','20260802','20251212','20251228','20260911','20261004','20261017','20261018','20261025']:
 r=run(rows[key],schedule='coverage',routes=2,quality_seconds=30)
 print(json.dumps({'experiment':'width_portfolio_16_32_goal48_wide96_24_routes2',**r}),flush=True)
