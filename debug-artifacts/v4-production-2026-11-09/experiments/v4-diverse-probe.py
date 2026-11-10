import inspect,json,textwrap
from solver.push_solver.benchmark_v4 import corpus,run
from solver.push_solver.v4 import Session
original=Session.serial_beam_search
rows={r['id']:r for r in corpus()}
for variant in ['diverse_goal','diverse_all','narrow_goal']:
 src=textwrap.dedent(inspect.getsource(original))
 if variant=='diverse_goal': src=src.replace('plans = ctx.initial_anchor_plans','plans = ctx.initial_commitments if objective == "goal" else ctx.initial_anchor_plans')
 if variant=='diverse_all': src=src.replace('plans = ctx.initial_anchor_plans','plans = ctx.initial_commitments')
 if variant=='narrow_goal': src=src.replace('((48, 60), (96, 90), (160, 120), (256, 160))','(((16, 60), (48, 90), (96, 120), (160, 160)) if objective == "goal" else ((48, 60), (96, 90), (160, 120), (256, 160)))')
 ns=dict(original.__globals__)
 exec(src,ns)
 Session.serial_beam_search=ns['serial_beam_search']
 for key in ['20260301','20260314','20260913']:
  r=run(rows[key],schedule='coverage',quality_seconds=30)
  print(json.dumps({'experiment':variant,**r}),flush=True)
