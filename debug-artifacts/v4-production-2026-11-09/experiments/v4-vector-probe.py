import json,unittest
import vector_routes
from solver.push_solver.benchmark_v4 import corpus,run
suite=unittest.TestLoader().discover('/release/tests',pattern='test_v4_goal_search.py')
assert unittest.TextTestRunner().run(suite).wasSuccessful()
rows={r['id']:r for r in corpus()}
for key in ['20260301','20260314','20260913']:
 r=run(rows[key],schedule='coverage',quality_seconds=30)
 print(json.dumps({'experiment':'vector_routes',**r}),flush=True)
