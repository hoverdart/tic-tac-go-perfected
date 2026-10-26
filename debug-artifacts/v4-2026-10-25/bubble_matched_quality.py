import json,sys,time
from dataclasses import asdict
from datetime import date
from solver.push_solver import solve_v3,solve_v4,verify_solution
from solver.push_solver.benchmark_v4 import corpus,ENGINE_SHA256
from apps.api.puzzle_catalog import CatalogEntry
row=next(r for r in corpus() if r['id']=='20261025')
board=CatalogEntry(date(2026,10,25),row['name'],row['width'],row['height'],row['puzzle']).board()
started=time.perf_counter();result=(solve_v3 if sys.argv[1]=='v3' else solve_v4)(board,timeout_seconds=30,max_nodes=500000,quality_timeout_seconds=10)
print(json.dumps({'id':row['id'],'board_sha256':row['board_sha256'],'version':sys.argv[1],'engine_sha256':ENGINE_SHA256 if sys.argv[1]=='v4' else None,'limits':{'seconds':30,'nodes':500000,'quality_seconds':10},'wall_seconds':time.perf_counter()-started,'verified':verify_solution(board,result.moves).ok,'result':asdict(result)}),flush=True)
