import time,json,heapq,itertools,sys
from pathlib import Path
from datetime import date
from solver.push_solver.v4 import Context,cells
from solver.push_solver.path_optimizer import PathRewriter,RewriteExpired
from solver.push_solver.benchmark_v4 import corpus,read_rows
from solver.push_solver.models import Push
from solver.push_solver import verify_solution
from apps.api.puzzle_catalog import CatalogEntry

def vicinity(ctx,moves,pushes,seconds):
 deadline=time.perf_counter()+seconds;rw=PathRewriter(ctx,deadline);best_moves,best_pushes=moves,pushes;nodes=0
 try:
  os,xs=ctx.os,ctx.xs;anchors={(os,xs)}
  for p in pushes:
   d=ctx.push_lookup[p.cell,p.move][1]
   if p.piece=='O':os=(os^(1<<p.cell))|(1<<d)
   else:xs=(xs^(1<<p.cell))|(1<<d)
   anchors.add((os,xs))
  allowed=set(anchors)
  for os,xs in anchors:
   rw.check();occupied=os|xs
   for piece,mask in [('O',os),('X',xs)]:
    for c in cells(mask):
     for move,stand,dest in ctx.transitions[c]:
      if occupied&(1<<dest):continue
      changed=(mask^(1<<c))|(1<<dest)
      allowed.add((changed,xs) if piece=='O' else (os,changed))
  start=(ctx.os,ctx.xs,ctx.player);dist={start:0};parents={};count=itertools.count();q=[(0,next(count),start)]
  while q:
   rw.check();g,_,state=heapq.heappop(q)
   if g!=dist[state] or g>=len(best_moves):continue
   os,xs,player=state;paths,region=rw.walks(player,os|xs);nodes+=1
   if ctx.goal((os,xs,region))[0]:
    walk=min([paths[t] if t is not None else '' for t in ctx.goal_cache[os] if t is None or t in paths],key=lambda w:(len(w),w))
    if g+len(walk)<len(best_moves):
     chunks=[walk];ps=[];key=state
     while key!=start:
      previous,p,segment=parents[key];chunks.append(segment+p.move);ps.append(p);key=previous
     candidate=''.join(reversed(chunks));candidate_pushes=tuple(reversed(ps))
     assert verify_solution(ctx.board,candidate).ok
     best_moves,best_pushes=candidate,candidate_pushes
   occupied=os|xs
   for piece,mask in [('O',os),('X',xs)]:
    for c in cells(mask):
     for move,stand,dest in ctx.transitions[c]:
      if occupied&(1<<dest) or stand not in paths:continue
      changed=(mask^(1<<c))|(1<<dest)
      a,b=(changed,xs) if piece=='O' else (os,changed)
      if (a,b) not in allowed or (piece=='X' and ctx.x_loss(b,dest)):continue
      nxt=a,b,c;ng=g+len(paths[stand])+1
      if ng>=len(best_moves) or ng>=dist.get(nxt,1e9):continue
      dist[nxt]=ng;parents[nxt]=(state,Push(piece,c,move),paths[stand]);heapq.heappush(q,(ng,next(count),nxt))
 except RewriteExpired:pass
 return best_moves,best_pushes,nodes

if __name__=='__main__':
 cat={r['id']:r for r in corpus()};raw=read_rows('debug-artifacts/v4-regressions-2026-11-09/corpus-coverage.jsonl');seconds=float(sys.argv[1]);total=[0,0];wins=0
 with Path(sys.argv[2]).open('w') as f:
  for r in raw:
   row=cat[r['id']];board=CatalogEntry(date.fromisoformat(row['date']),row['name'],row['width'],row['height'],row['puzzle']).board();ctx=Context(board,None,None);pushes=tuple(Push(**p) for p in r['result']['pushes']);rewritten=PathRewriter(ctx,time.perf_counter()+.1).optimize(r['result']['moves'],pushes);start=time.perf_counter();moves,ps,nodes=vicinity(ctx,rewritten.moves,rewritten.pushes,seconds)
   assert verify_solution(board,moves).ok
   total[0]+=len(rewritten.moves);total[1]+=len(moves);wins+=len(moves)<len(rewritten.moves)
   f.write(json.dumps({'id':r['id'],'before':len(rewritten.moves),'after':len(moves),'moves':moves,'nodes':nodes,'seconds':time.perf_counter()-start})+'\n');f.flush()
 print(json.dumps({'totals':total,'improved':wins,'mean':total[1]/397}),flush=True)
