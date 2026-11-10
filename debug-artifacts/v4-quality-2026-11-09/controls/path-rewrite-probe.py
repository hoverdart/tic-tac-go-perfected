import time,json,sys,itertools
from pathlib import Path
from solver.push_solver.v4 import Context,cells
from solver.push_solver.models import Push
from solver.push_solver import verify_solution
from solver.push_solver.benchmark_v4 import read_rows,corpus
from apps.api.puzzle_catalog import CatalogEntry
from datetime import date

class Rewriter:
 def __init__(self,ctx,seconds):
  self.ctx=ctx;self.deadline=time.perf_counter()+seconds;self.cache={};self.evaluations=0
  self.initial=tuple(cells(ctx.os))+tuple(cells(ctx.xs));self.types=tuple('O' if ctx.os&(1<<p) else 'X' for p in self.initial)
 def check(self):
  self.ctx.check()
  if time.perf_counter()>=self.deadline:raise TimeoutError()
 def walks(self,player,occupied):
  key=player,occupied
  if key in self.cache:return self.cache[key]
  paths={player:''};queue=[player]
  for current in queue:
   for move,nxt in self.ctx.adj[current]:
    if occupied&(1<<nxt) or nxt in paths:continue
    paths[nxt]=paths[current]+move;queue.append(nxt)
  region=sum(1<<p for p in paths)
  if len(self.cache)>=1024:self.cache.pop(next(iter(self.cache)))
  self.cache[key]=(paths,region)
  return paths,region
 def label(self,pushes):
  positions={p:i for i,p in enumerate(self.initial)};ops=[]
  for p in pushes:
   label=positions.pop(p.cell);dest=self.ctx.push_lookup[p.cell,p.move][1];positions[dest]=label;ops.append((label,p.move))
  return tuple(ops)
 def replay(self,ops,bound):
  self.check();self.evaluations+=1
  ctx=self.ctx;positions=list(self.initial);os,xs,player=ctx.os,ctx.xs,ctx.player
  parts=[];cost=0;pushes=[]
  for i in range(len(ops)+1):
   self.check();paths,region=self.walks(player,os|xs)
   if ctx.goal((os,xs,region))[0]:
    options=[paths[t] if t is not None else '' for t in ctx.goal_cache[os] if t is None or t in paths]
    walk=min(options,key=lambda p:(len(p),p))
    if cost+len(walk)<bound:return ''.join(parts)+walk,tuple(pushes),ops[:i]
   if i==len(ops):return None
   label,move=ops[i];cell=positions[label]
   edge=ctx.push_lookup.get((cell,move))
   if edge is None:return None
   stand,dest=edge
   if dest in () or (os|xs)&(1<<dest) or stand not in paths:return None
   walk=paths[stand];cost+=len(walk)+1
   if cost>=bound:return None
   piece=self.types[label]
   if piece=='O':os=(os^(1<<cell))|(1<<dest)
   else:
    xs=(xs^(1<<cell))|(1<<dest)
    if ctx.x_loss(xs,dest):return None
   positions[label]=dest;player=cell;parts.extend((walk,move));pushes.append(Push(piece,cell,move))
 def optimize(self,moves,pushes):
  original=len(moves);ops=self.label(pushes);best=moves;bestpush=pushes
  def accept(candidate):
   nonlocal ops,best,bestpush
   r=self.replay(tuple(candidate),len(best))
   if r is not None and verify_solution(self.ctx.board,r[0]).ok:
    best,bestpush,ops=r;return True
   return False
  try:
   accept(ops)
   for iteration in range(20):
    changed=False
    for label in sorted({op[0] for op in ops}):
     if accept([op for op in ops if op[0]!=label]):changed=True;break
    if changed:continue
    blocks=[tuple(g) for _,g in itertools.groupby(ops,key=lambda o:o[0])]
    for i,block in enumerate(blocks):
     candidate=sum(blocks[:i]+blocks[i+1:],())
     if accept(candidate):changed=True;break
    if changed:continue
    for i in range(len(ops)):
     if accept(ops[:i]+ops[i+1:]):changed=True;break
    if changed:continue
    for i in range(len(blocks)):
     for j in range(i+1,min(len(blocks),i+9)):
      candidate=sum(blocks[:i]+[blocks[j]]+blocks[i:j]+blocks[j+1:],())
      if accept(candidate):changed=True;break
     if changed:break
    if not changed:break
  except TimeoutError:pass
  return best,bestpush,self.evaluations

if __name__=='__main__':
 cat={r['id']:r for r in corpus()};runs=read_rows('debug-artifacts/v4-regressions-2026-11-09/corpus-coverage.jsonl');seconds=float(sys.argv[1]);totals=[0,0];improved=0
 with Path(sys.argv[2]).open('w') as f:
  for r in runs:
   row=cat[r['id']];board=CatalogEntry(date.fromisoformat(row['date']),row['name'],row['width'],row['height'],row['puzzle']).board();ctx=Context(board,None,None)
   pushes=tuple(Push(**p) for p in r['result']['pushes']);old=r['result']['moves'];start=time.perf_counter();rewriter=Rewriter(ctx,seconds);new,newpush,n=rewriter.optimize(old,pushes)
   assert verify_solution(board,new).ok
   totals[0]+=len(old);totals[1]+=len(new);improved+=len(new)<len(old)
   f.write(json.dumps({'id':r['id'],'old':len(old),'new':len(new),'moves':new,'seconds':time.perf_counter()-start,'evaluations':n})+'\n');f.flush()
 print(json.dumps({'totals':totals,'improved':improved,'mean':totals[1]/397}),flush=True)
