from collections import OrderedDict
import numpy as np
from solver.push_solver.v4 import Context
original=Context.ranked_rows

def ranked(self,state,*,full=False):
 os,xs,_=state
 geometry=yield from self.geometry(os)
 if full or self.size>64 or not 64<=len(geometry)<=4096 or self.route_limit==1:
  return (yield from original(self,state,full=full))
 cache=getattr(self,'_vector_routes',None)
 if cache is None:
  cache=self._vector_routes=OrderedDict()
 prepared=cache.get(os)
 if prepared is None:
  canonical=np.asarray([(r[0],1<<r[1],r[6][0]+r[7][0]+bool(r[6][1]&(1<<r[4]))+bool(r[7][1]&(1<<r[2])),r[6][1],r[7][1]) for r in geometry],dtype=np.uint64).T
  lookup={}
  alternatives=[[],[]]
  for row in geometry:
   _,_,oa,a,ob,b,_,_=row
   for side,(source,target,other) in enumerate(((oa,a,ob),(ob,b,oa))):
    key=source,target,other
    if key not in lookup:
     routes=yield from self.routes(source,target)
     lookup[key]=routes+(routes[-1],)*(self.route_limit-len(routes))
    alternatives[side].append(lookup[key])
  route_masks=np.asarray([[[r[1] for r in opts] for opts in side] for side in alternatives],dtype=np.uint64)
  bases=np.asarray([[[r[0]+bool(r[1]&(1<<(row[4] if side==0 else row[2]))) for r in opts] for row,opts in zip(geometry,alternatives[side])] for side in (0,1)],dtype=np.uint64)
  prepared=canonical,alternatives,route_masks,bases
  self.put(cache,os,prepared,128)
 line_masks,targets,base,ra,rb=prepared[0]
 x=np.uint64(xs)
 scores=base+np.bitwise_count(x&ra)+np.bitwise_count(x&rb)+.5*np.bitwise_count(x&line_masks)+2*((x&targets)!=0)
 indices=np.argsort(scores,kind='stable')[:24]
 costs=prepared[3][:,indices,:]+np.bitwise_count(x&prepared[2][:,indices,:])
 choices=np.argmin(costs,axis=2)
 needed=((ra[indices]|rb[indices])&x)!=0
 choices[:,~needed]=0
 scores=np.take_along_axis(costs,choices[:,:,None],axis=2)[:,:,0].sum(axis=0)+.5*np.bitwise_count(x&line_masks[indices])+2*((x&targets[indices])!=0)
 ia,ib=choices.tolist()
 refined=[(s,i,prepared[1][0][i][a],prepared[1][1][i][b]) for s,i,a,b in zip(scores.tolist(),indices.tolist(),ia,ib)]
 refined.sort()
 return geometry,tuple(refined)

Context.ranked_rows=ranked
