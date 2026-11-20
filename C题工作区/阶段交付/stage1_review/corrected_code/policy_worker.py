"""Stateless policy worker. No hidden identifiers, seeds or source files in protocol."""
import sys,json,math,pathlib,os
SCRIPT=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(SCRIPT))
from common import np,prefix_features
model_path=pathlib.Path(sys.argv[1]).resolve();MODEL=json.loads(model_path.read_text(encoding='utf8'))
# Deny private reads/enumeration and process/network escape in this worker.
allowed=[pathlib.Path(sys.base_prefix).resolve(),SCRIPT,(SCRIPT.parents[1]/'运行依赖314').resolve(),model_path.parent]
def audit(event,args):
 if event=='open':
  p=args[0]
  if isinstance(p,(str,bytes,os.PathLike)):
   path=pathlib.Path(os.fsdecode(p)).resolve()
   if not any(path==r or r in path.parents for r in allowed):raise PermissionError('policy filesystem boundary')
   mode=args[1]
   if isinstance(mode,str) and any(c in mode for c in 'wax+'):raise PermissionError('policy read only')
 if event in ['os.listdir','os.scandir']:
  p=pathlib.Path(args[0] or '.').resolve()
  if not any(p==r or r in p.parents for r in allowed):raise PermissionError('policy enumeration boundary')
 if event.startswith(('subprocess.','socket.')) or event in ['os.system','os.exec','os.spawn']:raise PermissionError('policy process/network boundary')
sys.addaudithook(audit)
def estimate(job,q,alpha,beta,kind):
 family=job['family'];length=str(float(job['prefix_length']))
 if kind=='family':d=MODEL['families'][family][str(q)]
 else:
  m=MODEL['ridge'][length];x=np.r_[job['features'],[float(family==v) for v in ['P','A','L']]];z=np.r_[1,(x-np.array(m['mu']))/np.array(m['sd'])];d=max(.001,float(np.exp(np.clip(z@m['coef'],-10,10)))+m['residual_quantiles'][str(q)])
 return max(.2,alpha*d+beta)
def act(o):
 if o.get('probe'):
  try:open(o['probe'],'rb');return {'denied':False}
  except PermissionError:return {'denied':True}
 now=o['time'];H=o['H'];B=o['B'];method=o['method'];q=o['quantile'];jobs=o['jobs'];done={j['job_id'] for j in jobs if j['completed']};kind='family' if method in ['family_edf','family_density'] else 'ridge';candidates=[]
 for j in jobs:
  if j['attempted'] or j['release_s']>now+1e-12 or any(p not in done for p in j['predecessors']):continue
  b=estimate(j,q,o['alpha'],o['beta'],kind);D=math.dist(o['zone_xy'],j['zone_xy']);start=now+D+.5;fee=2+.1*b+.2*D
  if start+b>H+1e-12 or o['cost']+fee>B+1e-12:continue
  key=(j['due_s'],j['job_id']) if method!='family_density' else (-j['weight']/(D+.5+b),j['due_s'],j['job_id'])
  candidates.append((key,j,b))
 if candidates:
  _,j,b=min(candidates,key=lambda x:x[0])
  if method=='ridge_review_edf' and j['reviews']==0 and o['reviews']<o['K'] and now+1<=H and o['cost']+1<=B:
   return {'type':'review','job_id':j['job_id']}
  return {'type':'execute','job_id':j['job_id'],'reserve':b}
 future=[j['release_s'] for j in jobs if not j['attempted'] and j['release_s']>now+1e-12 and j['release_s']<=H]
 if future:return {'type':'wait','duration':min(future)-now}
 return {'type':'stop'}
for line in sys.stdin:
 try:
  o=json.loads(line);print(json.dumps(act(o),allow_nan=False),flush=True)
 except Exception as e:print(json.dumps({'error':type(e).__name__+': '+str(e)}),flush=True)
