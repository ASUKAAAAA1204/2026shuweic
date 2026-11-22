import math

def action(o,medians,ordering,reservation):
 done={j['job_id'] for j in o['jobs'] if j['completed']};candidates=[]
 for j in o['jobs']:
  if j['attempted'] or j['release_s']>o['time']+1e-12 or not set(j['predecessors'])<=done:continue
  estimate=max(.2,o['alpha']*medians[j['family']]+o['beta']);b=.2 if reservation=='minimum' else estimate;D=math.dist(o['zone_xy'],j['zone_xy']);start=o['time']+D+.5;fee=2+.1*b+.2*D
  if start+b>o['H']+1e-12 or o['cost']+fee>o['B']+1e-12:continue
  key=(j['due_s'],j['job_id']) if ordering=='edf' else (-j['weight']/(D+.5+max(estimate,b)),j['due_s'],j['job_id']);candidates.append((key,j['job_id'],b))
 if candidates:
  _,job,b=min(candidates);return dict(type='execute',job_id=job,reserve=b)
 future=[j['release_s'] for j in o['jobs'] if not j['attempted'] and o['time']+1e-12<j['release_s']<=o['H']]
 return dict(type='wait',duration=min(future)-o['time']) if future else dict(type='stop')
