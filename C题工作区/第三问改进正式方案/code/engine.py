"""Existing corrected Q3 engine, isolated from historical filesystem globals."""
import math

class Engine:
 def __init__(self,jobs,hidden,zones,H,B,K,alpha=1,beta=.5):
  self.jobs={j['job_id']:dict(j) for j in jobs};self.hidden=hidden;self.zones=zones;self.H=H;self.B=B;self.K=K;self.alpha=alpha;self.beta=beta;self.time=0.;self.cost=0.;self.zone='Z0';self.reviews=0;self.actions=0;self.terminal=False;self.events=[];self.status={k:dict(attempted=False,completed=False,reviews=0,completion=None) for k in self.jobs}
 def observe(self,method='family_edf',quantile=.8):
  jj=[]
  for key,j in self.jobs.items():
   st=self.status[key];length=[.2,.6,1.][st['reviews']];features=self.hidden[key]['features'][str(length)] if j['release_s']<=self.time+1e-12 else []
   jj.append(dict(job_id=key,family=j['family_id'],release_s=j['release_s'],due_s=j['due_s'],weight=j['priority_weight'],zone_xy=self.zones[j['zone_id']],predecessors=j['predecessors'],attempted=st['attempted'],completed=st['completed'],reviews=st['reviews'],prefix_length=length,features=features))
  return dict(time=self.time,H=self.H,B=self.B,K=self.K,cost=self.cost,reviews=self.reviews,zone_xy=self.zones[self.zone],alpha=self.alpha,beta=self.beta,method=method,quantile=quantile,jobs=jj)
 def step(self,a):
  if self.terminal:raise RuntimeError('terminal engine')
  before=dict(time=self.time,cost=self.cost,zone=self.zone);self.actions+=1;e={'decision_time':self.time,'action':a,'before':before};typ=a.get('type');reason=None
  def finite(v):return isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v)
  j=self.jobs.get(a.get('job_id'));st=self.status.get(a.get('job_id'))
  if self.actions>200:reason='action_limit'
  elif typ=='stop':self.terminal=True
  elif typ=='wait':
   v=a.get('duration')
   if not finite(v) or v<=0 or self.time+v>self.H+1e-12:reason='invalid_wait'
   else:self.time=min(self.H,self.time+v)
  elif typ in ['execute','review']:
   if j is None or st['attempted'] or j['release_s']>self.time+1e-12:reason='unavailable_job'
   elif typ=='review':
    if st['reviews']>=2 or self.reviews>=self.K or self.cost+1>self.B+1e-12 or self.time+1>self.H+1e-12:reason='invalid_review'
    else:
     self.cost+=1;self.time+=1;st['reviews']+=1;self.reviews+=1;e['return_time']=self.time;e['prefix_length']=[.2,.6,1.][st['reviews']]
   else:
    b=a.get('reserve');D=math.dist(self.zones[self.zone],self.zones[j['zone_id']]);s=self.time+D+.5
    if any(not self.status[p]['completed'] for p in j['predecessors']):reason='predecessor'
    elif not finite(b) or b<.2:reason='invalid_reserve'
    elif self.cost+2+.1*b+.2*D>self.B+1e-12:reason='budget'
    elif s+b>self.H+1e-12:reason='planned_horizon'
    else:
     fee=2+.1*b+.2*D;self.cost+=fee;self.zone=j['zone_id'];st['attempted']=True;p=self.alpha*self.hidden[j['job_id']]['duration']+self.beta;C=s+p;completed=C<=self.H+1e-12;st['completed']=completed;st['completion']=min(C,self.H) if completed else None;self.time=min(self.H,s+max(p,b));e.update(service_start=s,completion=C if completed else None,reserve=b,realized_duration=p,distance=D,fee=fee,overrun=max(min(p,self.H-s)-b,0),idle=max(b-p,0) if completed else 0,censored=int(not completed),source_reference=self.hidden[j['job_id']].get('unit_id','synthetic'))
  else:reason='unknown_action'
  if reason:e['violation']=reason;self.terminal=True
  if self.time>=self.H-1e-12:self.time=self.H;self.terminal=True
  e['after']=dict(time=self.time,cost=self.cost,zone=self.zone);self.events.append(e);return e
 def metrics(self):
  W=sum(j['priority_weight'] for j in self.jobs.values());complete=ontime=tardy=0.
  for key,j in self.jobs.items():
   st=self.status[key];C=st['completion'] if st['completed'] else self.H;w=j['priority_weight'];complete+=w*st['completed'];ontime+=w*(st['completed'] and C<=j['due_s']+1e-12);tardy+=w*max(C-j['due_s'],0)
  return dict(weighted_completion_rate=complete/W,weighted_on_time_rate=ontime/W,weighted_tardiness_s=tardy,cost=self.cost,observations=self.reviews,elapsed_s=self.time,observed_overrun_s=sum(e.get('overrun',0) for e in self.events),reserved_idle_s=sum(e.get('idle',0) for e in self.events),censored_units=sum(e.get('censored',0) for e in self.events),invalid_actions=sum('violation' in e for e in self.events),valid_protocol=not any('violation' in e for e in self.events))

def independent_metrics(jobs,events,H):
 # Rebuild from action records and realized duration, ignoring Engine status/derived event metrics.
 done={};over=idle=cost=reviews=censored=invalid=0;elapsed=0
 for e in events:
  elapsed=e['after']['time']
  if 'violation' in e:invalid+=1;continue
  typ=e['action']['type']
  if typ=='review':cost+=1;reviews+=1
  if typ=='execute':
   b=e['action']['reserve'];p=e['realized_duration'];s=e['service_start'];cost+=2+.1*b+.2*e['distance'];over+=max(min(p,H-s)-b,0)
   if s+p<=H+1e-12:done[e['action']['job_id']]=min(s+p,H);idle+=max(b-p,0)
   else:censored+=1
 W=sum(j['priority_weight'] for j in jobs)
 return dict(weighted_completion_rate=sum(j['priority_weight'] for j in jobs if j['job_id'] in done)/W,weighted_on_time_rate=sum(j['priority_weight'] for j in jobs if j['job_id'] in done and done[j['job_id']]<=j['due_s']+1e-12)/W,weighted_tardiness_s=sum(j['priority_weight']*max(done.get(j['job_id'],H)-j['due_s'],0) for j in jobs),cost=cost,observations=reviews,elapsed_s=elapsed,observed_overrun_s=over,reserved_idle_s=idle,censored_units=censored,invalid_actions=invalid,valid_protocol=invalid==0)
