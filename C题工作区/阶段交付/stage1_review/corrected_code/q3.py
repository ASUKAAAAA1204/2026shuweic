from common import *
import math,random,subprocess,collections,copy

class Worker:
 def __init__(self,model):
  self.p=subprocess.Popen([sys.executable,'-I',str(Path(__file__).with_name('policy_worker.py')),str(Path(model).resolve())],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,encoding='utf8')
 def call(self,o):
  self.p.stdin.write(json.dumps(o,allow_nan=False)+'\n');self.p.stdin.flush();line=self.p.stdout.readline()
  if not line:raise RuntimeError('policy exited: '+self.p.stderr.read())
  r=json.loads(line)
  if 'error' in r:raise RuntimeError(r['error'])
  return r
 def close(self):
  self.p.stdin.close();self.p.wait(timeout=10)

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

def run(out):
 out=Path(out);q=out/'q3';q.mkdir(exist_ok=True);cond=load(CONFIG/'common_conditions.json');raw=readcsv(CONFIG/'tasks.csv');zones={r['zone_id']:[float(r['x']),float(r['y'])] for r in readcsv(CONFIG/'service_zones.csv')};segs=readcsv(DATA/'segments_private.csv');pools={f:[s for s in segs if s['split']=='validation' and s['family_id']==f and s['eligible']=='1'] for f in ['P','A','L']};features={}
 for pool in pools.values():
  for s in pool:
   features[s['unit_id']]={}
   for length in [.2,.6,1.]:
    with np.load(DATA/'private'/(s['unit_id']+'_'+str(length)+'.npz')) as p:features[s['unit_id']][str(length)]=prefix_features(p).tolist()
 methods=['family_edf','family_density','ridge_edf','ridge_review_edf'];mappings=[(1,.5),(.75,.5),(1.5,.5),(1,0),(1,1)];results=[];assignments=[];worker=Worker(out/'models/duration.json');diffmax=0.;leaktests=0
 try:
  for method in methods:
   for quantile in [.5,.8,.9]:
    with (q/(method+'_'+str(quantile)+'.jsonl')).open('w',encoding='utf8') as log:
     for alpha,beta in mappings:
      for case in cond['cases']:
       jobs=[]
       for r in sorted([r for r in raw if r['case_id']==case['case_id']],key=lambda r:r['job_id']):
        jobs.append({**r,**{k:float(r[k]) for k in ['release_s','due_s','priority_weight']},'predecessors':r['predecessors'].split('|') if r['predecessors'] else []})
       for trial,seed in enumerate(cond['validation_seeds']):
        rng=random.Random(int(seed));hidden={}
        for j in jobs:
         s=rng.choice(pools[j['family_id']]);hidden[j['job_id']]=dict(duration=float(s['duration_s']),unit_id=s['unit_id'],features=features[s['unit_id']])
         if method==methods[0] and quantile==.5 and alpha==1 and beta==.5:assignments.append(dict(case_id=case['case_id'],seed=seed,job_id=j['job_id'],unit_id=s['unit_id']))
        e=Engine(jobs,hidden,zones,case['horizon_s'],case['cost_budget'],case['observation_limit'],alpha,beta);tag=dict(method=method,quantile=quantile,case_id=case['case_id'],split='validation',trial_number=trial,seed=seed,alpha=alpha,beta=beta)
        while not e.terminal:
         o=e.observe(method,quantile);a=worker.call(o)
         # At every decision, perturb only still-hidden durations/prefix extensions.
         other=copy.deepcopy(e)
         other.hidden={key:{**v,'features':{length:list(f) for length,f in v['features'].items()}} for key,v in other.hidden.items()}
         for key,hidden_value in other.hidden.items():
          st=other.status[key]
          if not st['attempted']:hidden_value['duration']+=100
          current=[.2,.6,1.][st['reviews']]
          for length in [.2,.6,1.]:
           if length>current or other.jobs[key]['release_s']>other.time:hidden_value['features'][str(length)]=[999]*10
         assert worker.call(other.observe(method,quantile))==a;leaktests+=1
         event=e.step(a);log.write(json.dumps(dict(**tag,observation=o,event=event),ensure_ascii=False,allow_nan=False)+'\n')
        m=e.metrics();ind=independent_metrics(jobs,e.events,e.H)
        for k,v in m.items():diffmax=max(diffmax,abs(float(v)-float(ind[k])))
        assert all(abs(float(m[k])-float(ind[k]))<=1e-9 for k in m)
        results.append(dict(**tag,**m))
    print('q3',method,quantile,'complete',len(results),flush=True)
 finally:worker.close()
 writecsv(q/'trials.csv',results);writecsv(q/'assignments_private.csv',assignments)
 summaries=[];buckets=collections.defaultdict(list)
 for r in results:buckets[tuple(r[k] for k in ['method','quantile','case_id','alpha','beta'])].append(r)
 for key,rr in buckets.items():
  for metric in ['weighted_completion_rate','weighted_on_time_rate','weighted_tardiness_s','cost','observations','elapsed_s','observed_overrun_s','reserved_idle_s','censored_units','invalid_actions']:
   vals=np.asarray([r[metric] for r in rr]);rng=np.random.default_rng(20261120);boot=vals[rng.integers(0,len(vals),(2000,len(vals)))].mean(1);summaries.append(dict(zip(['method','quantile','case_id','alpha','beta'],key))|dict(metric=metric,n=len(vals),mean=float(vals.mean()),sd=float(vals.std(ddof=1)),lo=float(np.quantile(boot,.025)),hi=float(np.quantile(boot,.975))))
 writecsv(q/'summary.csv',summaries)
 paired=[]
 lookup={(r['method'],r['quantile'],r['case_id'],r['alpha'],r['beta'],r['seed']):r for r in results}
 for r in results:
  if r['method']=='family_edf':continue
  b=lookup[('family_edf',r['quantile'],r['case_id'],r['alpha'],r['beta'],r['seed'])];paired.append({k:r[k] for k in ['method','quantile','case_id','alpha','beta','seed']}|{'on_time_difference':r['weighted_on_time_rate']-b['weighted_on_time_rate'],'cost_difference':r['cost']-b['cost']})
 writecsv(q/'paired_differences.csv',paired)
 dump(q/'checks.json',dict(trials=len(results),expected=5400,invalid_actions=sum(r['invalid_actions'] for r in results),independent_metric_max_error=diffmax,hidden_future_invariance_checks=leaktests,test_evaluated=False))
