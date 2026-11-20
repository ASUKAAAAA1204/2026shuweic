from common import *
from q3 import Engine,independent_metrics,Worker
import math,random

def job(name='J',release=0,due=5,weight=1,zone='Z0',pred=None):return dict(job_id=name,family_id='P',release_s=release,due_s=due,priority_weight=weight,zone_id=zone,predecessors=pred or [])
def engine(p=2,H=10,B=100,K=4,jobs=None):
 jobs=jobs or [job()];return Engine(jobs,{j['job_id']:dict(duration=p,unit_id='synthetic',features={str(l):[0]*10 for l in [.2,.6,1.]}) for j in jobs},{'Z0':[0,0],'Z1':[1,0]},H,B,K,1,0)
def run(out):
 tests=[]
 def check(name,fn):
  try:fn();tests.append(dict(name=name,passed=True))
  except Exception as ex:tests.append(dict(name=name,passed=False,error=repr(ex)))
 def execute(p,b,expected_time,expected_idle,expected_over):
  e=engine(p);e.step(dict(type='execute',job_id='J',reserve=b));assert abs(e.time-expected_time)<1e-12;assert abs(e.metrics()['reserved_idle_s']-expected_idle)<1e-12;assert abs(e.metrics()['observed_overrun_s']-expected_over)<1e-12
 check('early completion retains reservation',lambda:execute(1,3,3.5,2,0))
 check('overrun extends occupancy',lambda:execute(3,1,3.5,0,2))
 check('same-zone changeover .5',lambda:execute(2,2,2.5,0,0))
 def invalid(action,**kwargs):
  e=engine(**kwargs);before=(e.time,e.cost,e.zone);e.step(action);assert e.metrics()['invalid_actions']==1 and e.terminal;assert (e.time,e.cost,e.zone)==before
 for name,a,kw in [
 ('unreleased',dict(type='execute',job_id='J',reserve=1),{'jobs':[job(release=1)]}),
 ('unknown job',dict(type='execute',job_id='X',reserve=1),{}),
 ('reserve below .2',dict(type='execute',job_id='J',reserve=.1),{}),
 ('NaN reserve',dict(type='execute',job_id='J',reserve=float('nan')),{}),
 ('infinite reserve',dict(type='execute',job_id='J',reserve=float('inf')),{}),
 ('boolean reserve',dict(type='execute',job_id='J',reserve=True),{}),
 ('budget excess',dict(type='execute',job_id='J',reserve=1),{'B':2}),
 ('planned horizon',dict(type='execute',job_id='J',reserve=10),{}),
 ('zero wait',dict(type='wait',duration=0),{}),
 ('negative wait',dict(type='wait',duration=-1),{}),
 ('wait cross H',dict(type='wait',duration=11),{}),
 ('NaN wait',dict(type='wait',duration=float('nan')),{}),
 ('unknown action',dict(type='bad'),{}),
 ('review K',dict(type='review',job_id='J'),{'K':0}),
 ('review budget',dict(type='review',job_id='J'),{'B':.5}),
 ('review horizon',dict(type='review',job_id='J'),{'H':.5})]:check(name,lambda a=a,kw=kw:invalid(a,**kw))
 def censored():
  e=engine(p=20);e.step(dict(type='execute',job_id='J',reserve=1));m=e.metrics();assert e.time==10 and m['censored_units']==1 and m['weighted_completion_rate']==0 and m['observed_overrun_s']==8.5
 check('censor and clipped overrun',censored)
 def exact():
  e=engine(p=9.5);e.step(dict(type='execute',job_id='J',reserve=9.5));assert e.time==10 and e.status['J']['completed']
 check('completion exactly H',exact)
 def budget():
  e=engine(B=2.1);e.step(dict(type='execute',job_id='J',reserve=1));assert not e.metrics()['invalid_actions'] and abs(e.cost-2.1)<1e-12
 check('budget equality',budget)
 def repeat():
  e=engine();e.step(dict(type='execute',job_id='J',reserve=1));e.step(dict(type='execute',job_id='J',reserve=1));assert e.metrics()['invalid_actions']==1
 check('repeat rejected',repeat)
 def reviews():
  e=engine();e.step(dict(type='review',job_id='J'));assert e.time==1 and e.cost==1 and e.observe()['jobs'][0]['prefix_length']==.6;e.step(dict(type='review',job_id='J'));assert e.observe()['jobs'][0]['prefix_length']==1.;e.step(dict(type='review',job_id='J'));assert e.metrics()['invalid_actions']==1
 check('review .2 .6 1 and max two',reviews)
 def pred():
  e=engine(jobs=[job('A'),job('B',pred=['A'])]);e.step(dict(type='execute',job_id='B',reserve=1));assert e.metrics()['invalid_actions']==1
 check('predecessor execution required',pred)
 def reviewpred():
  e=engine(jobs=[job('A'),job('B',pred=['A'])]);e.step(dict(type='review',job_id='B'));assert e.reviews==1
 check('review ignores predecessor',reviewpred)
 def movement():
  e=engine(jobs=[job(zone='Z1')]);e.step(dict(type='execute',job_id='J',reserve=1));assert e.events[0]['service_start']==1.5 and abs(e.cost-2.3)<1e-12
 check('movement fee and time',movement)
 def actionlimit():
  e=engine(H=1000)
  for _ in range(200):e.step(dict(type='wait',duration=.01))
  e.step(dict(type='stop'));assert e.metrics()['invalid_actions']==1
 check('201st action invalid',actionlimit)
 def stop():
  e=engine();e.step(dict(type='stop'));m=e.metrics();assert m['elapsed_s']==0 and m['weighted_tardiness_s']==5 and m['weighted_on_time_rate']==0
 check('unfinished denominator and tardiness H',stop)
 def retain():
  e=engine();e.step(dict(type='review',job_id='J'));e.step(dict(type='wait',duration=-1));assert e.cost==1 and e.time==1
 check('invalid retains consumed resources',retain)
 def stress():
  rng=random.Random(20261120)
  for k in range(1000):
   js=[job(str(i),due=rng.uniform(1,15),zone=rng.choice(['Z0','Z1'])) for i in range(4)];e=engine(p=rng.uniform(.3,12),H=15,B=20,jobs=js)
   while not e.terminal:
    options=[]
    for j in js:
     st=e.status[j['job_id']];D=math.dist(e.zones[e.zone],e.zones[j['zone_id']]);b=rng.uniform(.2,5)
     if not st['attempted'] and e.time+D+.5+b<=e.H and e.cost+2+.1*b+.2*D<=e.B:options.append(dict(type='execute',job_id=j['job_id'],reserve=b))
    a=rng.choice(options) if options else dict(type='stop');before=e.time;e.step(a);assert e.time>=before and e.cost<=e.B+1e-9
   assert not e.metrics()['invalid_actions'];ind=independent_metrics(js,e.events,e.H);assert all(abs(float(v)-float(ind[k]))<1e-9 for k,v in e.metrics().items())
 check('1000 randomized legal traces independent metrics',stress)
 worker=Worker(Path(out)/'models/duration.json')
 try:
  check('worker denied private source open',lambda:assert_denied(worker,DATA/'metadata_private.json'))
 finally:worker.close()
 writecsv(Path(out)/'tests.csv',tests);dump(Path(out)/'test_summary.json',dict(total=len(tests),passed=sum(r['passed'] for r in tests),failed=[r for r in tests if not r['passed']]))
 assert all(r['passed'] for r in tests),[r for r in tests if not r['passed']]
 print('tests passed',len(tests),flush=True)
def assert_denied(worker,p):assert worker.call({'probe':str(p.resolve())})=={'denied':True}
