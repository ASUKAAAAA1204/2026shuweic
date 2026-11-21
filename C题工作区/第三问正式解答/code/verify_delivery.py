import gzip
"""Reconstruct validation schedule from public conditions and action records."""
from pathlib import Path
import hashlib,json,math,random
import numpy as np
import pandas as pd
from engine import Engine
from duration_information import features
BASE=Path(__file__).resolve().parents[1];WORK=BASE.parent
def main():
 cfg=WORK/'Restocking_Problem_and_Data_EN/Restocking_Tasks_and_Constraints';tasks=pd.read_csv(cfg/'tasks.csv').fillna('');source=pd.read_csv(cfg/'source_segment_index.csv');source=source[(source.split=='validation')&(source.eligible==1)];zones={r.zone_id:(r.x,r.y) for r in pd.read_csv(cfg/'service_zones.csv').itertuples()};results=pd.read_csv(BASE/'results/validation/trials.csv');groups={}
 with gzip.open(BASE/'results/validation/actions.jsonl.gz','rt',encoding='utf8') as f:
  for line in f:
   r=json.loads(line);key=tuple(r[k] for k in ['case_id','seed','alpha','beta','method']);groups.setdefault(key,[]).append(r)
 maxerror=0.;checked=0
 for key,events in groups.items():
  case,seed,alpha,beta,method=key;jobs=tasks[tasks.case_id==case].sort_values('job_id');lookup={r.job_id:r for r in jobs.itertuples()};rng=random.Random(seed);assigned={}
  for r in jobs.itertuples():assigned[r.job_id]=source[source.family_id==r.family_id].iloc[rng.randrange(len(source[source.family_id==r.family_id]))]
  time=0.;cost=0.;zone='Z0';completed={};attempted=set();over=idle=censored=0.;H=float(events[0]['observation']['H'])
  for row in events:
   e=row['event'];a=e['action'];assert abs(row['observation']['time']-time)<1e-9
   if a['type']=='execute':
    job=lookup[a['job_id']];assert job.job_id not in attempted and job.release_s<=time+1e-12;parents=job.predecessors.split('|') if job.predecessors else [];assert set(parents)<=set(completed)
    b=a['reserve'];D=math.dist(zones[zone],zones[job.zone_id]);s=time+D+.5;src=assigned[job.job_id];p=alpha*float(src.duration_s)+beta;assert b>=.2 and s+b<=H+1e-12;cost+=2+.1*b+.2*D;assert cost<=row['observation']['B']+1e-12
    assert e['source_reference']==src.unit_id;assert abs(e['realized_duration']-p)<1e-9;attempted.add(job.job_id);zone=job.zone_id;over+=max(min(p,H-s)-b,0)
    if s+p<=H+1e-12:completed[job.job_id]=min(s+p,H);idle+=max(b-p,0)
    else:censored+=1
    time=min(H,s+max(p,b))
   elif a['type']=='wait':assert a['duration']>0;time+=a['duration'];assert time<=H+1e-12
   else:assert a['type']=='stop'
   assert abs(time-e['after']['time'])<1e-9 and abs(cost-e['after']['cost'])<1e-9 and zone==e['after']['zone']
  W=jobs.priority_weight.sum();metrics=dict(weighted_completion_rate=sum(r.priority_weight for r in jobs.itertuples() if r.job_id in completed)/W,weighted_on_time_rate=sum(r.priority_weight for r in jobs.itertuples() if r.job_id in completed and completed[r.job_id]<=r.due_s+1e-12)/W,weighted_tardiness_s=sum(r.priority_weight*max(completed.get(r.job_id,H)-r.due_s,0) for r in jobs.itertuples()),cost=cost,elapsed_s=time,observed_overrun_s=over,reserved_idle_s=idle,censored_units=censored,observations=0,invalid_actions=0)
  match=results[(results.case_id==case)&(results.seed==seed)&(results.alpha==alpha)&(results.beta==beta)&(results.method==method)];assert len(match)==1
  for k,v in metrics.items():maxerror=max(maxerror,abs(float(match.iloc[0][k])-v))
  checked+=1
 assert checked==1800 and maxerror<1e-9
 # Boundary tests use synthetic evaluator-only data; not performance evidence.
 jobs=[dict(job_id='a',family_id='P',release_s=0,due_s=10,priority_weight=1,zone_id='Z0',predecessors=[])];hidden={'a':dict(duration=2,features={str(v):[] for v in [.2,.6,1.]})}
 def eng():return Engine(jobs,hidden,{'Z0':[0,0]},10,10,2)
 tests=[]
 for act in [dict(type='execute',job_id='a',reserve=.1),dict(type='wait',duration=0),dict(type='execute',job_id='a',reserve=float('nan'))]:
  e=eng();event=e.step(act);assert 'violation' in event;tests.append(event['violation'])
 e=eng();e.step(dict(type='review',job_id='a'));assert e.time==1 and e.cost==1 and e.status['a']['reviews']==1
 # Bad coordinates must not influence new prefix features.
 data=WORK/'处理结果/v1';feature_checks=0
 for path in sorted((data/'private').glob('*_0.2.npz'))[:20]:
  with np.load(path,allow_pickle=False) as z:sample={k:z[k].copy() for k in z.files}
  before=features(sample)
  for side in ['left_hand','right_hand']:
   invalid=~sample[side+'__quality_known']|sample[side+'__bad'];sample[side+'__position'][invalid]=1e50
  assert np.array_equal(before,features(sample));feature_checks+=1
 result=dict(passed=True,independently_reconstructed_validation_trials=checked,maximum_score_difference=maxerror,source_assignments_reconstructed=True,geometry_fees_predecessors_horizon_reconstructed=True,boundary_cases=tests,review_delay_cost_checked=True,prefix_quality_perturbation_checks=feature_checks,test_performance_evaluated=False)
 (BASE/'audit/independent_delivery_checks.json').write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps(result))
if __name__=='__main__':main()
