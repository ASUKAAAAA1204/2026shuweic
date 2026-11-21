"""Fixed 2x2 baselines; training only; no prefix fitting or policy search."""
from pathlib import Path
import copy,json,math,random,gzip
import numpy as np
import pandas as pd
from engine import Engine,independent_metrics
BASE=Path(__file__).resolve().parents[1];WORK=BASE.parent
MAPPINGS=[(1.,.5),(.75,.5),(1.5,.5),(1.,0.),(1.,1.)]
SEEDS=list(range(2026112000,2026112030))
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
def main():
 cfg=WORK/'Restocking_Problem_and_Data_EN/Restocking_Tasks_and_Constraints';cond=json.loads((cfg/'common_conditions.json').read_text());seg=pd.read_csv(WORK/'处理结果/v1/segments_private.csv');seg=seg[(seg.split=='train')&(seg.eligible==1)];tasks=pd.read_csv(cfg/'tasks.csv').fillna('');zz=pd.read_csv(cfg/'service_zones.csv');zones={r.zone_id:[r.x,r.y] for r in zz.itertuples()};out=BASE/'results/baseline_train';out.mkdir(parents=True,exist_ok=True);results=[];skip=[];audits=0;maxdiff=0.;assignments=[]
 with gzip.open(out/'actions.jsonl.gz','wt',encoding='utf8') as log:
  for held in sorted(seg.environment_id.unique()):
   fit=seg[seg.environment_id!=held];pool=seg[seg.environment_id==held];med=fit.groupby('family_id').duration_s.median().to_dict()
   for case in cond['cases']:
    raw=tasks[tasks.case_id==case['case_id']].sort_values('job_id');required=set(raw.family_id)
    if not required<=set(pool.family_id):skip.append(dict(environment_id=held,case_id=case['case_id']));continue
    jobs=[dict(job_id=r.job_id,family_id=r.family_id,release_s=float(r.release_s),due_s=float(r.due_s),priority_weight=float(r.priority_weight),zone_id=r.zone_id,predecessors=r.predecessors.split('|') if r.predecessors else []) for r in raw.itertuples()]
    for seed in SEEDS:
     rng=random.Random(seed);hidden={}
     for j in jobs:
      candidates=pool[pool.family_id==j['family_id']];r=candidates.iloc[rng.randrange(len(candidates))];hidden[j['job_id']]=dict(duration=float(r.duration_s),unit_id=r.unit_id,features={str(v):[] for v in [.2,.6,1.]});assignments.append(dict(environment_id=held,case_id=case['case_id'],seed=seed,job_id=j['job_id'],unit_id=r.unit_id))
     for alpha,beta in MAPPINGS:
      for ordering in ['edf','density']:
       for reservation in ['minimum','median']:
        tag=dict(environment_id=held,case_id=case['case_id'],split='train_oof',seed=seed,trial_number=seed-SEEDS[0],alpha=alpha,beta=beta,method=ordering+'_'+reservation);e=Engine(jobs,copy.deepcopy(hidden),zones,case['horizon_s'],case['cost_budget'],case['observation_limit'],alpha,beta)
        while not e.terminal:
         observation=e.observe();decision=action(observation,med,ordering,reservation);other=copy.deepcopy(e)
         for key,v in other.hidden.items():
          if not other.status[key]['attempted']:v['duration']+=12345;v['features']={str(length):[999]*10 for length in [.2,.6,1.]}
         assert decision==action(other.observe(),med,ordering,reservation);audits+=1;event=e.step(decision);log.write(json.dumps(dict(**tag,observation=observation,event=event),ensure_ascii=False)+'\n')
        m=e.metrics();ind=independent_metrics(jobs,e.events,e.H)
        for k,v in m.items():maxdiff=max(maxdiff,abs(float(v)-float(ind[k])))
        assert maxdiff<=1e-9 and m['invalid_actions']==0;results.append(dict(**tag,**m))
   print('Completed training environment',held,flush=True)
 d=pd.DataFrame(results);d.to_csv(out/'trials.csv',index=False);pd.DataFrame(assignments).to_csv(out/'assignments.csv',index=False);pd.DataFrame(skip).to_csv(out/'skipped.csv',index=False)
 metrics=['weighted_completion_rate','weighted_on_time_rate','weighted_tardiness_s','cost','observations','elapsed_s','observed_overrun_s','reserved_idle_s','censored_units','invalid_actions'];fold=d.groupby(['environment_id','case_id','alpha','beta','method'],as_index=False)[metrics].mean();fold.to_csv(out/'fold_summary.csv',index=False);summary=fold.groupby(['case_id','alpha','beta','method'],as_index=False)[metrics].mean();summary.to_csv(out/'summary.csv',index=False)
 audit=dict(passed=True,trials=len(d),skipped_environment_cases=len(skip),future_payload_invariance_checks=audits,independent_metric_max_error=maxdiff,invalid_actions=0,validation_reads=0,test_reads=0,methods=4,training_seeds=SEEDS)
 (out/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf8');print(json.dumps(audit));print(summary[(summary.alpha==1)&(summary.beta==.5)].to_string(index=False))
if __name__=='__main__':main()
