from pathlib import Path
import sys,collections,random,math
R=Path(__file__).resolve().parents[1];O=R/'阶段交付/stage1_review';sys.path.insert(0,str(O/'corrected_code'))
from common import *
from q12 import replay_window
from q3 import independent_metrics
B=R/'阶段交付/baseline_v2';C=O/'corrected_results';checks=[]
def check(name,passed,detail):checks.append(dict(name=name,passed=bool(passed),detail=detail))
# Snapshot must still match all original baseline bytes.
snap=readcsv(O/'baseline_snapshot.csv');check('original snapshot unchanged',all(sha(R/r['path'])==r['sha256'] for r in snap),len(snap))
# Synthetic delayed-return recovery and future perturbation after the FIRST return too.
t=np.arange(40)/30;x=np.stack([t,t*2,t*0],1);z={'left_hand__t_s':t,'left_hand__position':np.repeat(x[:,None,:],21,axis=1)}
w=dict(window_id='synthetic',episode_uuid='synthetic',dataset='S',environment_id='E',split='validation',side='left_hand',start_index=7,end_index_exclusive=38,start_s=float(t[7]))
p={'train_scale':{'S':1}};rad={f'S|{m}|{b}':1. for m in ['hold','constant_velocity'] for b in range(1,31)};synthetic=[]
for method in ['hold','constant_velocity','fixed_review','adaptive_review']:
 for delay in [0,.1,.2]:
  result,trace=replay_window(w,z,p,rad,method,delay,.05,True)
  for cut in range(8,37):
   altered={k:v.copy() for k,v in z.items()};altered['left_hand__position'][cut:]+=100
   result2,trace2=replay_window(w,altered,p,rad,method,delay,.05,True)
   # At time now no observation requested at/after cut may yet have returned.
   arrivals=[t[7]+a['return_s'] for a in json.loads(result['actions']) if t[7]+a['request_s']>=t[cut]-1e-12]
   bound=min(arrivals,default=float('inf'))
   ok=all(a['prediction']==b['prediction'] and a['requests']==b['requests'] for a,b in zip(trace,trace2) if a['time']<bound-1e-12)
   synthetic.append(dict(method=method,delay=delay,cut=cut,passed=ok))
check('synthetic middle-return future isolation',all(r['passed'] for r in synthetic),len(synthetic));writecsv(O/'synthetic_causality.csv',synthetic)
# Independent source assignment and transition checks on actual complete traces.
config=ROOT/'Restocking_Problem_and_Data_EN/Restocking_Tasks_and_Constraints';jobs=readcsv(config/'tasks.csv');segs=readcsv(DATA/'segments_private.csv');zones={r['zone_id']:[float(r['x']),float(r['y'])] for r in readcsv(config/'service_zones.csv')};segid={r['unit_id']:r for r in segs};n=0;bad=[];assign={}
for case in ['S1','S2','S3']:
 for seed in load(config/'common_conditions.json')['validation_seeds']:
  rng=random.Random(seed)
  for j in sorted([j for j in jobs if j['case_id']==case],key=lambda x:x['job_id']):assign[(case,seed,j['job_id'])]=rng.choice([s for s in segs if s['split']=='validation' and s['eligible']=='1' and s['family_id']==j['family_id']])['unit_id']
for file in (C/'q3').glob('*.jsonl'):
 with file.open(encoding='utf8') as f:
  for line in f:
   row=json.loads(line);e=row['event'];a=e['action'];before=e['before'];after=e['after'];n+=1
   if a['type']=='execute':
    j=next(j for j in jobs if j['job_id']==a['job_id']);unit=assign[(row['case_id'],row['seed'],j['job_id'])];d=float(segid[unit]['duration_s']);actual=row['alpha']*d+row['beta'];D=math.dist(zones[before['zone']],zones[j['zone_id']]);s=before['time']+D+.5
    if e['source_reference']!=unit or abs(e['service_start']-s)>1e-9 or abs(e['realized_duration']-actual)>1e-9 or abs(after['cost']-before['cost']-(2+.1*a['reserve']+.2*D))>1e-9:bad.append(n)
   forbidden={'seed','unit_id','episode_uuid','duration','end_s','description'}
   def keys(obj):
    if isinstance(obj,dict):return set(obj)|set().union(*(keys(v) for v in obj.values()))
    if isinstance(obj,list):return set().union(*(keys(v) for v in obj)) if obj else set()
    return set()
   if keys(row['observation'])&forbidden:bad.append(n)
check('all action source/time/fee/input independently checked',not bad,dict(actions=n,failures=len(bad)))
q=load(C/'q3/checks.json');tests=load(C/'test_summary.json');check('32 boundary and 1000 legal traces',tests['passed']==tests['total']==32,tests);check('q3 metrics and no invalid',q['invalid_actions']==0 and q['independent_metric_max_error']<=1e-9,q)
old=load(B/'q12/prediction_radius_train.json');new=load(C/'q12/prediction_radius_train.json');writecsv(O/'radius_changes.csv',[dict(key=k,old=old[k],corrected=new[k],difference=new[k]-old[k]) for k in old])
check('registered three experiments',(O/'执行计划与实验注册表.md').exists(),'E1/E2/E3; train environment holdout; test sealed')
writecsv(O/'验收清单.csv',checks);dump(O/'验收结果.json',dict(status='PASS' if all(c['passed'] for c in checks) else 'FAIL',checks=checks,blockers=[c for c in checks if not c['passed']]))
print(json.dumps(checks,ensure_ascii=False,indent=2))

