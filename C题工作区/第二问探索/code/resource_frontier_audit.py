from pathlib import Path
import json
import numpy as np,pandas as pd
BASE=Path(__file__).resolve().parents[1];p=BASE/'results/resource_curve_safe';m=pd.read_csv(p/'all_windows.csv.gz');old=pd.read_csv(BASE/'results/resource_curve_8/all_windows.csv.gz')
keys=['window_id','length_s','delay_s','budget_cap'];b=m.merge(old,on=keys,suffixes=('_safe','_old'),validate='one_to_one');assert len(b)==41229
err=float(np.abs(b.nade_safe-b.nade_old).max());assert err<1e-12;assert (b.requests_safe<=b.requests_old).all()
metrics=['nade','exceed_02','exceed_05','exceed_10','requests','waiting_s','nfde'];g=m.copy()
for lev in [['dataset','environment_id','episode_uuid','side'],['dataset','environment_id','episode_uuid'],['dataset','environment_id'],['dataset'],[]]:g=g.groupby(['length_s','delay_s','budget_cap']+lev,as_index=False)[metrics].mean()
expected=pd.read_csv(p/'summary.csv');delta=float(np.abs(g.set_index(['length_s','delay_s','budget_cap'])[metrics]-expected.set_index(['length_s','delay_s','budget_cap'])[metrics]).to_numpy().max());assert delta<1e-12
s=expected.copy();s['policy']=['uniform_B'+str(v) for v in s.budget_cap]
f=pd.read_csv(BASE/'results/calibration_closed_loop/summary.csv').query("method=='fixed'").iloc[0]
extra=dict(length_s=1.,delay_s=.1,budget_cap=2,nade=f.nade,requests=f.requests,waiting_s=f.waiting_s,nfde=f.nfde,exceed_05=f.exceed05,policy='fixed_03_06')
s=pd.concat([s,pd.DataFrame([extra])],ignore_index=True)
front=[];intervals=[]
for delay,group in s.groupby('delay_s'):
 group=group.reset_index(drop=True)
 for i,r in group.iterrows():
  dominated=any((o.requests<=r.requests+1e-12 and o.nade<=r.nade+1e-12 and (o.requests<r.requests-1e-12 or o.nade<r.nade-1e-12)) for j,o in group.iterrows() if j!=i)
  front.append(dict(**r.to_dict(),pareto_nade_cost=not dominated))
 # Exact lower envelope of observed mean NADE + lambda*mean paid requests.
 cuts={0.}
 for i,r in group.iterrows():
  for j,o in group.iterrows():
   if i<j and abs(r.requests-o.requests)>1e-12:
    c=(o.nade-r.nade)/(r.requests-o.requests)
    if c>0:cuts.add(float(c))
 cuts=sorted(cuts)
 for lo,hi in zip(cuts,cuts[1:]+[np.inf]):
  mid=(lo+hi)/2 if np.isfinite(hi) else lo+1
  row=group.loc[(group.nade+mid*group.requests).idxmin()]
  if intervals and intervals[-1]['delay_s']==delay and intervals[-1]['policy']==row.policy:intervals[-1]['lambda_hi']=hi
  else:intervals.append(dict(delay_s=delay,lambda_lo=lo,lambda_hi=hi,policy=row.policy,nade=row.nade,requests=row.requests))
pd.DataFrame(front).to_csv(p/'finite_pareto.csv',index=False);pd.DataFrame(intervals).to_csv(p/'cost_weight_intervals.csv',index=False)
audit=dict(passed=True,paired_old_new=len(b),prediction_error_metric_max_difference=err,summary_recompute_max_difference=delta,removed_requests_unweighted=int((b.requests_old-b.requests_safe).sum()),development_reads=0,test_reads=0,frontier_scope='finite observed policies; NADE and mean paid requests only; not theoretical optimum',utility='mean NADE + lambda * mean requests, lambda>=0')
(p/'independent_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf8');print(json.dumps(audit));print(pd.DataFrame(intervals).query('delay_s==.1').to_string(index=False))
