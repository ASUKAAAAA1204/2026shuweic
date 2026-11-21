from pathlib import Path
import json,math
import numpy as np,pandas as pd
B=Path(__file__).resolve().parents[1];p=B/'results/neighbor_age';inp=B/'results/distance_nested_calibration'
c=pd.read_csv(inp/'inner_residuals.csv.gz');old=pd.read_csv(inp/'outer_points.csv.gz');a=pd.read_csv(p/'points.csv.gz');s=pd.read_csv(p/'support.csv');groups={k:g for k,g in c.groupby(['outer','model','dataset','after'])};mx=0.
for r in s.itertuples():
 pool=groups[(r.outer,r.model,r.dataset,r.after)];assert r.outer not in set(pool.environment_id)
 distance=np.abs(pool.age_bin.to_numpy()-r.age_bin);d=sorted(distance)[99];assert d==r.distance
 local=pool[distance<=d];assert len(local)==r.points and len(local)>=100
 values=sorted(local.error);expected=values[math.ceil(.9*(len(values)-1)-1e-12)]
 observed=a[(a.environment_id==r.outer)&(a.model==r.model)&(a.dataset==r.dataset)&(a.after==r.after)&(a.age_bin==r.age_bin)&(a.calibration=='pooled')].radius
 delta=float(abs(observed-expected).max());mx=max(mx,delta);assert delta<1e-12
z=a.merge(old,on=['window_id','index','model','calibration'],suffixes=('_new','_old'),validate='one_to_one');assert len(z)==188148
assert np.array_equal(z.error_new,z.error_old)
unchanged=z[~z.original_fallback];assert abs(unchanged.radius_new-unchanged.radius_old).max()<1e-12
assert (a.supported).all() and np.allclose(a.coverage,(a.error<=a.radius).astype(float))
# Report whether each source benefits; no model or calibration selection uses these results.
metrics=['coverage','width','pinball'];out=[]
for variant,frame in [('all_age',z.rename(columns={x+'_old':x for x in metrics})),('neighbor',a)]:
 for scope,g in [('all',frame),('after_return',frame[frame['after_old'] if variant=='all_age' else frame.after])]:
  g=g.groupby(['model','calibration','window_id','dataset','environment_id','episode_uuid','side'],as_index=False)[metrics].mean() if variant=='neighbor' else None
  if g is None:continue
  for lev in [['dataset','environment_id','episode_uuid','side'],['dataset','environment_id','episode_uuid'],['dataset','environment_id'],['dataset']]:g=g.groupby(['model','calibration']+lev,as_index=False)[metrics].mean()
  g['scope']=scope;out.append(g)
pd.concat(out).to_csv(p/'neighbor_sources.csv',index=False)
audit=dict(passed=True,checked_groups=len(s),independent_minimum_distance_order_statistic=True,radius_max_difference=mx,unchanged_error_rows=len(z),adequate_groups_unchanged=True,coverage_recomputed=True,scope='neighborhood minimality, pooled quantile, point pairing and supported outputs; env_mean formula not independently recomputed here')
(p/'independent_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf8');print(json.dumps(audit))
