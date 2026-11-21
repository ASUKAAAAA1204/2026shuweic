from pathlib import Path
import json,math
import numpy as np,pandas as pd
BASE=Path(__file__).resolve().parents[1];p=BASE/'results/distance_nested_calibration'
def q90(v):
 v=sorted(v);return v[math.ceil(.9*(len(v)-1)-1e-12)]
def main():
 fits=json.loads((p/'pair_fits.json').read_text());assert len(fits)==120
 for f in fits:assert len(f['excluded'])==2 and not set(f['excluded'])&set(f['fit_environments']) and len(f['fit_environments'])==14
 c=pd.read_csv(p/'inner_residuals.csv.gz');e=pd.read_csv(p/'outer_points.csv.gz');support=pd.read_csv(p/'support.csv');assert not (c.outer==c.environment_id).any()
 assert e.groupby(['model','calibration']).window_id.nunique().eq(1527).all()
 assert not e.duplicated(['window_id','model','calibration','index']).any()
 groups={key:g for key,g in c.groupby(['outer','model','dataset','after'])};maxdiff=0.
 radii=e.groupby(['environment_id','model','dataset','after','age_bin','calibration']).radius.agg(['min','max']);assert abs(radii['max']-radii['min']).max()<1e-12
 for r in support.itertuples():
  pool=groups[(r.outer,r.model,r.dataset,r.after)];local=pool[pool.age_bin==r.age_bin]
  fallback=len(local)<100
  if fallback:local=pool
  assert fallback==r.age_fallback and len(local)==r.points
  pooled=q90(local.error);vals=[q90(g.error) for _,g in local.groupby('environment_id') if len(g)>=20]
  for method,expected in [('pooled',pooled),('env_mean',sum(vals)/len(vals) if len(vals)>=2 else pooled)]:
   actual=radii.loc[(r.outer,r.model,r.dataset,r.after,r.age_bin,method),'min'];d=abs(actual-expected);maxdiff=max(maxdiff,float(d));assert d<1e-12
 assert np.allclose(e.coverage,(e.error<=e.radius).astype(float))
 metrics=['coverage','width','pinball'];s=pd.read_csv(p/'summary.csv');maxsummary=0.
 for scope,z in [('all',e),('after_return',e[e.after])]:
  z=z.groupby(['model','calibration','window_id','dataset','environment_id','episode_uuid','side'],as_index=False)[metrics].mean()
  for lev in [['dataset','environment_id','episode_uuid','side'],['dataset','environment_id','episode_uuid'],['dataset','environment_id'],['dataset'],[]]:z=z.groupby(['model','calibration']+lev,as_index=False)[metrics].mean()
  expected=s[s.scope==scope].set_index(['model','calibration'])[metrics];d=float(abs(z.set_index(['model','calibration'])[metrics]-expected).to_numpy().max());maxsummary=max(maxsummary,d);assert d<1e-12
 audit=dict(passed=True,pair_fits=120,no_excluded_env_in_fit=True,outer_not_in_calibration_points=True,independent_order_statistic_radius_max_difference=maxdiff,summary_max_difference=maxsummary,scope='saved fit provenance, calibration quantiles, coverage and aggregation; not independent training refit')
 (p/'independent_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf8');print(json.dumps(audit))
if __name__=='__main__':main()
