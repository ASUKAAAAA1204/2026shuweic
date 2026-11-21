import json,math
import numpy as np,pandas as pd
from request_value import BASE,OLD

def main():
 out=BASE/'results/return_speed_ranges';raw=pd.read_csv(out/'raw_points.csv.gz');points=pd.read_csv(out/'points.csv.gz');support=pd.read_csv(out/'support.csv');sc=json.loads((OLD/'results/S0/fold_scales.json').read_text());assert not raw.duplicated(['window_id','index']).any();assert (raw.available_at<=raw.time+1e-12).all() and (raw.sample_time<=raw.available_at).all();assert raw.window_id.nunique()==1527;maximum=0.
 for r in support.itertuples():
  pool=raw[(raw.environment_id!=r.held)&(raw.dataset==r.dataset)].copy();scale=sc[r.held][r.dataset];cuts=np.quantile(pool.drop_duplicates(['window_id','stage']).speed/scale,[1/3,2/3]);assert np.max(abs(cuts-[r.cut1,r.cut2]))<1e-12;base=pool[pool.age_bin==r.age_bin];bf=len(base)<100
  if bf:base=pool
  fb=bf;cal=base
  if r.method=='source_age_speed':
   cal=base[np.searchsorted(cuts,base.speed/scale,side='right')==r.speed_group];fb=len(cal)<100 or cal.environment_id.nunique()<2
   if fb:cal=base
  ordered=sorted(cal.error/scale);radius=ordered[math.ceil(.9*(len(ordered)-1))];assert len(cal)==r.points and fb==r.fallback;maximum=max(maximum,abs(radius-r.radius));assert maximum<1e-12
  g=points[(points.environment_id==r.held)&(points.dataset==r.dataset)&(points.age_bin==r.age_bin)&(points.speed_group==r.speed_group)&(points.method==r.method)];assert len(g)>0 and np.max(abs(g.radius-radius))<1e-12
 assert not points.duplicated(['method','window_id','index']).any();delta=points.err-points.radius;assert np.array_equal(points.coverage,(delta<=0).astype(float));assert np.max(abs(points.pinball-np.maximum(.9*delta,-.1*delta)))<1e-12
 metrics=['coverage','width','pinball','fallback'];q=points.groupby(['method','window_id','dataset','environment_id','episode_uuid','side'],as_index=False)[metrics].mean()
 for lev in [['dataset','environment_id','episode_uuid','side'],['dataset','environment_id','episode_uuid'],['dataset','environment_id'],['dataset'],[]]:q=q.groupby(['method']+lev,as_index=False)[metrics].mean()
 saved=pd.read_csv(out/'summary.csv').set_index('method');diff=float(abs(q.set_index('method')[metrics]-saved[metrics]).to_numpy().max());assert diff<1e-12
 audit=dict(passed=True,raw_points=len(raw),support_cells=len(support),quantile_max_difference=maximum,summary_max_difference=diff,scope='saved event timing and residual arithmetic; not independent raw replay');(out/'independent_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf8');print(json.dumps(audit))
if __name__=='__main__':main()
