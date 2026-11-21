import json,math
import numpy as np,pandas as pd
from request_value import BASE,OLD

def main():
 out=BASE/'results/initial_speed_ranges';d=pd.read_csv(out/'points.csv.gz');support=pd.read_csv(out/'support.csv');sc=json.loads((OLD/'results/S0/fold_scales.json').read_text());base=d[d.method=='source_age'].copy();assert not d.duplicated(['method','window_id','index']).any();maximum=0.
 for r in support.itertuples():
  pool=base[(base.environment_id!=r.held)&(base.dataset==r.dataset)].copy();pool['err']=pool.error/sc[r.held][r.dataset];pool['speed_normalized']=pool.speed/sc[r.held][r.dataset];cuts=np.quantile(pool.drop_duplicates('window_id').speed_normalized,[1/3,2/3]);assert np.max(abs(cuts-[r.cut1,r.cut2]))<1e-12
  same=pool[pool.age_bin==r.age_bin];basefallback=len(same)<100
  if basefallback:same=pool
  if r.method=='source_age_speed':
   group=np.searchsorted(cuts,same.speed_normalized.to_numpy(),side='right');local=same[group==r.speed_group];fallback=len(local)<100 or local.environment_id.nunique()<2
   if fallback:local=same
  else:local=same;fallback=basefallback
  values=sorted(local.err);radius=values[math.ceil(.9*(len(values)-1))];assert len(values)==r.points and fallback==r.fallback and local.environment_id.nunique()==r.environments;maximum=max(maximum,abs(radius-r.radius));assert maximum<1e-12
  g=d[(d.environment_id==r.held)&(d.dataset==r.dataset)&(d.age_bin==r.age_bin)&(d.speed_group==r.speed_group)&(d.method==r.method)];assert len(g)>0 and np.max(abs(g.radius-radius))<1e-12
 assert np.max(abs(d.coverage-(d.err<=d.radius).astype(float)))==0
 delta=d.err-d.radius;assert np.max(abs(d.pinball-np.maximum(.9*delta,-.1*delta)))<1e-12
 metrics=['coverage','width','pinball','fallback'];q=d.groupby(['method','window_id','dataset','environment_id','episode_uuid','side'],as_index=False)[metrics].mean()
 for levels in [['dataset','environment_id','episode_uuid','side'],['dataset','environment_id','episode_uuid'],['dataset','environment_id'],['dataset'],[]]:q=q.groupby(['method']+levels,as_index=False)[metrics].mean()
 saved=pd.read_csv(out/'summary.csv').set_index('method');diff=float(abs(q.set_index('method')[metrics]-saved[metrics]).to_numpy().max());assert diff<1e-12
 audit=dict(passed=True,support_cells=len(support),max_order_statistic_difference=maximum,summary_max_difference=diff,scope='independent reconstruction from saved raw errors and initial speeds; not independent trajectory extraction');(out/'independent_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf8');print(json.dumps(audit))
if __name__=='__main__':main()
