import json
import numpy as np,pandas as pd
from request_value import BASE

def main():
 out=BASE/'results/hierarchy_weighted_calibration';cal=pd.read_csv(BASE/'results/distance_nested_calibration/inner_residuals.csv.gz').query("model=='distance_return'");support=pd.read_csv(out/'support.csv');difference=0.;checks=0
 for outer,g in support.groupby('outer'):
  c=cal[cal.outer==outer].copy();v=np.ones(len(c))/c.dataset.nunique();keys=['dataset','environment_id','episode_uuid','side','window_id']
  for k in range(1,len(keys)):v/=c.groupby(keys[:k])[keys[k]].transform('nunique').to_numpy()
  v/=c.groupby('window_id').window_id.transform('size').to_numpy();c['independent_weight']=v;assert abs(v.sum()-1)<1e-12
  for r in g.itertuples():
   local=c[(c.dataset==r.dataset)&(c.after==r.after)&(c.age_bin==r.age_bin)]
   if len(local)<100:local=c[(c.dataset==r.dataset)&(c.after==r.after)]
   pairs=sorted(zip(local.error,local.independent_weight));total=sum(vv for _,vv in pairs);running=0.;radius=None
   for error,weight in pairs:
    running+=weight
    if running>=.9*total:radius=error;break
   assert radius is not None;difference=max(difference,abs(radius-r.radius));checks+=1
 assert difference<1e-12
 audit=dict(passed=True,groups=checks,max_radius_difference=difference,scope='independent group-count weights and scalar weighted CDF; not independent residual refit')
 (out/'independent_weight_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf8');print(json.dumps(audit))
if __name__=='__main__':main()
