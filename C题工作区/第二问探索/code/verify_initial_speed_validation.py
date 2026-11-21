import json,hashlib,math
import numpy as np,pandas as pd
from request_value import BASE

def main():
 out=BASE/'results/initial_speed_validation';tab=pd.read_csv(out/'frozen_ranges.csv');train=pd.read_csv(BASE/'results/initial_speed_ranges/points.csv.gz').query("method=='source_age'");model=json.loads((BASE/'results/validation_candidates/full_train_models.json').read_text());maxdiff=0.
 for r in tab.itertuples():
  pool=train[train.dataset==r.dataset].copy();v=pool.drop_duplicates('window_id').speed/model['scale'][r.dataset];cuts=np.quantile(v,[1/3,2/3]);assert np.max(abs(cuts-[r.cut1,r.cut2]))<1e-12;base=pool[pool.age_bin==r.age_bin]
  if len(base)<100:base=pool
  local=base[np.searchsorted(cuts,base.speed/model['scale'][r.dataset],side='right')==r.speed_group];fallback=len(local)<100 or local.environment_id.nunique()<2
  if fallback:local=base
  errors=sorted(local.error/model['scale'][r.dataset]);expected=errors[math.ceil(.9*(len(errors)-1))];assert fallback==r.fallback and len(errors)==r.points;maxdiff=max(maxdiff,abs(expected-r.radius));assert maxdiff<1e-12
 d=pd.read_csv(out/'points.csv.gz');assert not d.duplicated(['variant','method','window_id','index']).any();delta=d.nade-d.radius_normalized;assert np.max(abs(d.coverage-(delta<=0).astype(float)))==0;assert np.max(abs(d.pinball-np.maximum(.9*delta,-.1*delta)))<1e-12
 keys=['method','window_id','index'];a=d[d.variant=='original'].set_index(keys);b=d[d.variant=='speed_conditioned'].set_index(keys);assert np.array_equal(a.nade,b.nade);assert np.array_equal(a.loc[a.after,'radius_normalized'],b.loc[b.after,'radius_normalized'])
 metrics=['coverage','width','pinball','nade','speed_fallback'];summaries=[]
 for scope,f in [('all',d),('before_return',d[~d.after]),('after_return',d[d.after])]:
  q=f.groupby(['variant','method','window_id','dataset','environment_id','episode_uuid','side'],as_index=False)[metrics].mean()
  for lev in [['dataset','environment_id','episode_uuid','side'],['dataset','environment_id','episode_uuid'],['dataset','environment_id'],['dataset'],[]]:q=q.groupby(['variant','method']+lev,as_index=False)[metrics].mean()
  q['scope']=scope;summaries.append(q)
 keys=['variant','method','scope'];expected=pd.concat(summaries).set_index(keys)[metrics];saved=pd.read_csv(out/'summary.csv').set_index(keys)[metrics];diff=float(abs(expected-saved).to_numpy().max());assert diff<1e-12
 audit=json.loads((out/'audit.json').read_text());assert hashlib.sha256((out/'frozen_ranges.csv').read_bytes()).hexdigest()==audit['range_hash'];assert hashlib.sha256((BASE/'results/validation_candidates/full_train_models.json').read_bytes()).hexdigest()==audit['model_hash']
 result=dict(passed=True,quantile_max_difference=maxdiff,summary_max_difference=diff,rows=len(d),scope='independent saved-error quantiles, paired invariants and arithmetic; not fresh data');(out/'independent_audit.json').write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps(result))
if __name__=='__main__':main()
