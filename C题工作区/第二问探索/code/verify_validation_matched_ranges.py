import hashlib,json,math
import numpy as np,pandas as pd
from distance_aligned_return import BASE

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
 root=BASE/'results/validation_matched_ranges';train=pd.read_csv(root/'training_oof_points.csv.gz');val=pd.read_csv(root/'validation_points.csv.gz');ranges=pd.read_csv(root/'frozen_ranges.csv');audit=json.loads((root/'audit.json').read_text());
 assert train.method.nunique()==3 and val.method.nunique()==3 and len(val)==19608
 assert not train.duplicated(['method','window_id','index']).any();assert not val.duplicated(['method','window_id','index']).any()
 assert train.groupby('method').window_id.nunique().eq(1527).all();assert val.groupby('method').window_id.nunique().eq(212).all()
 assert ranges.groupby('method').size().eq(180).all()
 # Recompute empirical higher quantiles with a scalar order statistic.
 max_radius=0.
 for row in ranges.itertuples():
  pool=train[(train.method==row.method)&(train.dataset==row.dataset)&(train.after==row.after)]
  cell=pool[pool.age_bin==row.age_bin];fallback=len(cell)<100
  if fallback:cell=pool
  q=sorted(float(v) for v in cell.nade);expected=q[math.ceil(.9*(len(q)-1))]
  assert fallback==row.age_fallback and len(cell)==row.points and abs(expected-row.radius_normalized)<1e-14
  max_radius=max(max_radius,abs(expected-row.radius_normalized))
 # Recompute point errors and interval statistics from saved vectors and radii.
 prediction=np.stack([json.loads(v) for v in val.prediction]);reference=np.stack([json.loads(v) for v in val.reference]);model=json.loads((BASE/'results/validation_candidates/full_train_models.json').read_text())
 err=np.linalg.norm(prediction-reference,axis=1)/np.asarray([model['scale'][s] for s in val.dataset])
 assert np.max(abs(err-val.nade))<1e-12
 assert np.max(abs((val.nade<=val.radius_normalized).astype(float)-val.coverage))==0
 assert np.max(abs(2*val.radius_normalized-val.width))<1e-12
 summary=pd.read_csv(root/'summary.csv');metrics=['coverage','width','pinball','nade'];calc=[]
 for scope,frame in [('all',val),('after_return',val[val.after])]:
  z=frame.groupby(['method','window_id','dataset','environment_id','episode_uuid','side'],as_index=False)[metrics].mean()
  for method,g in z.groupby('method'):
   levels=['dataset','environment_id','episode_uuid','side']
   g=g.groupby(levels,as_index=False)[metrics].mean().groupby(['dataset','environment_id','episode_uuid'],as_index=False)[metrics].mean().groupby(['dataset','environment_id'],as_index=False)[metrics].mean().groupby('dataset',as_index=False)[metrics].mean()[metrics].mean()
   calc.append(dict(method=method,scope=scope,**g.to_dict()))
 expected=pd.DataFrame(calc).set_index(['method','scope']).sort_index();saved=summary.set_index(['method','scope'])[metrics].sort_index();max_summary=float(abs(expected-saved).to_numpy().max());assert max_summary<1e-12
 assert audit['test_reads']==0 and audit['model_and_ranges_unchanged'] and audit['coverage_guarantee'] is False
 result=dict(passed=True,train_points=len(train),validation_points=len(val),range_rows=len(ranges),max_scalar_quantile_difference=max_radius,max_recomputed_hierarchical_summary_difference=max_summary,max_point_error_difference=float(np.max(abs(err-val.nade))),model_hash_verified=digest(BASE/'results/validation_candidates/full_train_models.json')==audit['model_hash'],range_hash_verified=digest(root/'frozen_ranges.csv')==audit['range_hash'],scope='independent order-statistic and aggregation check; not raw trajectory replay')
 (root/'independent_audit.json').write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps(result))
if __name__=='__main__':main()
