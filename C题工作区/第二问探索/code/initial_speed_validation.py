"""Frozen three-speed-bin transfer; validation never determines radii or cuts."""
import hashlib,json
import numpy as np,pandas as pd
from request_value import BASE,OLD
from update_calibration_experiment import hierarchical
import s0_audit_replay as s0

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 out=BASE/'results/initial_speed_validation';out.mkdir(exist_ok=True);modelpath=BASE/'results/validation_candidates/full_train_models.json';mh=sha(modelpath);model=json.loads(modelpath.read_text());sc=json.loads((OLD/'results/S0/fold_scales.json').read_text())
 train=pd.read_csv(BASE/'results/initial_speed_ranges/points.csv.gz').query("method=='source_age'").copy();train['error_full']=train.error/train.dataset.map(model['scale']);train['speed_full']=train.speed/train.dataset.map(model['scale']);cells=[]
 for src,pool in train.groupby('dataset'):
  cuts=np.quantile(pool.drop_duplicates('window_id').speed_full,[1/3,2/3]);pool=pool.copy();pool['group']=np.searchsorted(cuts,pool.speed_full,side='right')
  for age in range(1,31):
   base=pool[pool.age_bin==age]
   if len(base)<100:base=pool
   for group in range(3):
    local=base[base.group==group];fallback=len(local)<100 or local.environment_id.nunique()<2
    if fallback:local=base
    cells.append(dict(dataset=src,age_bin=age,speed_group=group,cut1=cuts[0],cut2=cuts[1],radius=float(local.error_full.quantile(.9,interpolation='higher')),points=len(local),environments=local.environment_id.nunique(),fallback=fallback,control_radius=float(base.error_full.quantile(.9,interpolation='higher'))))
 table=pd.DataFrame(cells);path=out/'frozen_ranges.csv';table.to_csv(path,index=False);rh=sha(path);print('Training ranges frozen',rh,flush=True)
 # Validation history contributes only the two already-visible initial frames.
 w=pd.read_csv(s0.Q1.parent/'C题工作区/阶段交付/stage1_review/corrected_results/data/windows.csv').query("split=='validation'");cache={};speeds={}
 for r in w.itertuples():
  key=(r.episode_uuid,r.side)
  if key not in cache:
   with np.load(s0.DATA/'records'/f'{r.episode_uuid}.npz',allow_pickle=False) as z:cache[key]=(z[r.side+'__t_s'],z[r.side+'__position'].mean(axis=1))
  t,x=cache[key];i=r.start_index;speeds[r.window_id]=np.linalg.norm((x[i-1]-x[i-2])/(t[i-1]-t[i-2]))/model['scale'][r.dataset]
 val=pd.read_csv(BASE/'results/validation_matched_ranges/validation_points.csv.gz');lookup=table.set_index(['dataset','age_bin','speed_group']);rows=[]
 for variant in ['original','source_age_control','speed_conditioned']:
  z=val.copy();z['variant']=variant;z['speed_fallback']=0.
  for idx,r in z[~z.after].iterrows():
   source=table[table.dataset==r.dataset].iloc[0];group=int(np.searchsorted([source.cut1,source.cut2],speeds[r.window_id],side='right'));cell=lookup.loc[(r.dataset,r.age_bin,group)]
   if variant!='original':z.at[idx,'radius_normalized']=cell.radius if variant=='speed_conditioned' else cell.control_radius
   z.at[idx,'speed_fallback']=float(cell.fallback)
  delta=z.nade-z.radius_normalized;z['coverage']=(delta<=0).astype(float);z['width']=2*z.radius_normalized;z['pinball']=np.maximum(.9*delta,-.1*delta);rows.append(z)
 result=pd.concat(rows,ignore_index=True);result.to_csv(out/'points.csv.gz',index=False);metrics=['coverage','width','pinball','nade','speed_fallback'];summaries=[]
 for scope,f in [('all',result),('before_return',result[~result.after]),('after_return',result[result.after])]:
  q=f.groupby(['variant','method','window_id','dataset','environment_id','episode_uuid','side'],as_index=False)[metrics].mean();s=hierarchical(q,['variant','method'],metrics);s['scope']=scope;summaries.append(s)
  e=q.groupby(['variant','method','dataset','environment_id','episode_uuid','side'],as_index=False)[metrics].mean().groupby(['variant','method','dataset','environment_id','episode_uuid'],as_index=False)[metrics].mean().groupby(['variant','method','dataset','environment_id'],as_index=False)[metrics].mean();e.to_csv(out/(scope+'_environments.csv'),index=False);e.groupby(['variant','method','dataset'],as_index=False)[metrics].mean().to_csv(out/(scope+'_sources.csv'),index=False)
 summary=pd.concat(summaries,ignore_index=True);summary.to_csv(out/'summary.csv',index=False);assert sha(path)==rh and sha(modelpath)==mh
 # Same prediction and unchanged return-stage radius, exact paired order.
 a=rows[0];b=rows[2];assert np.array_equal(a.nade,b.nade);assert np.array_equal(a.loc[a.after,'radius_normalized'],b.loc[b.after,'radius_normalized'])
 audit=dict(training_windows=train.window_id.nunique(),validation_windows=len(w),range_cells=len(table),validation_records=len({k[0] for k in cache}),test_reads=0,model_hash=mh,range_hash=rh,unchanged_predictions=True,unchanged_after_return_ranges=True,scope='descriptive previously viewed validation; control corrects fold-normalization transfer separately from speed effect');(out/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf8');print(summary[summary.method=='baseline'].to_string(index=False));print(json.dumps(audit))
if __name__=='__main__':main()
