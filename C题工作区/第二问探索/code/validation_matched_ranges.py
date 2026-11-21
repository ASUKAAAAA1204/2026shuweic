"""Descriptive, fixed-rule OOF-to-validation range transfer. No validation fitting."""
import hashlib,json
import numpy as np,pandas as pd
from distance_aligned_return import BASE,OLD
from candidate_replay_engine import replay,FixedPolicy
from candidate_replay_integration import DistanceState
from online_corrected_selector import OnlineCorrectedSelector
from update_calibration_experiment import hierarchical
import s0_audit_replay as s0

METHODS=['baseline','distance_return','crossfit_corrected']
METRICS=['coverage','width','pinball','nade']
KEYS=['window_id','dataset','environment_id','episode_uuid','side']

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
 out=BASE/'results/validation_matched_ranges';out.mkdir(exist_ok=True)
 modelpath=BASE/'results/validation_candidates/full_train_models.json';modelhash=sha(modelpath);model=json.loads(modelpath.read_text())
 w=pd.read_csv(s0.Q1.parent/'C题工作区/阶段交付/stage1_review/corrected_results/data/windows.csv');train=w[w.split=='train'];val=w[w.split=='validation'];assert not set(train.environment_id)&set(val.environment_id)
 scales=json.loads((OLD/'results/S0/fold_scales.json').read_text());distance={f['held']:f for f in json.loads((BASE/'results/distance_aligned_return/folds.json').read_text())};cross={f['held']:f for f in json.loads((BASE/'results/crossfit_corrected_selector/fits.json').read_text())}
 cache={};reads={'train':set(),'validation':set()}
 def run_window(r,method,scale,coeff,fit):
  key=(r.episode_uuid,r.side)
  if key not in cache:
   with np.load(s0.DATA/'records'/f'{r.episode_uuid}.npz',allow_pickle=False) as z:cache[key]=(z[r.side+'__t_s'],z[r.side+'__position'].mean(axis=1))
  reads[r.split].add(r.episode_uuid);t,x=cache[key];i,j=r.start_index,r.end_index_exclusive;factory=None
  if method=='distance_return':factory=lambda *a:DistanceState(*a,coefficients=coeff)
  if method=='crossfit_corrected':factory=lambda *a:OnlineCorrectedSelector(*a,fit=fit,scale=scale[r.dataset],source=r.dataset,sources=model['sources'])
  run=replay(t[i:j],x[i:j],t[i-2:i],x[i-2:i],FixedPolicy(),delay=.1,budget=2,window_end=float(t[j-1]),state_factory=factory)
  points=pd.DataFrame(run['points']);points['index']=np.arange(len(points));points['after']=points.returns>0;points['age_bin']=np.clip(np.ceil(points.age*30).astype(int),1,30);points['nade']=points.error/scale[r.dataset]
  for k in KEYS:points[k]=getattr(r,k)
  points['method']=method
  return points
 residuals=[]
 for r in train.itertuples():
  env=r.environment_id;f=distance[env];c=cross[env];assert env not in f['fit_environments'] and env not in c['fit_environments']
  for p in c['event_provenance']:assert env in p['excluded'] and p['target'] in p['excluded'] and not set(p['excluded'])&set(p['fit_environments'])
  for method in METHODS:
   p=run_window(r,method,scales[env],[[f['first_a'],f['first_b']],[f['second_a'],f['second_b']]],c)
   residuals.append(p[KEYS+['method','index','after','age_bin','nade']])
 residuals=pd.concat(residuals,ignore_index=True);residuals.to_csv(out/'training_oof_points.csv.gz',index=False)
 cells=[]
 for method in METHODS:
  for source in model['sources']:
   for after in [False,True]:
    pool=residuals[(residuals.method==method)&(residuals.dataset==source)&(residuals.after==after)]
    for age in range(1,31):
     local=pool[pool.age_bin==age];fallback=len(local)<100
     if fallback:local=pool
     assert len(local)>0
     cells.append(dict(method=method,dataset=source,after=after,age_bin=age,radius_normalized=float(local.nade.quantile(.9,interpolation='higher')),points=len(local),environments=local.environment_id.nunique(),age_fallback=fallback))
 table=pd.DataFrame(cells);tablepath=out/'frozen_ranges.csv';table.to_csv(tablepath,index=False);rangehash=sha(tablepath);assert not reads['validation'];print('Training-only ranges frozen',rangehash,flush=True)
 lookup=table.set_index(['method','dataset','after','age_bin']).radius_normalized.to_dict();rows=[]
 for r in val.itertuples():
  for method in METHODS:
   p=run_window(r,method,model['scale'],[model['first'],model['second']],model['crossfit']);p['radius_normalized']=[lookup[(method,r.dataset,bool(a),int(b))] for a,b in zip(p.after,p.age_bin)];p['radius']=p.radius_normalized*model['scale'][r.dataset];delta=p.nade-p.radius_normalized;p['coverage']=(delta<=0).astype(float);p['width']=2*p.radius_normalized;p['pinball']=np.maximum(.9*delta,-.1*delta);rows.append(p)
 result=pd.concat(rows,ignore_index=True);result.to_csv(out/'validation_points.csv.gz',index=False)
 summaries=[]
 for scope,frame in [('all',result),('after_return',result[result.after])]:
  windows=frame.groupby(['method']+KEYS,as_index=False)[METRICS].mean();s=hierarchical(windows,['method'],METRICS);s['scope']=scope;summaries.append(s)
  e=windows.groupby(['method','dataset','environment_id','episode_uuid','side'],as_index=False)[METRICS].mean().groupby(['method','dataset','environment_id','episode_uuid'],as_index=False)[METRICS].mean().groupby(['method','dataset','environment_id'],as_index=False)[METRICS].mean();e.to_csv(out/(scope+'_environments.csv'),index=False);e.groupby(['method','dataset'],as_index=False)[METRICS].mean().to_csv(out/(scope+'_sources.csv'),index=False)
 summary=pd.concat(summaries,ignore_index=True);summary.to_csv(out/'summary.csv',index=False)
 current=result.groupby(['method','window_id']).nade.agg(['mean','last']);prior=pd.read_csv(BASE/'results/validation_candidates/windows.csv.gz').query('method in @METHODS').set_index(['method','window_id']);diff=max(float(abs(current['mean']-prior.nade).max()),float(abs(current['last']-prior.nfde).max()));assert diff<1e-10;assert sha(modelpath)==modelhash and sha(tablepath)==rangehash
 audit=dict(training_windows=len(train),validation_windows=len(val),training_points=len(residuals),validation_points=len(result),range_cells=len(table),position_metric_max_difference=diff,model_hash=modelhash,range_hash=rangehash,model_and_ranges_unchanged=True,train_record_reads=len(reads['train']),validation_record_reads=len(reads['validation']),test_reads=0,coverage_guarantee=False,scope='descriptive OOF-to-full-fit transfer; validation previously viewed; no validation calibration or tuning')
 (out/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf8');print(summary.to_string(index=False));print(json.dumps(audit))
if __name__=='__main__':main()
