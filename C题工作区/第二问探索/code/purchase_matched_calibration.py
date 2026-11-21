import json,time
import numpy as np,pandas as pd
from request_value import BASE,OLD,weights
from candidate_replay_engine import replay,FixedPolicy
from update_calibration_experiment import hierarchical
import s0_audit_replay as s0

def main():
 out=BASE/'results/purchase_matched_calibration';out.mkdir(exist_ok=True);start=time.perf_counter();w=pd.read_csv(s0.Q1.parent/'C题工作区/阶段交付/stage1_review/corrected_results/data/windows.csv').query("split=='train'");features=pd.read_csv(BASE/'results/causal_decay_selector/visible_features_and_labels.csv.gz').query('stage==0');raw=pd.read_csv(BASE/'results/second_request_value/candidate_windows.csv.gz');loss=raw.pivot(index='window_id',columns='method',values='raw_ade');sources=sorted(w.dataset.unique());cache={};data={}
 for r in w.itertuples():
  key=(r.episode_uuid,r.side)
  if key not in cache:
   with np.load(s0.DATA/'records'/f'{r.episode_uuid}.npz',allow_pickle=False) as z:cache[key]=(z[r.side+'__t_s'],z[r.side+'__position'].mean(axis=1))
  t,x=cache[key];i,j=r.start_index,r.end_index_exclusive;data[r.window_id]={}
  for method,schedule in [('fixed2',(.3,.6)),('first_only',(.3,))]:
   run=replay(t[i:j],x[i:j],t[i-2:i],x[i-2:i],FixedPolicy(schedule),delay=.1,budget=2,window_end=float(t[j-1]));data[r.window_id][method]=dict(error=np.array([p['error'] for p in run['points']]),after=np.array([p['returns']>0 for p in run['points']]),age_bin=np.clip(np.ceil(np.array([p['age'] for p in run['points']])*30).astype(int),1,30),returns=np.array([p['returns'] for p in run['points']]),relative=np.array([p['relative'] for p in run['points']]))
 def matrix(z,scale):
  x=z[[f'f{k}' for k in range(5)]].to_numpy().copy();x[:,:3]/=np.array([scale[s] for s in z.dataset])[:,None];return np.column_stack([x,*[(z.dataset==s).astype(float).to_numpy() for s in sources]])
 def fit(excluded,scale):
  z=features[~features.environment_id.isin(excluded)];wt=weights(z);x=matrix(z,scale);mu=np.sum(wt[:,None]*x,axis=0);sd=np.sqrt(np.sum(wt[:,None]*(x-mu)**2,axis=0));sd[sd<1e-10]=1;x=np.column_stack([np.ones(len(x)),(x-mu)/sd]);y=np.array([(loss.loc[r.window_id,'first_only']-loss.loc[r.window_id,'fixed2'])/scale[r.dataset] for r in z.itertuples()]);pen=np.eye(x.shape[1])*.1;pen[0,0]=0;coef=np.linalg.solve(x.T@(wt[:,None]*x)+pen,x.T@(wt*y));return dict(mu=mu.tolist(),sd=sd.tolist(),coefficient=coef.tolist(),fit_environments=sorted(z.environment_id.unique()))
 def evaluate(env,f,scale):
  z=features[features.environment_id==env];x=matrix(z,scale);gain=np.column_stack([np.ones(len(x)),(x-np.array(f['mu']))/np.array(f['sd'])])@np.array(f['coefficient']);frames=[]
  for r,value in zip(z.itertuples(),gain):
   buy=bool(value>.02);d=data[r.window_id]['fixed2' if buy else 'first_only'];meta={k:getattr(r,k) for k in ['window_id','dataset','environment_id','episode_uuid','side']};frames.append(pd.DataFrame(dict(**meta,index=np.arange(len(d['error'])),**d,second_bought=buy)))
  return pd.concat(frames,ignore_index=True)
 pairs=json.loads((BASE/'results/distance_nested_calibration/pair_fits.json').read_text());cal=[];fits=[]
 for number,p in enumerate(pairs,1):
  excluded=p['excluded'];f=fit(excluded,p['scale']);fits.append(dict(excluded=excluded,scale=p['scale'],**f));assert not set(excluded)&set(f['fit_environments'])
  for outer,inner in [excluded,excluded[::-1]]:
   z=evaluate(inner,f,p['scale']);z['outer']=outer;cal.append(z)
  if number%40==0:print('pair',number,flush=True)
 cal=pd.concat(cal,ignore_index=True);cal.to_csv(out/'inner_residuals.csv.gz',index=False);(out/'pair_fits.json').write_text(json.dumps(fits,indent=2),encoding='utf8');outerfits={f['held']:f for f in json.loads((BASE/'results/second_request_value/fits.json').read_text())};sc=json.loads((OLD/'results/S0/fold_scales.json').read_text());rows=[];support=[]
 for outer in sorted(w.environment_id.unique()):
  ev=evaluate(outer,outerfits[outer],sc[outer]);c=cal[cal.outer==outer];assert outer not in set(c.environment_id)
  for (src,after,age),g in ev.groupby(['dataset','after','age_bin']):
   local=c[(c.dataset==src)&(c.after==after)&(c.age_bin==age)];fallback=len(local)<100
   if fallback:local=c[(c.dataset==src)&(c.after==after)]
   radius=float(local.error.quantile(.9,interpolation='higher'));z=g.copy();delta=z.error-radius;z['radius']=radius;z['coverage']=(delta<=0).astype(float);z['width']=2*radius/sc[outer][src];z['pinball']=np.maximum(.9*delta,-.1*delta)/sc[outer][src];z['model']='adaptive_buy';rows.append(z);support.append(dict(outer=outer,dataset=src,after=after,age_bin=age,points=len(local),environments=local.environment_id.nunique(),age_fallback=fallback))
 result=pd.concat(rows);result.to_csv(out/'outer_points.csv.gz',index=False);pd.DataFrame(support).to_csv(out/'support.csv',index=False);fixed=pd.read_csv(BASE/'results/distance_nested_calibration/outer_points.csv.gz').query("model=='secant' and calibration=='pooled'");result_all=pd.concat([result,fixed]);metrics=['coverage','width','pinball'];summaries=[]
 for scope,f in [('all',result_all),('after_return',result_all[result_all.after]),('skipped_after_second_due',result[(~result.second_bought)&(result.relative>=.7)])]:
  if not len(f):continue
  g=f.groupby(['model','window_id','dataset','environment_id','episode_uuid','side'],as_index=False)[metrics].mean();s=hierarchical(g,['model'],metrics);s['scope']=scope;summaries.append(s);env=g.groupby(['model','dataset','environment_id','episode_uuid','side'],as_index=False)[metrics].mean().groupby(['model','dataset','environment_id','episode_uuid'],as_index=False)[metrics].mean().groupby(['model','dataset','environment_id'],as_index=False)[metrics].mean();env.to_csv(out/(scope+'_environments.csv'),index=False);env.groupby(['model','dataset'],as_index=False)[metrics].mean().to_csv(out/(scope+'_sources.csv'),index=False)
 summary=pd.concat(summaries);summary.to_csv(out/'summary.csv',index=False);audit=dict(pair_fits=120,inner_points=len(cal),outer_points=len(result),raw_skipped_windows=int(result[~result.second_bought].window_id.nunique()),development_reads=0,test_reads=0,formal_model_changed=False,coverage_guarantee=False,elapsed_s=time.perf_counter()-start);(out/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf8');print(summary.to_string(index=False));print(json.dumps(audit))
if __name__=='__main__':main()
