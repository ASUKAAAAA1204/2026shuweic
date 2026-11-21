import json,time
import numpy as np,pandas as pd
from request_value import BASE,OLD,weights
from update_calibration_experiment import hierarchical
from candidate_replay_engine import replay,FixedPolicy
import s0_audit_replay as s0

def main():
 out=BASE/'results/selector_matched_calibration';out.mkdir(exist_ok=True);start=time.perf_counter();w=pd.read_csv(s0.Q1.parent/'C题工作区/阶段交付/stage1_review/corrected_results/data/windows.csv').query("split=='train'");events=pd.read_csv(BASE/'results/causal_decay_selector/visible_features_and_labels.csv.gz');sources=sorted(w.dataset.unique());cache={};data={}
 for r in w.itertuples():
  key=(r.episode_uuid,r.side)
  if key not in cache:
   with np.load(s0.DATA/'records'/f'{r.episode_uuid}.npz',allow_pickle=False) as z:cache[key]=(z[r.side+'__t_s'],z[r.side+'__position'].mean(axis=1))
  t,x=cache[key];i,j=r.start_index,r.end_index_exclusive;tt=t[i:j];run=replay(tt,x[i:j],t[i-2:i],x[i-2:i],FixedPolicy(),delay=.1,budget=2,window_end=float(tt[-1]));ret=[v for v in run['actions'] if v['event']=='return'];arr=[int(np.argmin(abs(tt-v['processed_at']))) for v in ret];sample=[int(np.argmin(abs(tt-v['sample_time']))) for v in ret];errors=np.tile(np.array([p['error'] for p in run['points']])[:,None],(1,3));last=x[i-1];last_t=t[i-1];ages=tt-last_t;after=np.zeros(len(tt),bool)
  for stage in [0,1]:
   q=sample[stage];lo=arr[stage];hi=arr[stage+1] if stage==0 else len(tt);sec=(x[i+q]-last)/(tt[q]-last_t);age=tt[lo:hi]-tt[q]
   for k,tau in enumerate([.1,.2,.4]):errors[lo:hi,k]=np.linalg.norm(x[i+q]-tau*np.expm1(-age/tau)[:,None]*sec-x[i+lo:i+hi],axis=1)
   ages[lo:hi]=age;after[lo:hi]=True;last=x[i+q];last_t=tt[q]
  data[r.window_id]=dict(errors=errors,arrivals=arr,after=after,ages=np.clip(np.ceil(ages*30).astype(int),1,30))
 def matrix(frame,scale):
  x=frame[[f'f{k}' for k in range(6)]].to_numpy().copy();x[:,:3]/=np.array([scale[s] for s in frame.dataset])[:,None];return np.column_stack([x,*[(frame.dataset==s).astype(float).to_numpy() for s in sources]])
 def fit(excluded,scale):
  frame=events[~events.environment_id.isin(excluded)];x=matrix(frame,scale);wt=weights(frame)*frame.fraction.to_numpy();wt/=wt.sum();mu=np.sum(wt[:,None]*x,axis=0);sd=np.sqrt(np.sum(wt[:,None]*(x-mu)**2,axis=0));sd[sd<1e-10]=1;x=np.column_stack([np.ones(len(x)),(x-mu)/sd]);y=frame[['loss0','loss1','loss2']].to_numpy();y=(y-y[:,1,None])/np.array([scale[s] for s in frame.dataset])[:,None];pen=np.eye(x.shape[1])*.1;pen[0,0]=0;coef=np.linalg.solve(x.T@(wt[:,None]*x)+pen,x.T@(wt[:,None]*y));return dict(mu=mu.tolist(),sd=sd.tolist(),coefficient=coef.tolist(),fit_environments=sorted(frame.environment_id.unique()))
 def evaluate(env,f,scale):
  frame=events[events.environment_id==env];xx=(matrix(frame,scale)-np.array(f['mu']))/np.array(f['sd']);pick=np.argmin(np.column_stack([np.ones(len(xx)),xx])@np.array(f['coefficient']),axis=1);choice={(r.window_id,r.stage):int(k) for r,k in zip(frame.itertuples(),pick)};rows=[]
  for r in w[w.environment_id==env].itertuples():
   d=data[r.window_id];e=d['errors'][:,1].copy()
   for stage in [0,1]:
    lo=d['arrivals'][stage];hi=d['arrivals'][stage+1] if stage==0 else len(e);e[lo:hi]=d['errors'][lo:hi,choice[(r.window_id,stage)]]
   meta={k:getattr(r,k) for k in ['window_id','dataset','environment_id','episode_uuid','side']};rows.append(pd.DataFrame(dict(**meta,index=np.arange(len(e)),after=d['after'],age_bin=d['ages'],error=e)))
  return pd.concat(rows,ignore_index=True)
 pairs=json.loads((BASE/'results/distance_nested_calibration/pair_fits.json').read_text());cal=[];fits=[]
 for number,p in enumerate(pairs,1):
  excluded=p['excluded'];scale=p['scale'];f=fit(excluded,scale);assert not set(excluded)&set(f['fit_environments']);fits.append(dict(excluded=excluded,scale=scale,**f))
  for outer,inner in [excluded,excluded[::-1]]:
   z=evaluate(inner,f,scale);z['outer']=outer;cal.append(z)
  if number%40==0:print('pair',number,flush=True)
 cal=pd.concat(cal,ignore_index=True);cal.to_csv(out/'inner_residuals.csv.gz',index=False);(out/'pair_fits.json').write_text(json.dumps(fits,indent=2),encoding='utf8');outerfits={f['held']:f for f in json.loads((BASE/'results/causal_decay_selector/fits.json').read_text())};sc=json.loads((OLD/'results/S0/fold_scales.json').read_text());results=[];support=[]
 for outer in sorted(w.environment_id.unique()):
  ev=evaluate(outer,outerfits[outer],sc[outer]);c=cal[cal.outer==outer];assert outer not in set(c.environment_id)
  for (src,after,age),g in ev.groupby(['dataset','after','age_bin']):
   local=c[(c.dataset==src)&(c.after==after)&(c.age_bin==age)];fallback=len(local)<100
   if fallback:local=c[(c.dataset==src)&(c.after==after)]
   radius=float(local.error.quantile(.9,interpolation='higher'));z=g.copy();delta=z.error-radius;z['radius']=radius;z['coverage']=(delta<=0).astype(float);z['width']=2*radius/sc[outer][src];z['pinball']=np.maximum(.9*delta,-.1*delta)/sc[outer][src];z['model']='causal_selector';z['calibration']='pooled';results.append(z);support.append(dict(outer=outer,dataset=src,after=after,age_bin=age,points=len(local),environments=local.environment_id.nunique(),age_fallback=fallback))
 result=pd.concat(results);result.to_csv(out/'outer_points.csv.gz',index=False);pd.DataFrame(support).to_csv(out/'support.csv',index=False);old=pd.read_csv(BASE/'results/distance_nested_calibration/outer_points.csv.gz').query("calibration=='pooled'");allpoints=pd.concat([old,result]);metrics=['coverage','width','pinball'];summaries=[]
 for scope,f in [('all',allpoints),('after_return',allpoints[allpoints.after])]:
  g=f.groupby(['model','window_id','dataset','environment_id','episode_uuid','side'],as_index=False)[metrics].mean();s=hierarchical(g,['model'],metrics);s['scope']=scope;summaries.append(s);env=g.groupby(['model','dataset','environment_id','episode_uuid','side'],as_index=False)[metrics].mean().groupby(['model','dataset','environment_id','episode_uuid'],as_index=False)[metrics].mean().groupby(['model','dataset','environment_id'],as_index=False)[metrics].mean();env.to_csv(out/(scope+'_environments.csv'),index=False);env.groupby(['model','dataset'],as_index=False)[metrics].mean().to_csv(out/(scope+'_sources.csv'),index=False)
 summary=pd.concat(summaries);summary.to_csv(out/'summary.csv',index=False);audit=dict(pair_fits=len(fits),directed_inner_folds=len(cal[['outer','environment_id']].drop_duplicates()),inner_points=len(cal),outer_points=len(result),development_reads=0,test_reads=0,formal_model_changed=False,coverage_guarantee=False,elapsed_s=time.perf_counter()-start);(out/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf8');print(summary.to_string(index=False));print(json.dumps(audit))
if __name__=='__main__':main()
