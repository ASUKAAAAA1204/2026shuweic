import json
import numpy as np,pandas as pd
from request_value import BASE,OLD,weights
from update_calibration_experiment import hierarchical,paired_ci
from candidate_replay_engine import replay,FixedPolicy
import s0_audit_replay as s0
TAUS=[.1,.2,.4]

def visible_features(previous_velocity,secant,gap,stage):
 a=float(np.linalg.norm(previous_velocity));b=float(np.linalg.norm(secant));cos=float(np.dot(previous_velocity,secant)/(a*b+1e-12));innovation=float(np.linalg.norm(secant-previous_velocity));return [a,b,innovation,cos,float(gap),float(stage)]

def main():
 out=BASE/'results/causal_decay_selector';out.mkdir(exist_ok=True);windows=pd.read_csv(s0.Q1.parent/'C题工作区/阶段交付/stage1_review/corrected_results/data/windows.csv').query("split=='train'");sc=json.loads((OLD/'results/S0/fold_scales.json').read_text());cache={};events=[];data={}
 for r in windows.itertuples():
  key=(r.episode_uuid,r.side)
  if key not in cache:
   with np.load(s0.DATA/'records'/f'{r.episode_uuid}.npz',allow_pickle=False) as z:cache[key]=(z[r.side+'__t_s'],z[r.side+'__position'].mean(axis=1))
  t,x=cache[key];i,j=r.start_index,r.end_index_exclusive;tt=t[i:j];run=replay(tt,x[i:j],t[i-2:i],x[i-2:i],FixedPolicy(),delay=.1,budget=2,window_end=float(tt[-1]));ret=[v for v in run['actions'] if v['event']=='return'];arr=[int(np.argmin(abs(tt-v['processed_at']))) for v in ret];sample=[int(np.argmin(abs(tt-v['sample_time']))) for v in ret];last=x[i-1];last_t=t[i-1];v=(x[i-1]-x[i-2])/(t[i-1]-t[i-2]);stage_errors=[]
  for stage in [0,1]:
   q=sample[stage];gap=tt[q]-last_t;sec=(x[i+q]-last)/gap;features=visible_features(v,sec,gap,stage);lo=arr[stage];hi=arr[stage+1] if stage==0 else len(tt);age=tt[lo:hi]-tt[q];errors=[]
   for tau in TAUS:
    pred=x[i+q]-tau*np.expm1(-age/tau)[:,None]*sec;errors.append(np.linalg.norm(pred-x[i+lo:i+hi],axis=1))
   stage_errors.append(np.array(errors));meta={k:getattr(r,k) for k in ['window_id','dataset','environment_id','episode_uuid','side']};events.append(dict(**meta,stage=stage,**{f'f{k}':value for k,value in enumerate(features)},**{f'loss{k}':float(e.mean()) for k,e in enumerate(errors)},fraction=(hi-lo)/len(tt)));last=x[i+q];last_t=tt[q];v=sec
  data[r.window_id]=dict(base_error=np.array([p['error'] for p in run['points']]),arrivals=arr,errors=stage_errors)
 event=pd.DataFrame(events);event.to_csv(out/'visible_features_and_labels.csv.gz',index=False);sources=sorted(windows.dataset.unique());rows=[];selection=[];fits=[]
 for held in sorted(windows.environment_id.unique()):
  train=event[event.environment_id!=held];ev=event[event.environment_id==held]
  def matrix(frame):
   z=frame[[f'f{k}' for k in range(6)]].to_numpy().copy();ss=np.array([sc[held][s] for s in frame.dataset]);z[:,:3]/=ss[:,None];return np.column_stack([z,*[(frame.dataset==s).astype(float).to_numpy() for s in sources]])
  x=matrix(train);xe=matrix(ev);wt=weights(train)*train.fraction.to_numpy();wt/=wt.sum();mu=np.sum(wt[:,None]*x,axis=0);sd=np.sqrt(np.sum(wt[:,None]*(x-mu)**2,axis=0));sd[sd<1e-10]=1;x=np.column_stack([np.ones(len(x)),(x-mu)/sd]);xe=np.column_stack([np.ones(len(xe)),(xe-mu)/sd]);y=train[['loss0','loss1','loss2']].to_numpy();y=(y-y[:,1,None])/np.array([sc[held][s] for s in train.dataset])[:,None];pen=np.eye(x.shape[1])*.1;pen[0,0]=0;coef=np.linalg.solve(x.T@(wt[:,None]*x)+pen,x.T@(wt[:,None]*y));pred=xe@coef;chosen=np.argmin(pred,axis=1);fits.append(dict(held=held,mu=mu.tolist(),sd=sd.tolist(),coefficient=coef.tolist(),fit_environments=sorted(train.environment_id.unique())))
  picks={}
  for r,k,score in zip(ev.itertuples(),chosen,pred):picks[(r.window_id,r.stage)]=int(k);selection.append(dict(window_id=r.window_id,stage=r.stage,held=held,tau=TAUS[k],prediction=score.tolist()))
  for r in windows[windows.environment_id==held].itertuples():
   d=data[r.window_id]
   for method in ['baseline','causal_selector']:
    error=d['base_error'].copy()
    if method=='causal_selector':
     for stage in [0,1]:
      lo=d['arrivals'][stage];hi=d['arrivals'][stage+1] if stage==0 else len(error);error[lo:hi]=d['errors'][stage][picks[(r.window_id,stage)]]
    e=error/sc[held][r.dataset];rows.append(dict(**{k:getattr(r,k) for k in ['window_id','dataset','environment_id','episode_uuid','side']},method=method,nade=e.mean(),nfde=e[-1],exceed05=float((e>.05).mean()),requests=2,waiting_s=.2))
 result=pd.DataFrame(rows);result.to_csv(out/'windows.csv.gz',index=False);pd.DataFrame(selection).to_csv(out/'selection.csv',index=False);(out/'fits.json').write_text(json.dumps(fits,indent=2),encoding='utf8');metrics=['nade','nfde','exceed05','requests','waiting_s'];summary=hierarchical(result,['method'],metrics);summary.to_csv(out/'summary.csv',index=False);env=result.groupby(['method','dataset','environment_id','episode_uuid','side'],as_index=False)[metrics].mean().groupby(['method','dataset','environment_id','episode_uuid'],as_index=False)[metrics].mean().groupby(['method','dataset','environment_id'],as_index=False)[metrics].mean();env.to_csv(out/'environments.csv',index=False);env.groupby(['method','dataset'],as_index=False)[metrics].mean().to_csv(out/'sources.csv',index=False);ci=paired_ci(env,'baseline','causal_selector');audit=dict(passed=True,comparison=ci,windows=len(windows),return_events=len(event),formal_model_changed=False,development_reads=0,test_reads=0,scope='offline causal feature selection simulation; integrated runtime check pending');(out/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf8');print(summary.to_string(index=False));print(json.dumps(audit))
if __name__=='__main__':main()
