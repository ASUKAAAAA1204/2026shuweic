import json,hashlib
import numpy as np,pandas as pd
from distance_aligned_return import fit_coeff,BASE,OLD
from request_value import weights
from update_calibration_experiment import hierarchical
from candidate_replay_engine import replay,FixedPolicy
from candidate_replay_integration import DistanceState
from online_decay_selector import OnlineDecaySelector
from online_corrected_selector import OnlineCorrectedSelector
from online_second_purchase import SecondPurchasePolicy
import s0_audit_replay as s0

def main():
 out=BASE/'results/validation_candidates';out.mkdir(exist_ok=True)
 freeze=json.loads((BASE/'results/validation_freeze/manifest.json').read_text())
 for item in freeze['candidates']:assert hashlib.sha256((BASE/item['path']).read_bytes()).hexdigest()==item['sha256'],item['path']
 windows=pd.read_csv(s0.Q1.parent/'C题工作区/阶段交付/stage1_review/corrected_results/data/windows.csv');train=windows[windows.split=='train'];val=windows[windows.split=='validation'];assert not set(train.environment_id)&set(val.environment_id);scale=json.loads((OLD/'results/S0/fold_scales.json').read_text())['development'];sources=sorted(train.dataset.unique());cache={};data={};reads={'train':set(),'validation':set()}
 def load(r):
  key=(r.episode_uuid,r.side)
  if key not in cache:
   with np.load(s0.DATA/'records'/f'{r.episode_uuid}.npz',allow_pickle=False) as z:cache[key]=(z[r.side+'__t_s'],z[r.side+'__position'].mean(axis=1))
  reads[r.split].add(r.episode_uuid);t,x=cache[key];i,j=r.start_index,r.end_index_exclusive;return t[i:j],x[i:j],t[i-2:i],x[i-2:i]
 for r in train.itertuples():
  tt,x,ht,h=load(r);run=replay(tt,x,ht,h,FixedPolicy(),delay=.1,budget=2,window_end=float(tt[-1]));ret=[v for v in run['actions'] if v['event']=='return'];data[r.window_id]=dict(t=tt,x=x,ht=ht,hist=h,samples=[int(np.argmin(abs(tt-v['sample_time']))) for v in ret],arrivals=[int(np.argmin(abs(tt-v['processed_at']))) for v in ret])
 c1=fit_coeff(data,train,scale,0,np.array([1.,0.]));c2=fit_coeff(data,train,scale,1,c1)
 def fit_event(frame,purchase=False):
  n=5 if purchase else 6;x=frame[[f'f{k}' for k in range(n)]].to_numpy().copy();ss=np.array([scale[s] for s in frame.dataset]);x[:,:3]/=ss[:,None];x=np.column_stack([x,*[(frame.dataset==s).astype(float).to_numpy() for s in sources]]);wt=weights(frame)
  if not purchase:wt*=frame.fraction.to_numpy();wt/=wt.sum()
  mu=np.sum(wt[:,None]*x,axis=0);sd=np.sqrt(np.sum(wt[:,None]*(x-mu)**2,axis=0));sd[sd<1e-10]=1;x=np.column_stack([np.ones(len(x)),(x-mu)/sd]);pen=np.eye(x.shape[1])*.1;pen[0,0]=0
  if purchase:
   raw=pd.read_csv(BASE/'results/second_request_value/candidate_windows.csv.gz').pivot(index='window_id',columns='method',values='raw_ade');y=np.array([(raw.loc[r.window_id,'first_only']-raw.loc[r.window_id,'fixed2'])/scale[r.dataset] for r in frame.itertuples()])
  else:
   y=frame[['loss0','loss1','loss2']].to_numpy();y=(y-y[:,1,None])/ss[:,None]
  weighted_y=wt*y if purchase else wt[:,None]*y
  coef=np.linalg.solve(x.T@(wt[:,None]*x)+pen,x.T@weighted_y);assert coef.shape==((x.shape[1],) if purchase else (x.shape[1],3));return dict(mu=mu.tolist(),sd=sd.tolist(),coefficient=coef.tolist())
 events=pd.read_csv(BASE/'results/causal_decay_selector/visible_features_and_labels.csv.gz');assert set(events.window_id)==set(train.window_id);selector=fit_event(events);purchase=fit_event(events[events.stage==0],True);frames=[]
 folds={f['held']:f for f in json.loads((BASE/'results/distance_aligned_return/folds.json').read_text())}
 for env in sorted(train.environment_id.unique()):
  assert env not in folds[env]['fit_environments'];z=pd.read_csv(BASE/'results/corrected_velocity_decay_selector/events'/f'{env}.csv.gz');z=z[z.environment_id==env];assert set(z.window_id)==set(train[train.environment_id==env].window_id);frames.append(z)
 cross_events=pd.concat(frames,ignore_index=True);cross_events.to_csv(out/'crossfit_training_events.csv.gz',index=False);cross=fit_event(cross_events);cross.update(first=c1.tolist(),second=c2.tolist())
 models=dict(scale=scale,sources=sources,first=c1.tolist(),second=c2.tolist(),selector=selector,purchase=purchase,crossfit=cross,training_windows=len(train),training_environments=sorted(train.environment_id.unique()))
 modelpath=out/'full_train_models.json';modelpath.write_text(json.dumps(models,indent=2),encoding='utf8');modelhash=hashlib.sha256(modelpath.read_bytes()).hexdigest();assert not reads['validation'];print('Full training models frozen',modelhash,flush=True);rows=[]
 for r in val.itertuples():
  tt,x,ht,h=load(r)
  for method in ['baseline','distance_return','causal_selector','crossfit_corrected','adaptive_buy']:
   factory=None;policy=FixedPolicy();kwargs=dict(fit=selector,scale=scale[r.dataset],source=r.dataset,sources=sources)
   if method=='distance_return':factory=lambda *a:DistanceState(*a,coefficients=[c1,c2])
   elif method=='causal_selector':factory=lambda *a:OnlineDecaySelector(*a,**kwargs)
   elif method=='crossfit_corrected':factory=lambda *a:OnlineCorrectedSelector(*a,**dict(kwargs,fit=cross))
   elif method=='adaptive_buy':policy=SecondPurchasePolicy(initial_velocity=(h[-1]-h[-2])/(ht[-1]-ht[-2]),initial_time=ht[-1],fit=purchase,scale=scale[r.dataset],source=r.dataset,sources=sources)
   run=replay(tt,x,ht,h,policy,delay=.1,budget=2,window_end=float(tt[-1]),state_factory=factory);e=np.array([v['error'] for v in run['points']])/scale[r.dataset];rows.append(dict(**{k:getattr(r,k) for k in ['window_id','dataset','environment_id','episode_uuid','side']},method=method,nade=e.mean(),nfde=e[-1],exceed05=float((e>.05).mean()),requests=run['metrics']['requests'],waiting_s=run['metrics']['waiting_s'],utility=e.mean()+.02*run['metrics']['requests']))
 assert hashlib.sha256(modelpath.read_bytes()).hexdigest()==modelhash
 result=pd.DataFrame(rows);result.to_csv(out/'windows.csv.gz',index=False);metrics=['nade','nfde','exceed05','requests','waiting_s','utility'];summary=hierarchical(result,['method'],metrics);summary.to_csv(out/'summary.csv',index=False);env=result.groupby(['method','dataset','environment_id','episode_uuid','side'],as_index=False)[metrics].mean().groupby(['method','dataset','environment_id','episode_uuid'],as_index=False)[metrics].mean().groupby(['method','dataset','environment_id'],as_index=False)[metrics].mean();env.to_csv(out/'environments.csv',index=False);env.groupby(['method','dataset'],as_index=False)[metrics].mean().to_csv(out/'sources.csv',index=False);audit=dict(validation_windows=len(val),validation_environments=val.environment_id.nunique(),train_record_reads=len(reads['train']),validation_record_reads=len(reads['validation']),test_reads=0,model_hash_before_validation=modelhash,models_unchanged_after_validation=True,formal_model_changed=False,in_sample_sensitivity_candidate_not_evaluated=True,uncertainty_ranges_not_evaluated=True,scope='previously viewed validation; not fresh independent test');(out/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf8');print(summary.to_string(index=False));print(json.dumps(audit))
if __name__=='__main__':main()
