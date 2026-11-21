import json
import numpy as np,pandas as pd
from request_value import BASE,OLD,weights
from candidate_replay_engine import replay,FixedPolicy
from update_calibration_experiment import hierarchical,paired_ci
import s0_audit_replay as s0
LAMBDA=.02

def main():
 out=BASE/'results/second_request_value';out.mkdir(exist_ok=True);w=pd.read_csv(s0.Q1.parent/'C题工作区/阶段交付/stage1_review/corrected_results/data/windows.csv').query("split=='train'");sc=json.loads((OLD/'results/S0/fold_scales.json').read_text());features=pd.read_csv(BASE/'results/causal_decay_selector/visible_features_and_labels.csv.gz').query('stage==0');cache={};rows=[]
 for r in w.itertuples():
  key=(r.episode_uuid,r.side)
  if key not in cache:
   with np.load(s0.DATA/'records'/f'{r.episode_uuid}.npz',allow_pickle=False) as z:cache[key]=(z[r.side+'__t_s'],z[r.side+'__position'].mean(axis=1))
  t,x=cache[key];i,j=r.start_index,r.end_index_exclusive
  for method,schedule in [('fixed2',(.3,.6)),('first_only',(.3,))]:
   run=replay(t[i:j],x[i:j],t[i-2:i],x[i-2:i],FixedPolicy(schedule),delay=.1,budget=2,window_end=float(t[j-1]));raw=np.array([p['error'] for p in run['points']]);e=raw/sc[r.environment_id][r.dataset];rows.append(dict(**{k:getattr(r,k) for k in ['window_id','dataset','environment_id','episode_uuid','side']},method=method,raw_ade=raw.mean(),nade=e.mean(),nfde=e[-1],exceed05=float((e>.05).mean()),requests=run['metrics']['requests'],waiting_s=run['metrics']['waiting_s']))
 raw=pd.DataFrame(rows);raw.to_csv(out/'candidate_windows.csv.gz',index=False);pivot=raw.pivot(index='window_id',columns='method',values='raw_ade');sources=sorted(w.dataset.unique());selected=[];picks=[];fits=[]
 for held in sorted(w.environment_id.unique()):
  train=features[features.environment_id!=held];ev=features[features.environment_id==held];wt=weights(train)
  def matrix(frame):
   x=frame[[f'f{k}' for k in range(5)]].to_numpy().copy();x[:,:3]/=np.array([sc[held][s] for s in frame.dataset])[:,None];return np.column_stack([x,*[(frame.dataset==s).astype(float).to_numpy() for s in sources]])
  x=matrix(train);xe=matrix(ev);mu=np.sum(wt[:,None]*x,axis=0);sd=np.sqrt(np.sum(wt[:,None]*(x-mu)**2,axis=0));sd[sd<1e-10]=1;x=np.column_stack([np.ones(len(x)),(x-mu)/sd]);xe=np.column_stack([np.ones(len(xe)),(xe-mu)/sd]);y=np.array([(pivot.loc[r.window_id,'first_only']-pivot.loc[r.window_id,'fixed2'])/sc[held][r.dataset] for r in train.itertuples()]);pen=np.eye(x.shape[1])*.1;pen[0,0]=0;coef=np.linalg.solve(x.T@(wt[:,None]*x)+pen,x.T@(wt*y));gain=xe@coef;fixed_choice='fixed2' if np.dot(wt,y)>LAMBDA else 'first_only'
  fits.append(dict(held=held,mu=mu.tolist(),sd=sd.tolist(),coefficient=coef.tolist(),fit_environments=sorted(train.environment_id.unique()),fixed_choice=fixed_choice))
  for r,value in zip(ev.itertuples(),gain):
   choice='fixed2' if value>LAMBDA else 'first_only';picks.append(dict(window_id=r.window_id,held=held,predicted_gain=value,choice=choice));
   for method,c in [('adaptive_buy',choice),('train_selected_fixed',fixed_choice)]:
    z=raw[(raw.window_id==r.window_id)&(raw.method==c)].copy();z['method']=method;selected.append(z)
 result=pd.concat([raw,*selected]);result['utility']=result.nade+LAMBDA*result.requests;result.to_csv(out/'windows.csv.gz',index=False);pd.DataFrame(picks).to_csv(out/'picks.csv',index=False);(out/'fits.json').write_text(json.dumps(fits,indent=2),encoding='utf8');metrics=['nade','nfde','exceed05','requests','waiting_s','utility'];summary=hierarchical(result,['method'],metrics);summary.to_csv(out/'summary.csv',index=False);env=result.groupby(['method','dataset','environment_id','episode_uuid','side'],as_index=False)[metrics].mean().groupby(['method','dataset','environment_id','episode_uuid'],as_index=False)[metrics].mean().groupby(['method','dataset','environment_id'],as_index=False)[metrics].mean();env.to_csv(out/'environments.csv',index=False);env.groupby(['method','dataset'],as_index=False)[metrics].mean().to_csv(out/'sources.csv',index=False);audit=dict(windows=len(w),lambda_cost=LAMBDA,comparison_utility=paired_ci(env,'train_selected_fixed','adaptive_buy',metric='utility'),raw_skipped_second=sum(r['choice']=='first_only' for r in picks),development_reads=0,test_reads=0,formal_model_changed=False,scope='offline causal-feature decision simulation; live policy integration pending');(out/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf8');print(summary.to_string(index=False));print(json.dumps(audit))
if __name__=='__main__':main()
