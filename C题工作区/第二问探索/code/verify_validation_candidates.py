import json,hashlib
import numpy as np,pandas as pd
from request_value import BASE,OLD

def main():
 out=BASE/'results/validation_candidates';m=json.loads((out/'full_train_models.json').read_text());audit=json.loads((out/'audit.json').read_text());assert hashlib.sha256((out/'full_train_models.json').read_bytes()).hexdigest()==audit['model_hash_before_validation'];points=pd.read_csv(out/'windows.csv.gz');assert len(points)==1060 and not points.duplicated(['method','window_id']).any();assert points.groupby('method').window_id.nunique().eq(212).all();assert np.max(abs(points.utility-points.nade-.02*points.requests))<1e-12
 metrics=['nade','nfde','exceed05','requests','waiting_s','utility'];g=points.copy()
 for levels in [['dataset','environment_id','episode_uuid','side'],['dataset','environment_id','episode_uuid'],['dataset','environment_id'],['dataset'],[]]:g=g.groupby(['method']+levels,as_index=False)[metrics].mean()
 saved=pd.read_csv(out/'summary.csv').set_index('method');delta=float(abs(g.set_index('method')[metrics]-saved[metrics]).to_numpy().max());assert delta<1e-12
 events=pd.read_csv(BASE/'results/causal_decay_selector/visible_features_and_labels.csv.gz');cross=pd.read_csv(out/'crossfit_training_events.csv.gz');maximum=0.
 for label,frame,n in [('selector',events,6),('crossfit',cross,6),('purchase',events[events.stage==0],5)]:
  keys=['dataset','environment_id','episode_uuid','side'];weight=np.ones(len(frame))/frame.dataset.nunique()
  for k in range(1,len(keys)):weight/=frame.groupby(keys[:k])[keys[k]].transform('nunique').to_numpy()
  weight/=frame.groupby(keys).window_id.transform('size').to_numpy()
  if label!='purchase':weight*=frame.fraction.to_numpy()
  weight/=weight.sum();ss=np.array([m['scale'][s] for s in frame.dataset]);x=frame[[f'f{k}' for k in range(n)]].to_numpy().copy();x[:,:3]/=ss[:,None];x=np.column_stack([x,*[(frame.dataset==s).astype(float).to_numpy() for s in m['sources']]]);mu=np.sum(weight[:,None]*x,axis=0);sd=np.sqrt(np.sum(weight[:,None]*(x-mu)**2,axis=0));sd[sd<1e-10]=1;assert max(abs(mu-m[label]['mu']).max(),abs(sd-m[label]['sd']).max())<1e-10;x=np.column_stack([np.ones(len(x)),(x-mu)/sd])
  if label=='purchase':
   raw=pd.read_csv(BASE/'results/second_request_value/candidate_windows.csv.gz').pivot(index='window_id',columns='method',values='raw_ade');y=np.array([(raw.loc[r.window_id,'first_only']-raw.loc[r.window_id,'fixed2'])/m['scale'][r.dataset] for r in frame.itertuples()]);target=np.r_[np.sqrt(weight)*y,np.zeros(x.shape[1])]
  else:
   y=frame[['loss0','loss1','loss2']].to_numpy();y=(y-y[:,1,None])/ss[:,None];target=np.vstack([np.sqrt(weight)[:,None]*y,np.zeros((x.shape[1],3))])
  reg=np.eye(x.shape[1])*np.sqrt(.1);reg[0,0]=0;coef=np.linalg.lstsq(np.vstack([np.sqrt(weight)[:,None]*x,reg]),target,rcond=None)[0];maximum=max(maximum,float(abs(coef-np.array(m[label]['coefficient'])).max()))
 assert maximum<1e-10
 result=dict(passed=True,validation_rows=len(points),summary_max_difference=delta,independent_ridge_coefficient_max_difference=maximum,model_hash_verified=True,scope='saved training events independent ridge and validation aggregation; not independent raw trajectory replay or base IRLS refit');(out/'independent_audit.json').write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps(result))
if __name__=='__main__':main()
