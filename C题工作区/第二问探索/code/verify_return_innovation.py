from pathlib import Path
import json
import numpy as np,pandas as pd
from return_innovation import returned_features,weights
BASE=Path(__file__).resolve().parents[1];p=BASE/'results/return_innovation'
h=pd.read_csv(p/'history_features.csv');r=pd.read_csv(p/'candidate_windows.csv.gz');a=pd.read_csv(p/'selected_windows.csv.gz');pred=pd.read_csv(p/'oof_predictions.csv')
assert len(h)==1527 and len(r)==4581 and len(a)==6108
assert (h.sample_index<h.decision_index).all()
assert (h.arrival_relative<=h.decision_relative+1e-9).all()
assert (h.decision_relative<.5-1e-9).all()
assert not a.duplicated(['method','window_id']).any()
assert r.groupby('window_id').requests.apply(lambda v:(v==2).all()).all()
# Pure feature function receives two initial observations and one returned payload only.
ht=np.array([0.,.03]);hx=np.array([[0.,0.,0.],[.03,0.,0.]])
f=returned_features(ht,hx,.35,np.array([.4,0.,0.]),.45)
# Independent analytic check of sample-time innovation, not arrival-time innovation.
v=np.array([1.,0.,0.]);forecast=hx[-1]+.2*(1-np.exp(-(.35-.03)/.2))*v
assert abs(f[2]-np.linalg.norm(np.array([.4,0.,0.])-forecast))<1e-12
assert abs(f[6]-.1)<1e-12
sc=json.loads((BASE.parent/'第二问求解/results/S0/fold_scales.json').read_text());mx=0.;folds=[]
for held in sorted(h.environment_id.unique()):
 fit=h[h.environment_id!=held];ev=h[h.environment_id==held];w=weights(fit)
 assert held not in set(fit.environment_id) and abs(w.sum()-1)<1e-12
 for src in fit.dataset.unique():assert abs(w[(fit.dataset==src).to_numpy()].sum()-1/fit.dataset.nunique())<1e-12
 def matrix(frame):
  z=frame[[f'f{k}' for k in range(7)]].to_numpy().copy();z[:,:4]/=np.array([sc[held][s] for s in frame.dataset])[:,None]
  return np.column_stack([z,*[(frame.dataset==s).astype(float) for s in sorted(h.dataset.unique())]])
 xx=matrix(fit);xe=matrix(ev);mu=np.average(xx,axis=0,weights=w);sd=np.sqrt(np.average((xx-mu)**2,axis=0,weights=w));sd[sd<1e-10]=1
 xx=np.column_stack([np.ones(len(xx)),(xx-mu)/sd]);xe=np.column_stack([np.ones(len(xe)),(xe-mu)/sd])
 pivot=r.pivot(index='window_id',columns='method',values='raw_ade').loc[fit.window_id]
 y=np.column_stack([(pivot.fixed-pivot[c]).to_numpy() for c in ['early','late']])/np.array([sc[held][s] for s in fit.dataset])[:,None]
 # Solve independently as augmented least squares, instead of normal equations.
 penalty=np.eye(xx.shape[1])*np.sqrt(.1);penalty[0,0]=0
 coef=np.linalg.lstsq(np.vstack([xx*np.sqrt(w[:,None]),penalty]),np.vstack([y*np.sqrt(w[:,None]),np.zeros((xx.shape[1],2))]),rcond=None)[0]
 actual=xe@coef;expected=pred[(pred.held==held)&(pred.method=='innovation_ridge')].set_index('window_id').loc[ev.window_id,['early_gain','late_gain']].to_numpy()
 delta=float(np.abs(actual-expected).max());mx=max(mx,delta);assert delta<1e-10
 folds.append(dict(held=held,fit_windows=len(fit),held_windows=len(ev),max_prediction_difference=delta))
checks=dict(passed=True,folds=len(folds),independent_augmented_least_squares_max_difference=mx,history_rows=len(h),decision_before_all_second_requests=True,arrival_before_decision=True,feature_api_receives_only_initial_pair_and_return=True,held_env_excluded=True,source_weight_balance_verified=True,scope='feature causality and independently solved fold predictions; not second raw replay engine')
pd.DataFrame(folds).to_csv(p/'fold_audit.csv',index=False);(p/'independent_audit.json').write_text(json.dumps(checks,indent=2),encoding='utf8');print(json.dumps(checks))
