"""Fixed family versus prefix duration evidence. No validation-driven model search."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
BASE=Path(__file__).resolve().parents[1];DATA=BASE.parent/'处理结果/v1'
def features(z):
 result=[]
 for side in ['left_hand','right_hand']:
  t=z[side+'__t_s'];x=z[side+'__position'].mean(axis=1);good=z[side+'__quality_known']&~z[side+'__bad']&np.isfinite(x).all(axis=1);pair=good[1:]&good[:-1]&(np.diff(t)>0);steps=np.linalg.norm(np.diff(x,axis=0),axis=1);speed=steps[pair]/np.diff(t)[pair];ids=np.flatnonzero(good)
  result.extend([len(t),float(good.mean()) if len(t) else 0.,float(np.linalg.norm(x[ids[-1]]-x[ids[0]])) if len(ids)>1 else 0.,float(steps[pair].sum()),float(np.median(speed)) if len(speed) else 0.])
 return np.array(result)
def fit(x,y):
 mu=x.mean(0);sd=x.std(0);sd[sd<1e-10]=1;z=np.c_[np.ones(len(x)),(x-mu)/sd];reg=np.eye(z.shape[1])*100.;reg[0,0]=0;coef=np.linalg.solve(z.T@z+reg,z.T@np.log(y));return mu,sd,coef
def predict(m,x):return np.exp(np.clip(np.c_[np.ones(len(x)),(x-m[0])/m[1]]@m[2],-10,10))
def main():
 out=BASE/'results/duration_information';out.mkdir(parents=True,exist_ok=True);s=pd.read_csv(DATA/'segments_private.csv');s=s[(s.eligible==1)&(s.split=='train')];rows=[];payload={}
 for r in s.itertuples():
  for length in [.2,.6,1.]:
   with np.load(DATA/'private'/(r.unit_id+'_'+str(length)+'.npz'),allow_pickle=False) as z:f=features(z)
   payload[(r.unit_id,length)]=np.r_[f,[float(r.family_id==v) for v in ['P','A','L']]]
 for held in sorted(s.environment_id.unique()):
  tr=s[s.environment_id!=held];ev=s[s.environment_id==held]
  for length in [.2,.6,1.]:
   x=np.stack([payload[(r.unit_id,length)] for r in tr.itertuples()]);y=tr.duration_s.to_numpy();m=fit(x,y);pred=predict(m,np.stack([payload[(r.unit_id,length)] for r in ev.itertuples()]))
   # Nested fold residuals for prefix interval calibration: no held-environment fitting.
   inner_res=[]
   for inner in sorted(tr.environment_id.unique()):
    mask=tr.environment_id.to_numpy()!=inner;im=fit(x[mask],y[mask]);inner_res.extend((y[~mask]-predict(im,x[~mask])).tolist())
   for r,p in zip(ev.itertuples(),pred):
    pool=tr[tr.family_id==r.family_id].duration_s.to_numpy();assert len(pool)>0
    for method in ['family','prefix_ridge']:
     quant=np.quantile(pool,[.1,.5,.9]) if method=='family' else p+np.quantile(inner_res,[.1,.5,.9]);quant=np.maximum(.001,quant);actual=r.duration_s
     row=dict(unit_id=r.unit_id,environment_id=held,dataset=r.dataset,episode_uuid=r.episode_uuid,family=r.family_id,length=length,method=method,duration=actual,prediction=float(quant[1]),mae=float(abs(quant[1]-actual)),coverage80=float(quant[0]<=actual<=quant[2]),width80=float(quant[2]-quant[0]),pinball90=float(max(.9*(actual-quant[2]),-.1*(actual-quant[2]))));rows.append(row)
  print('Duration fold',held,flush=True)
 d=pd.DataFrame(rows);d.to_csv(out/'oof_units.csv',index=False);metrics=['mae','coverage80','width80','pinball90'];env=d.groupby(['method','length','environment_id'],as_index=False)[metrics].mean();env.to_csv(out/'environments.csv',index=False);summary=env.groupby(['method','length'],as_index=False)[metrics].mean();summary.to_csv(out/'summary.csv',index=False)
 d.groupby(['method','length','family','environment_id'],as_index=False)[metrics].mean().groupby(['method','length','family'],as_index=False)[metrics].mean().to_csv(out/'families.csv',index=False)
 p=env[env.method=='prefix_ridge'].pivot(index='environment_id',columns='length',values='mae');gain=p[.2]-p[1.];base=summary[(summary.method=='family')&(summary.length==.2)].mae.iloc[0];late=summary[(summary.method=='prefix_ridge')&(summary.length==1.)].mae.iloc[0]
 decision=bool(gain.mean()>0 and (gain>0).sum()>len(gain)/2 and late<base)
 (out/'decision.json').write_text(json.dumps(dict(training_units=len(s),environments=s.environment_id.nunique(),ridge=100,hyperparameter_search=False,nested_residual_calibration=True,validation_reads=0,test_reads=0,late_minus_early_mae=float(-gain.mean()),environments_improved=int((gain>0).sum()),review_research_enabled=decision,decision_scope='necessary information screen, not proof of decision value; if false skip new review policy'),indent=2),encoding='utf8');print(summary.to_string(index=False));print('Review screen',decision)
if __name__=='__main__':main()
