from pathlib import Path
import sys,json
import numpy as np,pandas as pd
from request_value import weights,BASE,OLD
sys.path.insert(0,str(OLD/'code'))
from engine import FixedPolicy
from end_guard import safe_replay
from update_calibration_experiment import hierarchical,paired_ci
import s0_audit_replay as s0

FIT_SYSTEMS=[]

def make_stage(d,stage,c1):
 t,x=d['t'],d['x'];q=d['samples'];a=d['arrivals'];hist=d['hist'];ht=d['ht']
 v0=(hist[-1]-hist[-2])/(ht[-1]-ht[-2]);gap1=t[q[0]]-ht[-1];sec1=(x[q[0]]-hist[-1])/gap1;old1=v0*np.exp(-gap1/.2)
 if stage==0:sec=sec1;old=old1;lo=a[0];hi=a[1];last=x[q[0]];sample=t[q[0]]
 else:
  v1=c1[0]*sec1+c1[1]*old1;gap=t[q[1]]-t[q[0]];sec=(x[q[1]]-x[q[0]])/gap;old=v1*np.exp(-gap/.2);lo=a[1];hi=len(t);last=x[q[1]];sample=t[q[1]]
 factor=-.2*np.expm1(-(t[lo:hi]-sample)/.2)
 X=factor[:,None,None]*np.stack([sec,old],axis=-1)[None,:,:]
 return lo,hi,last,X,x[lo:hi]-last

def fit_coeff(data,frame,scale,stage,c1):
 w=weights(frame);XX=[];YY=[];WW=[]
 for weight,r in zip(w,frame.itertuples()):
  d=data[r.window_id];_,_,_,X,y=make_stage(d,stage,c1);ss=scale[r.dataset]
  XX.append(X/ss);YY.append(y/ss);WW.append(np.full(len(y),weight/len(d['t'])))
 X=np.concatenate(XX);y=np.concatenate(YY);w=np.concatenate(WW);prior=np.array([1.,0.]);coefficient=prior.copy();history=[]
 def objective(c):
  residual=X@c-y
  return float(np.sum(w*np.sqrt((residual*residual).sum(axis=1)+1e-8))+.0005*np.sum((c-prior)**2))
 history.append(objective(coefficient))
 for iteration in range(20):
  residual=X@coefficient-y;ww=w/np.sqrt((residual*residual).sum(axis=1)+1e-8)
  gram=np.einsum('nki,n,nkj->ij',X,ww,X);rhs=np.einsum('nki,n,nk->i',X,ww,y)
  coefficient=np.linalg.solve(gram+.001*np.eye(2),rhs+.001*prior)
  value=objective(coefficient);assert value<=history[-1]+1e-10;history.append(value)
 FIT_SYSTEMS.append(dict(stage=stage,gram=gram.tolist(),rhs=rhs.tolist(),coefficient=coefficient.tolist(),objective_history=history))
 return coefficient

def predictions(d,c1,c2):
 ht,h=d['ht'],d['hist'];v=(h[-1]-h[-2])/(ht[-1]-ht[-2]);factor=-.2*np.expm1(-(d['t']-ht[-1])/.2);pred=h[-1]+factor[:,None]*v
 for stage,c in [(0,c1),(1,c2)]:
  lo,hi,last,X,_=make_stage(d,stage,c1);pred[lo:hi]=last+X@c
 return pred

def main():
 out=BASE/'results/distance_aligned_return';out.mkdir(exist_ok=True)
 frame=pd.read_csv(s0.Q1.parent/'C题工作区/阶段交付/stage1_review/corrected_results/data/windows.csv').query("split=='train'");sc=json.loads((OLD/'results/S0/fold_scales.json').read_text());cache={};data={};maxdiff=0.
 for w in frame.itertuples():
  key=(w.episode_uuid,w.side)
  if key not in cache:
   with np.load(s0.DATA/'records'/f'{w.episode_uuid}.npz',allow_pickle=False) as z:cache[key]=(z[w.side+'__t_s'],z[w.side+'__position'].mean(axis=1))
  t,x=cache[key];i,j=w.start_index,w.end_index_exclusive;r=safe_replay(t[i:j],x[i:j],t[i-2:i],x[i-2:i],FixedPolicy(),delay=.1,budget=2,window_end=float(t[i]+1));ret=[a for a in r['actions'] if a['event']=='return'];assert len(ret)==2
  tt=t[i:j];d=dict(t=tt,x=x[i:j],ht=t[i-2:i],hist=x[i-2:i],samples=[int(np.argmin(abs(tt-a['sample_time']))) for a in ret],arrivals=[int(np.argmin(abs(tt-a['processed_at']))) for a in ret]);data[w.window_id]=d
  delta=float(np.max(abs(predictions(d,[1.,0.],[1.,0.])-np.array([p['prediction'] for p in r['points']]))));maxdiff=max(maxdiff,delta);assert delta<1e-12
 rows=[];folds=[]
 for held in sorted(frame.environment_id.unique()):
  fit=frame[frame.environment_id!=held];ev=frame[frame.environment_id==held];c1=fit_coeff(data,fit,sc[held],0,np.array([1.,0.]));c2=fit_coeff(data,fit,sc[held],1,c1);folds.append(dict(held=held,first_a=c1[0],first_b=c1[1],second_a=c2[0],second_b=c2[1],fit_environments=sorted(fit.environment_id.unique())))
  for w in ev.itertuples():
   d=data[w.window_id];scale=sc[held][w.dataset]
   for method,a,b in [('secant',[1.,0.],[1.,0.]),('distance_return',c1,c2)]:
    pred=predictions(d,a,b);e=np.linalg.norm(pred-d['x'],axis=1)/scale
    rows.append(dict(**{k:getattr(w,k) for k in ['window_id','dataset','environment_id','episode_uuid','side']},method=method,nade=e.mean(),nfde=e[-1],exceed05=float((e>.05).mean()),requests=2,waiting_s=.2))
 (out/'fit_systems.json').write_text(json.dumps(FIT_SYSTEMS,indent=2),encoding='utf8')
 m=pd.DataFrame(rows);m.to_csv(out/'windows.csv.gz',index=False);(out/'folds.json').write_text(json.dumps(folds,indent=2),encoding='utf8');metrics=['nade','nfde','exceed05','requests','waiting_s'];summary=hierarchical(m,['method'],metrics);summary.to_csv(out/'summary.csv',index=False)
 env=m.groupby(['method','dataset','environment_id','episode_uuid','side'],as_index=False)[metrics].mean().groupby(['method','dataset','environment_id','episode_uuid'],as_index=False)[metrics].mean().groupby(['method','dataset','environment_id'],as_index=False)[metrics].mean();env.to_csv(out/'environments.csv',index=False);src=env.groupby(['method','dataset'],as_index=False)[metrics].mean();src.to_csv(out/'sources.csv',index=False);ci=paired_ci(env,'secant','distance_return');sg=src.pivot(index='dataset',columns='method',values='nade');gains=100*(1-sg.distance_return/sg.secant)
 audit=dict(passed=True,reconstructed_baseline_max_difference=maxdiff,train_windows=len(frame),held_environments=len(folds),development_reads=0,test_reads=0,admitted=bool(ci['gain_pct']>=3 and ci['lo95']>0 and gains.min()>=-5),comparison=ci,source_gains=gains.to_dict(),regularization=.001,fit_loss='smoothed Euclidean',iterations=20)
 (out/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf8');print(summary.to_string(index=False));print(json.dumps(audit))
if __name__=='__main__':main()

