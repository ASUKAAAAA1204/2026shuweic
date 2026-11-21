import json
import numpy as np,pandas as pd
from request_value import BASE,OLD,weights
from candidate_replay_engine import replay,FixedPolicy
from update_calibration_experiment import hierarchical,paired_ci
import s0_audit_replay as s0

def fit(data,frame,scale):
 xx=[];yy=[];ww=[]
 for weight,r in zip(weights(frame),frame.itertuples()):
  d=data[r.window_id];mask=d['before'];xx.append(d['displacement'][mask]/scale[r.dataset]);yy.append((d['truth'][mask]-d['last'])/scale[r.dataset]);ww.append(np.full(mask.sum(),weight/len(mask)))
 x=np.concatenate(xx);y=np.concatenate(yy);w=np.concatenate(ww);a=1.;history=[]
 def loss(a):return float(np.sum(w*np.sqrt(np.sum((a*x-y)**2,axis=1)+1e-8))+.0005*(a-1)**2)
 history.append(loss(a))
 for _ in range(20):
  v=w/np.sqrt(np.sum((a*x-y)**2,axis=1)+1e-8);a=max(0.,float((np.sum(v*np.sum(x*y,axis=1))+.001)/(np.sum(v*np.sum(x*x,axis=1))+.001)));history.append(loss(a));assert history[-1]<=history[-2]+1e-10
 return a,history

def main():
 out=BASE/'results/initial_multiplier';out.mkdir(exist_ok=True);w=pd.read_csv(s0.Q1.parent/'C题工作区/阶段交付/stage1_review/corrected_results/data/windows.csv').query("split=='train'");sc=json.loads((OLD/'results/S0/fold_scales.json').read_text());cache={};data={}
 for r in w.itertuples():
  key=(r.episode_uuid,r.side)
  if key not in cache:
   with np.load(s0.DATA/'records'/f'{r.episode_uuid}.npz',allow_pickle=False) as z:cache[key]=(z[r.side+'__t_s'],z[r.side+'__position'].mean(axis=1))
  t,x=cache[key];i,j=r.start_index,r.end_index_exclusive;run=replay(t[i:j],x[i:j],t[i-2:i],x[i-2:i],FixedPolicy(),delay=.1,budget=2,window_end=float(t[j-1]));before=np.array([p['returns']==0 for p in run['points']]);v=(x[i-1]-x[i-2])/(t[i-1]-t[i-2]);displacement=-.2*np.expm1(-(t[i:j]-t[i-1])/.2)[:,None]*v;data[r.window_id]=dict(before=before,displacement=displacement,last=x[i-1],truth=x[i:j],baseline=np.array([p['prediction'] for p in run['points']]))
 rows=[];fits=[];postdiff=0.
 for held in sorted(w.environment_id.unique()):
  train=w[w.environment_id!=held];a,h=fit(data,train,sc[held]);source={};histories={}
  for src,g in train.groupby('dataset'):source[src],histories[src]=fit(data,g,sc[held])
  fits.append(dict(held=held,fit_environments=sorted(train.environment_id.unique()),global_multiplier=a,source_multipliers=source,global_loss_history=h,source_loss_histories=histories))
  for r in w[w.environment_id==held].itertuples():
   d=data[r.window_id]
   for method,m in [('baseline',1.),('global_multiplier',a),('source_multiplier',source.get(r.dataset,1.))]:
    pred=d['baseline'].copy();pred[d['before']]=d['last']+m*d['displacement'][d['before']];postdiff=max(postdiff,float(abs(pred[~d['before']]-d['baseline'][~d['before']]).max()));e=np.linalg.norm(pred-d['truth'],axis=1)/sc[held][r.dataset]
    for scope,mask in [('all',np.ones(len(e),bool)),('before_return',d['before'])]:
     z=e[mask];rows.append(dict(**{k:getattr(r,k) for k in ['window_id','dataset','environment_id','episode_uuid','side']},method=method,scope=scope,multiplier=m,nade=z.mean(),nfde=z[-1],exceed05=(z>.05).mean(),requests=2.,waiting_s=.2))
 result=pd.DataFrame(rows);result.to_csv(out/'windows.csv.gz',index=False);(out/'fits.json').write_text(json.dumps(fits,indent=2),encoding='utf8');metrics=['nade','nfde','exceed05','requests','waiting_s'];summary=hierarchical(result,['method','scope'],metrics);summary.to_csv(out/'summary.csv',index=False);env=result.groupby(['method','scope','dataset','environment_id','episode_uuid','side'],as_index=False)[metrics].mean().groupby(['method','scope','dataset','environment_id','episode_uuid'],as_index=False)[metrics].mean().groupby(['method','scope','dataset','environment_id'],as_index=False)[metrics].mean();env.to_csv(out/'environments.csv',index=False);env.groupby(['method','scope','dataset'],as_index=False)[metrics].mean().to_csv(out/'sources.csv',index=False)
 old=pd.read_csv(BASE/'results/initial_history_candidates/windows.csv.gz').query("method=='baseline' and scope=='all'").set_index('window_id');base=result.query("method=='baseline' and scope=='all'").set_index('window_id');difference=float(abs(old.nade-base.nade).max());assert difference<1e-12 and postdiff==0
 audit=dict(windows=len(w),baseline_max_difference=difference,post_return_difference=postdiff,validation_reads=0,test_reads=0,formal_model_changed=False,comparisons={m:paired_ci(env[env.scope=='all'],'baseline',m) for m in ['global_multiplier','source_multiplier']});(out/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf8');print(summary.to_string(index=False));print(json.dumps(audit))
if __name__=='__main__':main()
