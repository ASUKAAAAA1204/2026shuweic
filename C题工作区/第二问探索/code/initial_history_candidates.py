import json
import numpy as np,pandas as pd
from request_value import BASE,OLD
from candidate_replay_engine import MotionState,replay,FixedPolicy
from update_calibration_experiment import hierarchical
import s0_audit_replay as s0

def main():
 out=BASE/'results/initial_history_candidates';out.mkdir(exist_ok=True)
 w=pd.read_csv(s0.Q1.parent/'C题工作区/阶段交付/stage1_review/corrected_results/data/windows.csv').query("split=='train'");sc=json.loads((OLD/'results/S0/fold_scales.json').read_text());cache={};rows=[];support=[];max_post=0.;max_fallback=0.
 for r in w.itertuples():
  key=(r.episode_uuid,r.side)
  if key not in cache:
   with np.load(s0.DATA/'records'/f'{r.episode_uuid}.npz',allow_pickle=False) as z:cache[key]=(z[r.side+'__t_s'],z[r.side+'__position'].mean(axis=1),z[r.side+'__reference_candidate'])
  t,x,good=cache[key];i,j=r.start_index,r.end_index_exclusive;assert i>=2 and good[i-2:i].all();a=max(0,i-6);ok=i-a==6 and bool(good[a:i].all()) and bool((np.diff(t[a:i])<=.1).all());base=(x[i-1]-x[i-2])/(t[i-1]-t[i-2]);v={'baseline':base,'zero':np.zeros(3),'endpoint6':base.copy(),'ols6':base.copy()}
  if ok:
   ht=t[a:i]-t[i-1];hx=x[a:i];v['endpoint6']=(hx[-1]-hx[0])/(ht[-1]-ht[0]);center=ht-ht.mean();v['ols6']=np.sum(center[:,None]*(hx-hx.mean(axis=0)),axis=0)/np.sum(center**2)
  runs={};support.append(dict(window_id=r.window_id,history6=ok,history_first=a if ok else i-2,history_last=i-1,start=i))
  for method,velocity in v.items():
   def factory(*args):
    state=MotionState(*args);state.velocity=velocity.copy();return state
   run=replay(t[i:j],x[i:j],t[i-2:i],x[i-2:i],FixedPolicy(),state_factory=factory,delay=.1,budget=2,window_end=float(t[j-1]));runs[method]=run;p=pd.DataFrame(run['points']);p['nade']=p.error/sc[r.environment_id][r.dataset]
   for scope,f in [('all',p),('before_return',p[p.returns==0])]:
    rows.append(dict(**{k:getattr(r,k) for k in ['window_id','dataset','environment_id','episode_uuid','side']},method=method,scope=scope,nade=f.nade.mean(),nfde=f.nade.iloc[-1],exceed05=(f.nade>.05).mean(),requests=run['metrics']['requests'],history6=float(ok)))
  b=pd.DataFrame(runs['baseline']['points']);after=b.returns>0;bp=np.array(b.prediction.tolist())
  for method,run in runs.items():
   pp=np.array([p['prediction'] for p in run['points']]);max_post=max(max_post,float(abs(pp[after]-bp[after]).max()))
   if not ok and method in ['endpoint6','ols6']:max_fallback=max(max_fallback,float(abs(pp-bp).max()))
 result=pd.DataFrame(rows);result.to_csv(out/'windows.csv.gz',index=False);pd.DataFrame(support).to_csv(out/'history_support.csv',index=False);metrics=['nade','nfde','exceed05','requests','history6'];summary=hierarchical(result,['method','scope'],metrics);summary.to_csv(out/'summary.csv',index=False)
 env=result.groupby(['method','scope','dataset','environment_id','episode_uuid','side'],as_index=False)[metrics].mean().groupby(['method','scope','dataset','environment_id','episode_uuid'],as_index=False)[metrics].mean().groupby(['method','scope','dataset','environment_id'],as_index=False)[metrics].mean();env.to_csv(out/'environments.csv',index=False);env.groupby(['method','scope','dataset'],as_index=False)[metrics].mean().to_csv(out/'sources.csv',index=False)
 prior=pd.read_csv(BASE/'results/causal_decay_selector/windows.csv.gz').query("method=='baseline'").set_index('window_id');current=result.query("method=='baseline' and scope=='all'").set_index('window_id');diff=float(abs(prior.nade-current.nade).max());assert diff<1e-12 and max_post==0 and max_fallback==0
 audit=dict(windows=len(w),records=len({k[0] for k in cache}),six_history_windows=sum(s['history6'] for s in support),baseline_max_difference=diff,post_return_prediction_max_difference=max_post,fallback_prediction_max_difference=max_fallback,validation_reads=0,test_reads=0,formal_model_changed=False);(out/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf8');print(summary.to_string(index=False));print(json.dumps(audit))
if __name__=='__main__':main()
