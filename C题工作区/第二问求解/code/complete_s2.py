"""Complete S2 train common metrics for all preregistered durations/delays, OOF radii."""
from pathlib import Path
import json,time
import numpy as np
import pandas as pd
from engine import replay,FixedPolicy,NeverPolicy
from update_calibration_experiment import quantiles,hierarchical
import s0_audit_replay as s0
BASE=Path(__file__).resolve().parents[1];OUT=BASE/'results/S2'
def main():
 OUT.mkdir(parents=True,exist_ok=True);start=time.perf_counter();w=pd.read_csv(s0.Q1.parent/'C题工作区/阶段交付/stage1_review/corrected_results/data/windows.csv').query("split=='train'");scales=json.loads((BASE/'results/S0/fold_scales.json').read_text());cal=json.loads((BASE/'results/S3/fold_calibrators.json').read_text());cache={};res=[]
 for a in w.itertuples():
  key=(a.episode_uuid,a.side)
  if key not in cache:
   with np.load(s0.DATA/'records'/f'{a.episode_uuid}.npz',allow_pickle=False) as z:cache[key]=(z[a.side+'__t_s'],z[a.side+'__position'].mean(axis=1))
  t,x=cache[key];i,j=a.start_index,a.end_index_exclusive;age=t[i:j]-t[i-1];v=(x[i-1]-x[i-2])/(t[i-1]-t[i-2])
  for method in ['hold','cv']:
   pred=np.broadcast_to(x[i-1],(j-i,3)) if method=='hold' else x[i-1]+age[:,None]*v
   for h,e in zip(age,np.linalg.norm(pred-x[i:j],axis=1)):res.append(dict(dataset=a.dataset,environment_id=a.environment_id,method=method,age_bin=min(30,max(1,int(np.ceil(h*30)))),error=e))
 res=pd.DataFrame(res);fitted={}
 for held in sorted(w.environment_id.unique()):
  for method in ['hold','cv']:fitted[(held,method)]=quantiles(res[(res.method==method)&(res.environment_id!=held)])
 rows=[]
 for a in w.itertuples():
  t,x=cache[(a.episode_uuid,a.side)];i,j=a.start_index,a.end_index_exclusive;scale=scales[a.environment_id][a.dataset];meta={k:getattr(a,k) for k in ['window_id','episode_uuid','dataset','environment_id','side']}
  for length in [.2,.6,1.]:
   idx=np.arange(i,j)[t[i:j]-t[i]<=length+1e-9]
   for method,delays in [('hold',[0]),('cv',[0]),('damped',[0]),('fixed',[0,.1,.2])]:
    for delay in delays:
     r=replay(t[idx],x[idx],t[i-2:i],x[i-2:i],FixedPolicy() if method=='fixed' else NeverPolicy(),delay=delay,motion='damped' if method=='fixed' else method,window_end=float(t[i]+length));p=pd.DataFrame(r['points']);error=p.error.to_numpy();bins=np.clip(np.ceil(p.age*30).astype(int),1,30)
     radius=np.array([fitted[(a.environment_id,method)][(a.dataset,int(b))] if method in ['hold','cv'] else cal[a.environment_id]['open'][f'{a.dataset}|{b}'] for b in bins]);row={**meta,'method':method,'length_s':length,'delay_s':delay,**r['metrics'],'nade':r['metrics']['ade']/scale,'nfde':r['metrics']['fde']/scale,'np95':r['metrics']['p95']/scale,'coverage':float(np.mean(error<=radius)),'path_covered':bool(np.all(error<=radius)),'width':float(np.mean(2*radius/scale))}
     for frac in [.02,.05,.1]:row[f'exceed_{frac}']=float(np.mean(error>scale*frac))
     rows.append(row)
 m=pd.DataFrame(rows);m.to_csv(OUT/'windows.csv',index=False);cols=['nade','nfde','np95','coverage','path_covered','width','exceed_0.02','exceed_0.05','exceed_0.1','requests','cost','waiting_s','no_request_wait_s','strategy_compute_s','step_compute_p95_s'];hierarchical(m,['method','length_s','delay_s'],cols).to_csv(OUT/'summary.csv',index=False)
 audit=dict(rows=len(m),windows=len(w),test_trajectory_reads=0,development_replays=0,radius_fit_excludes_held_global_env=True,budget_violations=int((m.requests>2).sum()),elapsed_s=time.perf_counter()-start);(OUT/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf8');print(json.dumps(audit),flush=True)
if __name__=='__main__':main()
