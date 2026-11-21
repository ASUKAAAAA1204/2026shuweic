from pathlib import Path
import sys,json,hashlib
import numpy as np,pandas as pd
BASE=Path(__file__).resolve().parents[1];OLD=BASE.parent/'第二问求解';sys.path.insert(0,str(OLD/'code'))
import s0_audit_replay as s0
from update_calibration_experiment import hierarchical

def predict(last,last_t,v,now):return last-.2*np.expm1(-(now-last_t)/.2)*v

def main():
 out=BASE/'results/natural_recovery';out.mkdir(exist_ok=True);rec=pd.read_csv(s0.DATA/'record_index.csv').query("split=='train'");scale=json.loads((OLD/'results/S0/fold_scales.json').read_text());rows=[];elig=[];traces=[];events=0
 for r in rec.itertuples():
  with np.load(s0.DATA/'records'/f'{r.episode_uuid}.npz',allow_pickle=False) as z:
   for side in ['left_hand','right_hand']:
    t=z[side+'__t_s'];x=z[side+'__position'].mean(axis=1);valid=z[side+'__reference_candidate'];bad=z[side+'__bad'].astype(bool);med=np.median(np.diff(t));edges=np.diff(np.r_[False,bad,False].astype(int));aa=np.flatnonzero(edges==1);bb=np.flatnonzero(edges==-1)
    for start,end in zip(aa,bb):
     event=f'{r.episode_uuid}_{side}_{start}';events+=1
     for length in [.2,.6,1.]:
      why='eligible'
      if start<2 or not valid[max(0,start-2):start].all():why='no_continuous_qualified_initial_pair'
      elif end>=len(t):why='right_censored'
      else:
       j=np.searchsorted(t,t[end]+length,side='right')
       if j==len(t) or t[j-1]-t[end]<length-1.1*med:why='short_reference_span'
       elif not valid[end:j].all():why='nonqualified_recovery_reference'
       elif (np.diff(t[start-2:j])>3*med).any():why='timestamp_gap'
      elig.append(dict(event=event,dataset=r.dataset,environment_id=r.environment_id,length_s=length,reason=why))
      if why!='eligible':continue
      initial_v=(x[start-1]-x[start-2])/(t[start-1]-t[start-2]);sc=scale[r.environment_id][r.dataset]
      for method in ['stale','reset_decay','stream']:
       last=x[start-1].copy();last_t=float(t[start-1]);v=initial_v.copy();pre=[];post=[]
       for k in range(end,j):
        before=predict(last,last_t,v,t[k]);pre.append(float(np.linalg.norm(before-x[k])/sc))
        if method!='stale':
         if method=='stream' and k>end:v=(x[k]-x[k-1])/(t[k]-t[k-1])
         else:v=v*np.exp(-(t[k]-last_t)/.2)
         last=x[k].copy();last_t=float(t[k])
        after=predict(last,last_t,v,t[k]);post.append(float(np.linalg.norm(after-x[k])/sc))
        traces.append(dict(event=event,method=method,length_s=length,relative=t[k]-t[end],pre_error=pre[-1],post_error=post[-1]))
       rows.append(dict(window_id=event,event=event,episode_uuid=r.episode_uuid,dataset=r.dataset,environment_id=r.environment_id,side=side,length_s=length,method=method,nade_pre=np.mean(pre),nade_post=np.mean(post),nfde_pre=pre[-1],exceed_pre_05=np.mean(np.array(pre)>.05),points=len(pre),bad_duration_s=float(t[end]-t[start])))
 m=pd.DataFrame(rows);e=pd.DataFrame(elig);m.to_csv(out/'windows.csv',index=False);e.to_csv(out/'eligibility.csv',index=False);pd.DataFrame(traces).to_csv(out/'points.csv.gz',index=False);s=hierarchical(m,['length_s','method'],['nade_pre','nade_post','nfde_pre','exceed_pre_05']);s.to_csv(out/'summary.csv',index=False);e.groupby(['dataset','length_s','reason']).size().to_csv(out/'exclusion_summary.csv');m.groupby(['length_s']).agg(hand_events=('event','nunique'),records=('episode_uuid','nunique'),environments=('environment_id','nunique')).to_csv(out/'counts.csv')
 audit=dict(train_records=len(rec),bad_hand_events=events,eligible_hand_events=m.event.nunique(),test_reads=0,development_reads=0,hidden_bad_coordinates_enter_prediction=False,mode='naturally recovered zero-delay stream, pre-current-observation forecast vs post-observation state',formal_model_unchanged=True)
 (out/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf8');print(json.dumps(audit));print(s.to_string(index=False))
if __name__=='__main__':main()
