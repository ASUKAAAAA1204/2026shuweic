"""Fix actionable end-boundary waste without changing frozen original engine."""
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
BASE=Path(__file__).resolve().parents[1];OLD=BASE.parent/'第二问求解';sys.path.insert(0,str(OLD/'code'))
from engine import replay,FixedPolicy
import s0_audit_replay as s0
class Eager:
 def request(self,view):return True

def safe_replay(times,reference,history_times,history_positions,policy,**kwargs):
 nominal=kwargs.get('window_end',float(times[-1]));kwargs['window_end']=min(float(nominal),float(times[-1]))
 return replay(times,reference,history_times,history_positions,policy,**kwargs)

def main():
 out=BASE/'results/end_guard';out.mkdir(exist_ok=True);windows=pd.read_csv(s0.Q1.parent/'C题工作区/阶段交付/stage1_review/corrected_results/data/windows.csv').query("split=='train'");rows=[];cache={};maxdiff=0.;saved=0
 for w in windows.itertuples():
  key=(w.episode_uuid,w.side)
  if key not in cache:
   with np.load(s0.DATA/'records'/f'{w.episode_uuid}.npz',allow_pickle=False) as z:cache[key]=(z[w.side+'__t_s'],z[w.side+'__position'].mean(axis=1))
  t,x=cache[key];i,j=w.start_index,w.end_index_exclusive
  for length in [.2,.6,1.]:
   ix=np.arange(i,j)[t[i:j]-t[i]<=length+1e-9]
   for delay in [0,.1,.2]:
    for name,constructor in [('fixed',FixedPolicy),('eager',Eager)]:
     kwargs=dict(delay=delay,window_end=float(t[i]+length));args=(t[ix],x[ix],t[i-2:i],x[i-2:i]);a=replay(*args,constructor(),**kwargs);b=safe_replay(*args,constructor(),**kwargs)
     delta=float(np.max(np.abs(np.array([p['prediction'] for p in a['points']])-np.array([p['prediction'] for p in b['points']]))));maxdiff=max(maxdiff,delta)
     assert delta<1e-9 and b['metrics']['requests']<=a['metrics']['requests'];req=sum(v['event']=='request' for v in b['actions']);ret=sum(v['event']=='return' for v in b['actions']);assert req==ret
     rows.append(dict(window_id=w.window_id,policy=name,length_s=length,delay_s=delay,old_requests=a['metrics']['requests'],safe_requests=b['metrics']['requests'],max_prediction_diff=delta));saved+=a['metrics']['requests']-b['metrics']['requests']
 m=pd.DataFrame(rows);m.to_csv(out/'paired_train.csv.gz',index=False);m.groupby(['policy','length_s','delay_s'])[['old_requests','safe_requests']].mean().to_csv(out/'summary.csv');audit=dict(passed=True,replays_checked=len(m),prediction_max_difference=maxdiff,removed_useless_paid_requests=int(saved),safe_unreturned_paid_requests=0,train_only=True,development_reads=0,test_reads=0,scope='fixed and eager policies; remaining-time-dependent policies require separate evaluation because public remaining time changes',frozen_original_unchanged=True)
 (out/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf8');print(json.dumps(audit))
if __name__=='__main__':main()
