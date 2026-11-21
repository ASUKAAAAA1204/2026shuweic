from pathlib import Path
import json
import numpy as np
from recovery_interface import RecoveryTracker

def main():
 checks=[]
 def record(name):checks.append(name)
 s=RecoveryTracker();assert s.predict(0).mode=='uninitialized';record('uninitialized unavailable')
 s.receive(0,[0,0,0],True,0);assert s.predict(0).mode=='recovered_single';record('one frame cannot imply continuous velocity')
 s.receive(.03,[.03,0,0],True,.03);assert s.predict(.06).mode=='tracking';record('two continuous samples track')
 s.receive(.06,None,False,.06);assert s.predict(.07).mode=='missing';record('bad validity disclosed then missing')
 before=s.predict(.3);assert before.mode=='missing';record('no future recovery awareness')
 s.receive(.3,[.3,0,0],True,.4);after=s.predict(.4);assert after.mode=='recovered_single' and abs(after.observation_age-.1)<1e-12;record('delayed first sample retains age')
 s.receive(.33,[.33,0,0],True,.43);assert s.predict(.43).mode=='tracking';record('continuous pair can resume tracking')
 a=s.predict(.43,tolerance=.05,radius=.01,supported=True);assert a.tolerance_status=='empirical_within' and not a.reliable_guarantee;record('empirical fit never claims guarantee')
 assert s.predict(.7,tolerance=.05,radius=.01,supported=True).tolerance_status=='uncertified';record('stale state refuses implicit certification')
 try:s.receive(.33,[0,0,0],True,.5);raise AssertionError('duplicate accepted')
 except ValueError:record('duplicate sample rejected')
 try:s.receive(.8,[0,0,0],True,.7);raise AssertionError('future accepted')
 except ValueError:record('future sample rejected')
 # Independent replay receives current bad flag only after scoring; no first-frame oracle.
 import sys
 BASE=Path(__file__).resolve().parents[1];OLD=BASE.parent/'第二问求解';sys.path.insert(0,str(OLD/'code'));import s0_audit_replay as s0
 import pandas as pd
 rec=pd.read_csv(s0.DATA/'record_index.csv').query("split=='train'");rows=[]
 for r in rec.itertuples():
  with np.load(s0.DATA/'records'/f'{r.episode_uuid}.npz',allow_pickle=False) as z:
   for side in ['left_hand','right_hand']:
    t=z[side+'__t_s'];x=z[side+'__position'].mean(axis=1);good=z[side+'__reference_candidate'];tracker=RecoveryTracker(max_gap_s=3*float(np.median(np.diff(t))))
    for k,now in enumerate(t):
     pre=tracker.predict(float(now));tracker.receive(float(now),x[k] if good[k] else None,bool(good[k]),float(now));post=tracker.predict(float(now))
     rows.append(dict(dataset=r.dataset,episode_uuid=r.episode_uuid,environment_id=r.environment_id,side=side,index=k,pre_mode=pre.mode,post_mode=post.mode,current_qualified=bool(good[k]),pre_guarantee=pre.reliable_guarantee,post_guarantee=post.reliable_guarantee))
 m=pd.DataFrame(rows);assert not m.pre_guarantee.any() and not m.post_guarantee.any();out=BASE/'results/recovery_interface';out.mkdir(exist_ok=True);m.to_csv(out/'causal_modes.csv.gz',index=False);m.groupby(['dataset','pre_mode','post_mode','current_qualified']).size().to_csv(out/'mode_counts.csv');audit=dict(passed=True,checks=checks,train_records=len(rec),hand_timestamps=len(m),test_reads=0,development_reads=0,reference_enters_predict=False,current_quality_used_only_after_predict=True,guaranteed_claims=0);(out/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf8');print(json.dumps(audit))
if __name__=='__main__':main()
