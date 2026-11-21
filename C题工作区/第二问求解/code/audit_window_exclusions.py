"""Audit original window exclusions without changing the frozen evaluation set."""
from pathlib import Path
from collections import Counter
import json
import numpy as np
import pandas as pd
import s0_audit_replay as s0
BASE=Path(__file__).resolve().parents[1]
def main():
 records=pd.read_csv(s0.DATA/'record_index.csv').query("split!='test'");expected=pd.read_csv(s0.Q1.parent/'C题工作区/阶段交付/stage1_review/corrected_results/data/windows.csv').query("split!='test'");actual=set();rows=[]
 for r in records.itertuples():
  with np.load(s0.DATA/'records'/f'{r.episode_uuid}.npz',allow_pickle=False) as z:
   for side in ['left_hand','right_hand']:
    t=z[side+'__t_s'];valid=z[side+'__reference_candidate'];dt=np.diff(t);med=float(np.median(dt[dt>0]));last_end=-1e30;counts=Counter()
    for i in range(len(t)):
     if t[i]<last_end+1:counts['inter_window_spacing']+=1;continue
     if t[i]-t[0]<.2-1e-6:counts['insufficient_past_history']+=1;continue
     h=int(np.searchsorted(t,t[i]-.2,side='left'));j=int(np.searchsorted(t,t[i]+1,side='right'))
     if j>=len(t) or t[j-1]-t[i]<1-med*1.1:counts['insufficient_future_reference_span']+=1;continue
     if not valid[h:j].all():counts['nonqualified_reference_or_history']+=1;continue
     if (np.diff(t[h:j])>3*med).any():counts['timestamp_gap']+=1;continue
     counts['retained']+=1;actual.add(f'{r.episode_uuid}_{side}_{i}');last_end=t[i]+1
    assert sum(counts.values())==len(t)
    for reason,count in counts.items():rows.append(dict(episode_uuid=r.episode_uuid,dataset=r.dataset,split=r.split,side=side,reason=reason,candidate_anchors=count))
 assert actual==set(expected.window_id),(len(actual),len(expected))
 out=BASE/'results/S0';pd.DataFrame(rows).to_csv(out/'window_exclusion_reasons.csv',index=False);pd.DataFrame(rows).groupby(['split','dataset','reason']).candidate_anchors.sum().to_csv(out/'window_exclusion_summary.csv')
 (out/'window_reconstruction_audit.json').write_text(json.dumps(dict(passed=True,exact_window_id_match=True,reconstructed_windows=len(actual),test_trajectory_reads=0,exclusion_accounting='first failing criterion; spacing exclusions are not independent quality failures',timing='post-freeze read-only audit; no window or policy changes'),indent=2),encoding='utf8');print('exactly reconstructed',len(actual),'windows')
if __name__=='__main__':main()
