from pathlib import Path
import json,gzip
import numpy as np
from replay_engine import MotionState,FixedPolicy,replay
from rotation_state import factory
BASE=Path(__file__).resolve().parents[1]

def main():
 groups={}
 with gzip.open(BASE/'results/points.jsonl.gz','rt',encoding='utf8') as f:
  for line in f:
   p=json.loads(line);groups.setdefault(p['window_id'],[]).append(p)
 checked=0;maximum=0
 # Each actual window supplies timing and the frozen initial forecast until first return.
 # Extend with a deterministic synthetic tail: invariant concerns when payload becomes visible.
 for key,points in groups.items():
  times=np.array([p['time'] for p in points]);reference=np.array([p['reference'] for p in points]);sample_index=next(k for k,t in enumerate(times) if t-times[0]+1e-12>=.3);arrival=times[sample_index]+.1
  hist_t=np.array([times[0]-.0666666667,times[0]-.0333333333]);hist_x=np.zeros((2,3));forecast=np.array([p['prediction'] for p in points]);State=factory(times,forecast,MotionState)
  original=replay(times,reference,hist_t,hist_x,FixedPolicy(),state_factory=State)
  altered=reference.copy();altered[sample_index]+=np.array([10.,-7.,3.]);changed=replay(times,altered,hist_t,hist_x,FixedPolicy(),state_factory=State)
  before=times+1e-12<arrival;a=np.array([p['prediction'] for p in original['points']]);b=np.array([p['prediction'] for p in changed['points']]);difference=float(np.max(abs(a[before]-b[before])));maximum=max(maximum,difference);assert difference==0
  assert any(np.linalg.norm(x-y)>0 for x,y,t in zip(a,b,times) if t+1e-12>=arrival)
  assert original['actions']==changed['actions'];checked+=1
 audit=dict(passed=True,window_timing_cases=checked,max_prearrival_prediction_difference=maximum,postarrival_payload_effect_observed=True,request_schedule_unchanged=True,scope='runtime state and disclosure invariant on actual window clocks; synthetic initial-history values',test_reads=0)
 (BASE/'results/payload_perturbation.json').write_text(json.dumps(audit,indent=2),encoding='utf8');print(audit)
if __name__=='__main__':main()
