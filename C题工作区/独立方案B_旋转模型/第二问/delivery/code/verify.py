from pathlib import Path
import json,gzip
import numpy as np
import pandas as pd
from run_complete import aggregate
BASE=Path(__file__).resolve().parents[1]

def main():
 out=BASE/'results';groups={};actions=pd.read_csv(out/'actions.csv.gz');maximum=0.;checked=0
 with gzip.open(out/'points.jsonl.gz','rt',encoding='utf8') as f:
  for line in f:
   p=json.loads(line);groups.setdefault(p['window_id'],[]).append(p)
 rows=[];metrics=pd.read_csv(out/'windows.csv').set_index('window_id')
 for key,points in groups.items():
  a=actions[actions.window_id==key];req=a[a.event=='request'].sort_values('time');ret=a[a.event=='return'].sort_values('time');assert len(req)==len(ret)==2
  for request,returned in zip(req.itertuples(),ret.itertuples()):
   assert abs(request.arrival-request.time-.1)<1e-9 and returned.processed_at+1e-12>=returned.time and abs(request.sample_time-returned.sample_time)<1e-12
   # Check return is processed at the first scoring time at/after arrival.
   expected=next(p['time'] for p in points if p['time']+1e-12>=request.arrival);assert abs(expected-returned.processed_at)<1e-12
   sample=next(p for p in points if abs(p['time']-request.sample_time)<1e-12)
   assert sample['returns']==request.request_id-1
   checked+=1
  error=np.linalg.norm(np.array([p['prediction'] for p in points])-np.array([p['reference'] for p in points]),axis=1)/points[0]['scale'];radius=np.array([p['radius']/p['scale'] for p in points]);delta=error-radius
  result=dict(nade=error.mean(),nfde=error[-1],coverage=(delta<=0).mean(),width=(2*radius).mean(),pinball=np.maximum(.9*delta,-.1*delta).mean(),requests=2.,waiting_s=float(np.sum(req.arrival-req.time)))
  maximum=max(maximum,max(abs(float(metrics.loc[key,k])-v) for k,v in result.items()))
  rows.append({k:points[0][k] for k in ['window_id','dataset','environment_id','episode_uuid','side','split']}|result)
 assert len(groups)==1739 and checked==3478 and maximum<1e-9
 audit=dict(passed=True,windows=len(groups),request_return_pairs=checked,max_independent_metric_difference=float(maximum),arrival_first_scoring_frame_checked=True,test_reads=0)
 (out/'independent_verification.json').write_text(json.dumps(audit,indent=2),encoding='utf8');print(audit)
if __name__=='__main__':main()
