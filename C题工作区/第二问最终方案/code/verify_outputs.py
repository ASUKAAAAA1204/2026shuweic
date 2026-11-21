"""Independently recalculate scores and audit paid observation chronology."""
from pathlib import Path
import gzip,json
import numpy as np
import pandas as pd
BASE=Path(__file__).resolve().parents[1]
def main():
 out=BASE/'results';windows=pd.read_csv(out/'window_metrics.csv').set_index('window_id');groups={}
 with gzip.open(out/'joint_points.jsonl.gz','rt',encoding='utf8') as f:
  for line in f:
   p=json.loads(line);e=float(np.linalg.norm(np.array(p['prediction'])-p['reference']));assert abs(e-p['error'])<1e-12
   d=e/p['scale']-p['range']['radius']/p['scale'];groups.setdefault(p['window_id'],[]).append([e/p['scale'],float(d<=0),2*p['range']['radius']/p['scale'],max(.9*d,-.1*d)])
 maxdiff=0.
 for key,values in groups.items():
  a=np.array(values);r=windows.loc[key]
  expected=[r.nade,r.coverage,r.width,r.pinball]
  maxdiff=max(maxdiff,float(np.max(np.abs(a.mean(axis=0)-expected))));assert abs(a[-1,0]-r.nfde)<1e-12
 assert maxdiff<1e-12
 actions=pd.read_csv(out/'actions.csv.gz');count=0
 for key,g in actions.groupby('window_id'):
  requests=g[g.event=='request'];returns=g[g.event=='return'];assert len(requests)<=2 and len(requests)==len(returns)
  for q in requests.itertuples():
   ret=returns[returns.request_id==q.request_id].iloc[0]
   assert abs(q.arrival-q.time-.1)<1e-10 and ret.processed_at+1e-12>=q.arrival and ret.sample_time==q.sample_time
  count+=len(requests)
 result=dict(passed=True,windows=len(groups),points=sum(map(len,groups.values())),requests=count,independent_scoring_max_absolute_difference=maxdiff,scope='saved predictions independently rescored; request delay and budget checked')
 (out/'independent_verification.json').write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps(result))
if __name__=='__main__':main()
