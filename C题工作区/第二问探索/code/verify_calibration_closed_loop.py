from pathlib import Path
import json
import numpy as np,pandas as pd
p=Path(__file__).resolve().parents[1]/'results/calibration_closed_loop'
w=pd.read_csv(p/'windows.csv.gz');a=pd.read_csv(p/'actions.csv.gz');s=pd.read_csv(p/'summary.csv')
assert len(w)==6108 and w.groupby('method').window_id.nunique().eq(1527).all()
assert not w.duplicated(['window_id','method']).any()
requests=a[a.event=='request'];returns=a[a.event=='return']
assert len(requests)==len(returns)==len(w)*2
assert (requests.groupby(['window_id','method']).size()==2).all()
metrics=['nade','nfde','exceed05','requests','waiting_s','coverage','width']
g=w.groupby(['method','dataset','environment_id','episode_uuid','side'],as_index=False)[metrics].mean()
for keys in [['method','dataset','environment_id','episode_uuid'],['method','dataset','environment_id'],['method','dataset'],['method']]:g=g.groupby(keys,as_index=False)[metrics].mean()
d=float(np.max(np.abs(g.set_index('method')[metrics]-s.set_index('method')[metrics])))
assert d<1e-12
pairs=requests.merge(returns,on=['window_id','method','request_id'],suffixes=('_request','_return'),validate='one_to_one')
assert np.allclose(pairs.relative_return-pairs.relative_request,.1,atol=1e-9)
result=dict(passed=True,summary_max_abs_difference=d,request_return_pairs=len(pairs),full_common_windows=1527,delay_verified=.1,scope='saved-result independent arithmetic and action consistency; not raw trajectory reimplementation')
(p/'independent_audit.json').write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps(result))
