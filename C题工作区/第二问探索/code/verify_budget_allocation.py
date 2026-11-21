from pathlib import Path
import json
import numpy as np,pandas as pd
p=Path('C题工作区/第二问探索/results/budget_allocation');old=p.parent/'calibration_closed_loop'
w=pd.read_csv(p/'windows.csv.gz');a=pd.read_csv(p/'actions.csv.gz');rq=a[a.event=='request'];rt=a[a.event=='return']
assert len(w)==6108 and not w.duplicated(['method','window_id']).any()
pairs=rq.merge(rt,on=['window_id','method','request_id'],suffixes=('_request','_return'),validate='one_to_one');assert len(pairs)==12216
assert np.allclose(pairs.relative_return-pairs.relative_request,.1,atol=1e-9)
assert (rq.groupby(['window_id','method']).size()==2).all()
assert rq.query("method=='second_half' and request_id==2").relative.min()>=.5-1e-9
assert rq.query("method=='quarter_half' and request_id==1").relative.min()>=.25-1e-9
assert rq.query("method=='quarter_half' and request_id==2").relative.min()>=.5-1e-9
prev=pd.read_csv(old/'windows.csv.gz');delta=0.
for new,prior in [('fixed','fixed'),('unrestricted','post_env_mean')]:
 left=w[w.method==new].set_index('window_id').sort_index();right=prev[prev.method==prior].set_index('window_id').sort_index()
 cols=['nade','nfde','exceed05','requests','waiting_s'];d=np.abs(left[cols].to_numpy()-right[cols].to_numpy()).max();delta=max(delta,float(d));assert d<1e-12
# Build common recorded decision frame grid from the previous direct run's per-window actions
# Equality of second-half and quarter-half second request times is observable, not proof of threshold inactivity at all earlier frames.
x=rq.query("method=='second_half' and request_id==2").set_index('window_id').relative
y=rq.query("method=='quarter_half' and request_id==2").set_index('window_id').relative
assert np.allclose(x.sort_index(),y.sort_index())
metrics=['nade','nfde','exceed05','requests','waiting_s','coverage','width']
g=w.groupby(['method','dataset','environment_id','episode_uuid','side'],as_index=False)[metrics].mean()
for keys in [['method','dataset','environment_id','episode_uuid'],['method','dataset','environment_id'],['method','dataset'],['method']]:g=g.groupby(keys,as_index=False)[metrics].mean()
s=pd.read_csv(p/'summary.csv').set_index('method');err=float(np.abs(g.set_index('method')[metrics]-s[metrics]).to_numpy().max());assert err<1e-12
result=dict(passed=True,replays=len(w),paired_actions=len(pairs),previous_control_max_difference=delta,summary_max_difference=err,budget_and_delay_verified=True,gates_verified=True,identical_second_request_times_between_gated_methods=True,scope='independent saved-table arithmetic and action checks')
(p/'independent_audit.json').write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps(result))
