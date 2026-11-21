from pathlib import Path
import json
import numpy as np,pandas as pd
B=Path('C题工作区/第二问探索');p=B/'results/independent_rule_selection';splits=json.loads((p/'splits.json').read_text());sel=pd.read_csv(p/'selections.csv');scores=pd.read_csv(p/'inner_scores.csv');a=pd.read_csv(p/'points.csv.gz')
for f in splits:
 sets=[set(f[k]) for k in ['train','calibration','selection']];assert list(map(len,sets))==[7,4,4]
 assert all(not sets[i]&sets[j] for i in range(3) for j in range(i)) and all(f['outer'] not in s for s in sets)
 rank=scores[scores.outer==f['outer']].copy();rank['priority']=rank.method.map({'all_age':0,'prior_only':1,'neighbor':2});expected=rank.sort_values(['pinball','priority']).iloc[0].method;assert sel.set_index('outer').loc[f['outer'],'choice']==expected
assert not a.duplicated(['method','window_id','index']).any();assert a.groupby('method').window_id.nunique().eq(1527).all()
keys=['window_id','index'];old=a[a.method=='all_age'].set_index(keys);new=a[a.method=='inner_selected'].set_index(keys)
for r in sel[~sel.supported].itertuples():
 idx=old[old.environment_id==r.outer].index;assert np.allclose(old.loc[idx,['radius','coverage','width','pinball']],new.loc[idx,['radius','coverage','width','pinball']])
metrics=['coverage','width','pinball'];g=a.groupby(['method','window_id','dataset','environment_id','episode_uuid','side'],as_index=False)[metrics].mean()
for lev in [['dataset','environment_id','episode_uuid','side'],['dataset','environment_id','episode_uuid'],['dataset','environment_id'],['dataset'],[]]:g=g.groupby(['method']+lev,as_index=False)[metrics].mean()
s=pd.read_csv(p/'summary.csv').set_index('method');delta=float(abs(g.set_index('method')[metrics]-s[metrics]).to_numpy().max());assert delta<1e-12
result=dict(passed=True,disjoint_splits=len(splits),rank_choices_verified=True,unsupported_defaults_verified=3,all_windows_retained=1527,summary_max_difference=delta,scope='split provenance, saved inner ranking, defaults and aggregation; not independent residual refit')
(p/'independent_audit.json').write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps(result))
