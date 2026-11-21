from pathlib import Path
import json
import numpy as np,pandas as pd
B=Path('C题工作区/第二问探索');p=B/'results/distance_convergence';m=pd.read_csv(p/'windows.csv.gz');old=pd.read_csv(B/'results/distance_aligned_return/windows.csv.gz').query("method=='distance_return'")
z=m.query("method=='distance20'").merge(old,on='window_id',suffixes=('_new','_old'),validate='one_to_one');cols=['nade','nfde','exceed05'];assert len(z)==1527
assert max(float(abs(z[c+'_new']-z[c+'_old']).max()) for c in cols)<1e-12
v=m.pivot(index='window_id',columns='method',values='nade');d=float(abs(v.distance_return-v.distance20).max())
e=pd.read_csv(p/'environments.csv');pair=e.pivot(index=['dataset','environment_id'],columns='method',values='nade').reset_index();pair['gain_pct']=100*(1-pair.distance_return/pair.secant);pair.to_csv(p/'environment_effects.csv',index=False)
summary=pair.groupby('dataset').gain_pct.agg(['size','min','max',lambda x:int((x>0).sum()),lambda x:int((x< -5).sum())]);summary.to_csv(p/'source_environment_effects.csv')
s=pd.read_csv(p/'stage_windows.csv.gz');check=s.groupby(['window_id','method']).full_window_contribution.sum().reset_index().merge(m[['window_id','method','nade']],on=['window_id','method'],validate='one_to_one');assert abs(check.full_window_contribution-check.nade).max()<1e-12
labels=pd.read_csv(B/'results/stage_diagnostic/labels.csv');g=m.merge(labels[['window_id','boundary']],on='window_id',validate='many_to_one')
from sys import path
path.insert(0,str(B.parent/'第二问求解/code'))
# Explicit hierarchy to avoid importing paths from another workspace.
metrics=['nade','nfde','exceed05'];r=g.copy()
for lev in [['dataset','environment_id','episode_uuid','side'],['dataset','environment_id','episode_uuid'],['dataset','environment_id'],['dataset'],[]]:r=r.groupby(['boundary','method']+lev,as_index=False)[metrics].mean()
r.to_csv(p/'boundary_summary.csv',index=False)
audit=dict(passed=True,distance20_exact_previous=True,max_window_nade_20_to_100_difference=d,source_environment_units=len(pair),units_improved=int((pair.gain_pct>0).sum()),units_worse_over_5pct=int((pair.gain_pct< -5).sum()),worst_unit_gain=float(pair.gain_pct.min()),stage_full_contributions_reconcile=True)
(p/'stability_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf8');print(json.dumps(audit));print(summary.to_string());print(pd.read_csv(p/'stage_summary.csv').query("method!='distance20'").to_string(index=False));print(r.query("method!='distance20'").to_string(index=False))
