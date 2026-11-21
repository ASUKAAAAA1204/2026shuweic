from pathlib import Path
import json
import numpy as np,pandas as pd
B=Path('C题工作区/第二问探索');p=B/'results/calibration_gap_diagnostic';a=pd.read_csv(p/'diagnostic_points.csv.gz');e=pd.read_csv(B/'results/distance_nested_calibration/outer_points.csv.gz');assert len(a)==len(e)==188148
keys=['window_id','index','model','calibration'];z=a.merge(e,on=keys,suffixes=('_diag','_original'),validate='one_to_one');assert np.array_equal(z.coverage_diag,z.coverage_original)
assert ((a.return_phase>0)==a.after).all();assert a.return_phase.isin([0,1,2]).all()
# Check per-window stages are contiguous, with observed return age reset after each transition.
for _,g in a.query("model=='distance_return' and calibration=='pooled'").groupby('window_id'):
 g=g.sort_values('index');assert (np.diff(g.return_phase)>=0).all() and set(g.return_phase)=={0,1,2}
 # Arrival-frame age contains delay: must not be treated as age zero.
 for phase in [1,2]:assert g[g.return_phase==phase].iloc[0].age_bin>=3
metrics=['coverage','width','pinball','age_fallback','env_mean_fallback'];maxdiff=0.
for factor in ['return_phase','age_band','age_pool_status']:
 g=a.groupby(['model','calibration',factor,'window_id','dataset','environment_id','episode_uuid','side'],as_index=False)[metrics].mean()
 for lev in [['dataset','environment_id','episode_uuid','side'],['dataset','environment_id','episode_uuid'],['dataset','environment_id'],['dataset'],[]]:g=g.groupby(['model','calibration',factor]+lev,as_index=False)[metrics].mean()
 saved=pd.read_csv(p/(factor+'_summary.csv')).set_index(['model','calibration',factor]);d=float(abs(g.set_index(['model','calibration',factor])[metrics]-saved[metrics]).to_numpy().max());maxdiff=max(maxdiff,d);assert d<1e-12
# Overall fallback rate needs the original complete population, not averages of reweighted subgroups.
g=a.groupby(['model','calibration','window_id','dataset','environment_id','episode_uuid','side'],as_index=False).age_fallback.mean()
for lev in [['dataset','environment_id','episode_uuid','side'],['dataset','environment_id','episode_uuid'],['dataset','environment_id'],['dataset'],[]]:g=g.groupby(['model','calibration']+lev,as_index=False).age_fallback.mean()
g.to_csv(p/'overall_fallback.csv',index=False)
raw=a.groupby(['model','calibration']).agg(rows=('index','size'),fallback_points=('age_fallback','sum'));raw.to_csv(p/'fallback_raw_counts.csv')
audit=dict(passed=True,checked_rows=len(a),original_coverage_unchanged=True,phase_order_verified=True,return_age_includes_delay=True,summary_max_difference=maxdiff,scope='saved point consistency, phase order and independent hierarchical arithmetic')
(p/'independent_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf8');print(json.dumps(audit));print(g.to_string(index=False));print(raw.to_string())
