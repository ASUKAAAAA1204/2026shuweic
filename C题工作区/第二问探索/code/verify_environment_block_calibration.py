from pathlib import Path
import json
import numpy as np,pandas as pd
BASE=Path(__file__).resolve().parents[1];out=BASE/'results/environment_block_calibration';b=pd.read_csv(out/'environment_scores.csv');a=pd.read_csv(out/'oof_blocks.csv');maximum=0.
for r in a.itertuples():
 fit=b[(b.dataset==r.dataset)&(b.phase==r.phase)&(b.environment_id!=r.held)]
 # Independent integer arithmetic: .8=4/5 and .9=9/10.
 n=len(fit);num,den=(4,5) if r.nominal==.8 else (9,10);k=((n+1)*num+den-1)//den
 assert k==r.rank and n==r.calibration_environments
 if k>n:assert not r.finite and np.isnan(r.radius_raw) and r.status=='no_finite_range'
 else:
  value=sorted(fit.max_raw_error.tolist())[k-1];maximum=max(maximum,abs(value-r.radius_raw));assert abs(value-r.radius_raw)<1e-12
  held=b[(b.dataset==r.dataset)&(b.phase==r.phase)&(b.environment_id==r.held)].iloc[0];assert r.environment_all_covered==float(held.max_raw_error<=value)
 assert not r.guaranteed
assert ((a.nominal==.9)&a.finite).sum()==20
result=dict(passed=True,checked_blocks=len(a),independent_integer_rank=True,radius_max_difference=maximum,finite_90_blocks=20,total_90_blocks=int((a.nominal==.9).sum()),all_missing_and_true_recovery_90_abstain=bool((~a[(a.nominal==.9)&a.phase.isin(['missing','single_after_observed_invalid'])].finite).all()),warning='rank feasibility under specified construction, not a proof that no other bounded or parametric method can work')
(out/'independent_audit.json').write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps(result))
