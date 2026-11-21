"""Verify calibration quantiles and nested selection isolation from saved residuals."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
BASE=Path(__file__).resolve().parents[1]
def main():
 p=pd.read_csv(BASE/'results/S3/residual_points.csv.gz');f=json.loads((BASE/'results/S3/fold_calibrators.json').read_text());checked=0;delta=0.
 for held,model in f.items():
  op=p[(p.method=='open')&(p.environment_id!=held)];post=p[(p.method=='secant')&p.after&(p.environment_id!=held)]
  for name,g in [('open',op),('post',post)]:
   for key,group in g.groupby(['dataset','age_bin']):
    tag=f'{key[0]}|{key[1]}'
    if name=='post' and len(group)<100:
     assert tag not in model[name];continue
    expected=float(np.quantile(group.error.to_numpy(),.9,method='higher'));actual=model[name][tag];delta=max(delta,abs(expected-actual));assert abs(expected-actual)<1e-9;checked+=1
 out=pd.read_csv(BASE/'results/S4_adaptive/nested_selections.csv');counts=[]
 for row in out.itertuples():
  inner=pd.read_csv(BASE/f'results/S4_adaptive/inner_{row.outer}.csv.gz');assert row.outer not in set(inner.environment_id);assert inner.environment_id.nunique()==15;counts.append(len(inner))
 check=dict(passed=True,quantiles_independently_recomputed=checked,max_quantile_difference=delta,outer_folds=len(out),inner_results=sum(counts),outer_environment_in_selection_data=0)
 (BASE/'results/S3/independent_audit.json').write_text(json.dumps(check,indent=2),encoding='utf8');print(json.dumps(check))
if __name__=='__main__':main()
