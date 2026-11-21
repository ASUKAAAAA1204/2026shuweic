from pathlib import Path
import json
import numpy as np,pandas as pd
BASE=Path(__file__).resolve().parents[1];p=BASE/'results/nonconvex_return'
systems=json.loads((p/'fit_systems.json').read_text());folds=json.loads((p/'folds.json').read_text());assert len(systems)==32 and len(folds)==16
mx=0.
for i,fold in enumerate(folds):
 assert fold['held'] not in fold['fit_environments']
 for stage in [0,1]:
  system=systems[2*i+stage];assert system['stage']==stage
  G=np.array(system['gram']);r=np.array(system['rhs']);A=G+.001*np.eye(2);b=r+.001*np.array([1.,0.]);assert np.linalg.eigvalsh(G).min()>-1e-12
  # Eigenbasis solution is independent of the implementation's direct solve.
  val,vec=np.linalg.eigh(A);coef=vec@((vec.T@b)/val)
  expected=np.array([fold[('first' if stage==0 else 'second')+'_a'],fold[('first' if stage==0 else 'second')+'_b']]);delta=float(abs(coef-expected).max());mx=max(mx,delta);assert delta<1e-10
m=pd.read_csv(p/'windows.csv.gz');old=pd.read_csv(BASE/'results/calibration_closed_loop/windows.csv.gz').query("method=='fixed'")
a=m.query("method=='secant'").merge(old,on='window_id',suffixes=('_new','_old'),validate='one_to_one');assert len(a)==1527
for col in ['nade','nfde','exceed05','requests','waiting_s']:assert np.allclose(a[col+'_new'],a[col+'_old'],atol=1e-12)
params=pd.DataFrame([{k:v for k,v in f.items() if k!='fit_environments'} for f in folds]);params.to_csv(p/'coefficients.csv',index=False)
audit=dict(passed=True,systems=32,held_exclusion_verified=True,eigenbasis_coefficient_max_difference=mx,baseline_matches_previous_windows=1527,scope='saved sufficient-statistic equations and baseline comparison; not independent raw-data refit')
(p/'independent_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf8');print(json.dumps(audit));print(params.drop(columns='held').agg(['min','max']).to_string())
