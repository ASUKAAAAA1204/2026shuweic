from pathlib import Path
import sys,json
import pandas as pd
BASE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(BASE.parent/'第二问求解/code'));from update_calibration_experiment import paired_ci
p=BASE/'results/distance_nested_calibration';e=pd.read_csv(p/'after_return_environments.csv');e['method']=e.model+'_'+e.calibration;rows=[]
for c in ['pooled','env_mean']:
 r=paired_ci(e,'secant_'+c,'distance_return_'+c,metric='pinball');r['calibration']=c;rows.append(r)
pd.DataFrame(rows).to_csv(p/'after_return_pinball_ci.csv',index=False);print(pd.DataFrame(rows).to_string(index=False))
