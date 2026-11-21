"""Frozen family-distribution uncertainty audit on unique validation units."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]

def main():
    model=pd.DataFrame(json.loads((ROOT/'model/training_family_samples.json').read_text()));data=pd.read_csv(ROOT.parent/'处理结果/v1/segments_private.csv');data=data[(data.split=='validation')&(data.eligible==1)]
    rows=[];parameters=[]
    for family,g in model.groupby('family_id'):
        samples=g.duration_s.to_numpy();lo,med,hi=np.quantile(samples,[.1,.5,.9]);parameters.append(dict(family=family,training_units=len(samples),q10=float(lo),median=float(med),q90=float(hi)))
        for r in data[data.family_id==family].itertuples():
            error=r.duration_s-hi;rows.append(dict(environment_id=r.environment_id,unit_id=r.unit_id,family=family,mae=abs(r.duration_s-med),coverage80=float(lo<=r.duration_s<=hi),width=hi-lo,pinball90=max(.9*error,-.1*error)))
    out=ROOT/'results/duration_uncertainty';out.mkdir(parents=True,exist_ok=True);d=pd.DataFrame(rows);d.to_csv(out/'units.csv',index=False);pd.DataFrame(parameters).to_csv(out/'parameters.csv',index=False)
    summary=d.groupby('environment_id')[['mae','coverage80','width','pinball90']].mean().mean().to_dict();summary['unique_units']=len(d);summary['environments']=d.environment_id.nunique();summary['nominal_coverage']=.8;summary['coverage_guaranteed']=False
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8');print(summary)
if __name__=='__main__':main()
