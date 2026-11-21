"""Frozen family duration evaluation on validation units; no model selection."""
from pathlib import Path
import json
import pandas as pd
BASE=Path(__file__).resolve().parents[1]
def main():
 model=json.loads((BASE/'model/family_duration.json').read_text());d=pd.read_csv(BASE.parent/'处理结果/v1/segments_private.csv');d=d[(d.split=='validation')&(d.eligible==1)];rows=[]
 for r in d.itertuples():
  m=model[r.family_id];lo,mid,hi=[m[str(q)] for q in [.1,.5,.9]];delta=r.duration_s-hi;rows.append(dict(unit_id=r.unit_id,environment_id=r.environment_id,dataset=r.dataset,family=r.family_id,actual_duration=r.duration_s,predicted_median=mid,mae=abs(r.duration_s-mid),coverage80=float(lo<=r.duration_s<=hi),width80=hi-lo,pinball90=max(.9*delta,-.1*delta)))
 p=pd.DataFrame(rows);out=BASE/'results/duration_information';p.to_csv(out/'validation_units.csv',index=False);env=p.groupby('environment_id')[['mae','coverage80','width80','pinball90']].mean();env.to_csv(out/'validation_environments.csv');env.mean().to_csv(out/'validation_summary.csv',header=['value']);p.groupby(['family','environment_id'])[['mae','coverage80','width80','pinball90']].mean().groupby('family').mean().to_csv(out/'validation_families.csv');print(env.mean().to_string())
if __name__=='__main__':main()
