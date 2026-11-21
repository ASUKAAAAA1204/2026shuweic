"""Global-environment contribution to post-return calibration loss."""
from pathlib import Path
import pandas as pd
BASE=Path(__file__).resolve().parents[1];OLD=BASE.parent/'第二问求解';out=BASE/'results/calibration_environment';out.mkdir(exist_ok=True)
p=pd.read_csv(OLD/'results/S3/calibration_environments.csv').query("region=='after'");a=p[p.calibrator=='open_radius'];b=p[p.calibrator=='separate'];m=a.merge(b,on=['region','dataset','environment_id'],suffixes=('_open','_post'),validate='one_to_one');m['pinball_change']=m.pinball_post-m.pinball_open;m['relative_gain_pct']=100*(1-m.pinball_post/m.pinball_open);m['coverage_gain']=m.covered_post-m.covered_open;m.to_csv(out/'paired_source_environments.csv',index=False)
print(m[['dataset','environment_id','relative_gain_pct','coverage_gain']].sort_values('relative_gain_pct').head(7).to_string(index=False))
