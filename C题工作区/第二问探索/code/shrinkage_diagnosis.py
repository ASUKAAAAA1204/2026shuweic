"""Train OOF residual-only shrinkage diagnosis; not a promoted model."""
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
BASE=Path(__file__).resolve().parents[1];OLD=BASE.parent/'第二问求解';sys.path.insert(0,str(OLD/'code'))
from update_calibration_experiment import hierarchical,paired_ci

def main():
 p=pd.read_csv(OLD/'results/S3/calibration_points.csv.gz');keys=['window_id','episode_uuid','dataset','environment_id','side','relative'];a=p[p.calibrator=='open_radius'];b=p[p.calibrator=='separate'];g=a.merge(b[keys+['radius']],on=keys,validate='one_to_one',suffixes=('','_post'));rows=[]
 for weight in [0.,.25,.5,.75,1.]:
  z=g.copy();radius=z.radius+weight*(z.radius_post-z.radius);diff=z.error-radius;z['coverage']=(diff<=0).astype(float);z['width']=2*radius/z.scale;z['pinball']=np.maximum(.9*diff,-.1*diff)/z.scale;z['weight']=weight
  z=z[z.after];w=z.groupby(['weight','window_id','episode_uuid','dataset','environment_id','side'],as_index=False)[['coverage','width','pinball']].mean();rows.append(w)
 m=pd.concat(rows,ignore_index=True);out=BASE/'results/calibration_shrinkage';out.mkdir(exist_ok=True);m.to_csv(out/'oof_windows.csv',index=False);summary=hierarchical(m,['weight'],['coverage','width','pinball']);summary.to_csv(out/'summary.csv',index=False)
 # Fixed pre-registered candidates, paired intervals reported for each; no cherry-picked winner.
 intervals=[]
 for weight in [.25,.5,.75,1.]:
  q=m[m.weight.isin([0.,weight])].copy();q['method']=np.where(q.weight==0,'original','shrink');env=q.groupby(['method','dataset','environment_id','episode_uuid','side'],as_index=False).pinball.mean().groupby(['method','dataset','environment_id','episode_uuid'],as_index=False).pinball.mean().groupby(['method','dataset','environment_id'],as_index=False).pinball.mean();ci=paired_ci(env,'original','shrink',metric='pinball');ci['weight']=weight;intervals.append(ci)
 pd.DataFrame(intervals).to_csv(out/'paired_intervals.csv',index=False);(out/'audit.json').write_text(json.dumps(dict(train_only=True,test_reads=0,development_reads=0,fit='all source quantiles inherited from held-environment-excluded fits',candidate_selection=False,warning='multiple exploratory comparisons; no promotion and no new generalization claim'),indent=2),encoding='utf8');print(summary.to_string(index=False));print(pd.DataFrame(intervals)[['weight','gain_pct','lo95','hi95']].to_string(index=False))
if __name__=='__main__':main()
