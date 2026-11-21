import json
import numpy as np,pandas as pd
from request_value import BASE,OLD

def main():
 out=BASE/'results/purchase_matched_calibration';cal=pd.read_csv(out/'inner_residuals.csv.gz');points=pd.read_csv(out/'outer_points.csv.gz');support=pd.read_csv(out/'support.csv');fits=json.loads((out/'pair_fits.json').read_text());assert len(fits)==120
 for f in fits:assert not set(f['excluded'])&set(f['fit_environments']) and len(f['fit_environments'])==14
 maximum=0.;groups=0
 for outer,ev in points.groupby('environment_id'):
  c=cal[cal.outer==outer];assert outer not in set(c.environment_id)
  for (src,after,age),g in ev.groupby(['dataset','after','age_bin']):
   local=c[(c.dataset==src)&(c.after==after)&(c.age_bin==age)]
   if len(local)<100:local=c[(c.dataset==src)&(c.after==after)]
   ordered=sorted(local.error);r=ordered[int(np.ceil(.9*(len(ordered)-1)))];maximum=max(maximum,float(np.max(abs(g.radius-r))));groups+=1
 assert maximum<1e-12 and len(points)==47037 and not points.duplicated(['window_id','index']).any()
 sc=json.loads((OLD/'results/S0/fold_scales.json').read_text());mean=points.groupby(['window_id','dataset','environment_id']).error.mean();saved=pd.read_csv(BASE/'results/second_request_value/windows.csv.gz').query("method=='adaptive_buy'").set_index('window_id');difference=max(abs(value/sc[e][src]-saved.loc[w,'nade']) for (w,src,e),value in mean.items());assert difference<1e-12
 audit=dict(passed=True,sorted_quantile_groups=groups,radius_max_difference=maximum,outer_nade_max_difference=difference,fit_exclusions_verified=120,scope='saved residual quantiles, pair exclusion metadata and outer metric agreement; independent pair refitting pending');(out/'independent_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf8');print(json.dumps(audit))
if __name__=='__main__':main()
