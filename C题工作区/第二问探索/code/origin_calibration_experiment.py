"""Reproduce the exploratory origin calibration without new parameter search."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
BASE=Path(__file__).resolve().parents[1]
OLD=BASE.parent/'第二问求解'

def aggregate(p):
    metrics=['coverage','width','pinball','fallback']
    keys=['origin','dataset','environment_id','episode_uuid','side']
    g=p.groupby(keys,as_index=False)[metrics].mean()
    for keys in [keys[:-1],keys[:-2],['origin','dataset'],['origin']]:
        g=g.groupby(keys,as_index=False)[metrics].mean()
    return g

def main():
    previous=BASE/'results/initialization_split'
    out=BASE/'results/origin_reproduction';out.mkdir(exist_ok=True)
    p=pd.read_csv(previous/'labeled_residuals.csv.gz').query("mode=='recovered_single'").copy()
    scales=json.loads((OLD/'results/S0/fold_scales.json').read_text())
    rows=[];support=[]
    for held in sorted(p.environment_id.unique()):
        fit=p[p.environment_id!=held].copy();ev=p[p.environment_id==held].copy()
        for frame in [fit,ev]:
            frame['error']=[e/scales[held][s] for e,s in zip(frame.raw_error,frame.dataset)]
        for (src,origin),g in ev.groupby(['dataset','origin']):
            local=fit[(fit.dataset==src)&(fit.origin==origin)]
            fallback=len(local)<20 or local.environment_id.nunique()<2
            if fallback:local=fit[fit.origin==origin]
            supported=len(local)>=20 and local.environment_id.nunique()>=2
            assert held not in set(local.environment_id)
            r=float(np.quantile(local.error,.9,method='higher')) if supported else np.nan
            support.append(dict(held=held,dataset=src,origin=origin,points=len(local),envs=local.environment_id.nunique(),fallback=fallback,supported=supported,radius=r))
            z=g.copy();d=z.error-r
            z['coverage']=(d<=0).astype(float) if supported else np.nan
            z['width']=2*r;z['pinball']=np.maximum(.9*d,-.1*d);z['fallback']=float(fallback);rows.append(z)
    a=pd.concat(rows,ignore_index=True);summary=aggregate(a)
    expected=pd.read_csv(previous/'origin_calibration_summary.csv').sort_values('origin')
    actual=summary.sort_values('origin')
    metrics=['coverage','width','pinball','fallback']
    delta=float(np.max(np.abs(actual[metrics].to_numpy()-expected[metrics].to_numpy())))
    assert delta<1e-12,delta
    # Independent reconstruction of the quantile from sorted training residuals.
    for s in support:
        subset=p[(p.environment_id!=s['held'])&(p.origin==s['origin'])]
        if not s['fallback']:subset=subset[subset.dataset==s['dataset']]
        values=sorted(e/scales[s['held']][src] for e,src in zip(subset.raw_error,subset.dataset))
        order=math.ceil(.9*(len(values)-1))
        assert abs(values[order]-s['radius'])<1e-12
    a.to_csv(out/'points.csv.gz',index=False);pd.DataFrame(support).to_csv(out/'support.csv',index=False);summary.to_csv(out/'summary.csv',index=False)
    audit=dict(passed=True,summary_max_abs_difference=delta,points=len(a),held_environments=p.environment_id.nunique(),support_groups=len(support),unsupported=sum(not s['supported'] for s in support),independent_order_statistic_check=True,development_reads=0,test_reads=0,exploratory_not_preregistered=True)
    (out/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf8');print(json.dumps(audit))
import math
if __name__=='__main__':main()
