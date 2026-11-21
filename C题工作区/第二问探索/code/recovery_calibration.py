from pathlib import Path
import sys,json
import numpy as np,pandas as pd
BASE=Path(__file__).resolve().parents[1];OLD=BASE.parent/'第二问求解';sys.path.insert(0,str(OLD/'code'))
from update_calibration_experiment import hierarchical

def main():
 out=BASE/'results/recovery_calibration';out.mkdir(exist_ok=True);p=pd.read_csv(BASE/'results/natural_recovery/points.csv.gz').query("method=='stream' and length_s==0.2");meta=pd.read_csv(BASE/'results/natural_recovery/windows.csv').query("method=='stream' and length_s==0.2")
 p=p.merge(meta[['event','dataset','environment_id','episode_uuid','side']],on='event',validate='many_to_one');p['phase']=np.where(p.relative<1e-12,'first','flow');p['window_id']=p.event;sc=json.loads((OLD/'results/S0/fold_scales.json').read_text());p['raw_error']=[e*sc[env][s] for e,env,s in zip(p.pre_error,p.environment_id,p.dataset)];rows=[];folds=[]
 for held in sorted(p.environment_id.unique()):
  fit=p[p.environment_id!=held].copy();ev=p[p.environment_id==held].copy()
  for frame in [fit,ev]:frame['error']=[v/sc[held][s] for v,s in zip(frame.raw_error,frame.dataset)]
  for method in ['pooled','phase']:
   for (src,phase),g in ev.groupby(['dataset','phase']):
    local=fit[fit.dataset==src] if method=='pooled' else fit[(fit.dataset==src)&(fit.phase==phase)]
    fallback=local.environment_id.nunique()<2 or len(local)<20
    if fallback:local=fit if method=='pooled' else fit[fit.phase==phase]
    support=local.environment_id.nunique();calibratable=support>=2 and len(local)>=20
    folds.append(dict(held=held,method=method,dataset=src,phase=phase,points=len(local),environments=support,fallback=fallback,calibratable=calibratable))
    z=g.copy();r=float(local.error.quantile(.9,interpolation='higher')) if calibratable else np.nan;z['radius']=r;z['coverage']=(z.error<=r).astype(float) if calibratable else np.nan;z['pinball']=np.maximum(.9*(z.error-r),-.1*(z.error-r));z['width']=2*r;z['fallback']=fallback;z['method']=method;rows.append(z)
 a=pd.concat(rows,ignore_index=True);a.to_csv(out/'points.csv.gz',index=False);pd.DataFrame(folds).to_csv(out/'support.csv',index=False);w=a.groupby(['method','phase','window_id','dataset','environment_id','episode_uuid','side'],as_index=False)[['coverage','width','pinball','fallback']].mean();w.to_csv(out/'windows.csv',index=False)
 summary=hierarchical(w,['method','phase'],['coverage','width','pinball','fallback']);summary.to_csv(out/'summary.csv',index=False);src=w.groupby(['method','phase','dataset','environment_id','episode_uuid','side'],as_index=False)[['coverage','width','pinball']].mean().groupby(['method','phase','dataset','environment_id','episode_uuid'],as_index=False)[['coverage','width','pinball']].mean().groupby(['method','phase','dataset','environment_id'],as_index=False)[['coverage','width','pinball']].mean();src.to_csv(out/'environments.csv',index=False)
 audit=dict(held_envs=p.environment_id.nunique(),hand_events=p.event.nunique(),records=p.episode_uuid.nunique(),uncalibratable_groups=int((~pd.DataFrame(folds).calibratable).sum()),development_reads=0,test_reads=0,model_promoted=False,interpretation='exploratory natural zero-delay recovered stream, not paid single-point return')
 (out/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf8');print(summary.to_string(index=False));print(json.dumps(audit))
if __name__=='__main__':main()
