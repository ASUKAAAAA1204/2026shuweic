from pathlib import Path
import sys,json
import numpy as np,pandas as pd
BASE=Path(__file__).resolve().parents[1];OLD=BASE.parent/'第二问求解';sys.path.insert(0,str(OLD/'code'))
import s0_audit_replay as s0
from recovery_interface import RecoveryTracker

def main():
 out=BASE/'results/causal_calibration';out.mkdir(exist_ok=True);rec=pd.read_csv(s0.DATA/'record_index.csv').query("split=='train'");sc=json.loads((OLD/'results/S0/fold_scales.json').read_text());rows=[];counts=[]
 for r in rec.itertuples():
  with np.load(s0.DATA/'records'/f'{r.episode_uuid}.npz',allow_pickle=False) as z:
   for side in ['left_hand','right_hand']:
    t=z[side+'__t_s'];x=z[side+'__position'].mean(axis=1);good=z[side+'__reference_candidate'];tracker=RecoveryTracker(max_gap_s=3*float(np.median(np.diff(t))))
    for k,now in enumerate(t):
     pred=tracker.predict(float(now));e=float(np.linalg.norm(np.asarray(pred.position)-x[k])) if good[k] and pred.position is not None else np.nan
     counts.append(dict(dataset=r.dataset,environment_id=r.environment_id,episode_uuid=r.episode_uuid,side=side,mode=pred.mode,qualified=bool(good[k]),prediction_exists=pred.position is not None))
     if np.isfinite(e):rows.append(dict(dataset=r.dataset,environment_id=r.environment_id,episode_uuid=r.episode_uuid,side=side,index=k,mode=pred.mode,age=pred.observation_age,raw_error=e))
     tracker.receive(float(now),x[k] if good[k] else None,bool(good[k]),float(now))
 p=pd.DataFrame(rows);pd.DataFrame(counts).groupby(['dataset','mode','qualified','prediction_exists']).size().to_csv(out/'retention.csv');p.to_csv(out/'raw_residuals.csv.gz',index=False);res=[];support=[]
 for held in sorted(rec.environment_id.unique()):
  fit=p[p.environment_id!=held].copy();ev=p[p.environment_id==held].copy()
  for frame in [fit,ev]:frame['error']=[v/sc[held][s] for v,s in zip(frame.raw_error,frame.dataset)]
  for method in ['source_pooled','source_mode']:
   for (src,mode),g in ev.groupby(['dataset','mode']):
    local=fit[fit.dataset==src] if method=='source_pooled' else fit[(fit.dataset==src)&(fit['mode']==mode)];fallback=local.environment_id.nunique()<2 or len(local)<20
    if fallback:local=fit if method=='source_pooled' else fit[fit['mode']==mode]
    supported=local.environment_id.nunique()>=2 and len(local)>=20;r=float(local.error.quantile(.9,interpolation='higher')) if supported else np.nan;support.append(dict(held=held,method=method,dataset=src,mode=mode,points=len(local),environments=local.environment_id.nunique(),fallback=fallback,supported=supported))
    a=g.copy();a['radius']=r;a['coverage']=(a.error<=r).astype(float) if supported else np.nan;a['width']=2*r;d=a.error-r;a['pinball']=np.maximum(.9*d,-.1*d);a['method']=method;a['fallback']=fallback;a['supported']=supported;a['tolerance_candidate']=(r<=.05) if supported else False;res.append(a)
 a=pd.concat(res,ignore_index=True);a.to_csv(out/'oof_points.csv.gz',index=False);pd.DataFrame(support).to_csv(out/'support.csv',index=False);metrics=['coverage','width','pinball','fallback','supported','tolerance_candidate'];g=a.groupby(['method','mode','dataset','environment_id','episode_uuid','side'],as_index=False)[metrics].mean()
 for keys in [['method','mode','dataset','environment_id','episode_uuid'],['method','mode','dataset','environment_id'],['method','mode','dataset'],['method','mode']]:g=g.groupby(keys,as_index=False)[metrics].mean()
 g.to_csv(out/'summary.csv',index=False);audit=dict(train_records=len(rec),train_envs=rec.environment_id.nunique(),qualified_prediction_points=len(p),all_hand_timestamps=len(counts),unqualified_or_uninitialized=len(counts)-len(p),test_reads=0,development_reads=0,mode_is_causal=True,unsupported_groups=int((~pd.DataFrame(support).supported).sum()),formal_model_replaced=False)
 (out/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf8');print(json.dumps(audit));print(g.to_string(index=False))
if __name__=='__main__':main()
