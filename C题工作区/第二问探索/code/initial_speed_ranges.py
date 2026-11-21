import json
import numpy as np,pandas as pd
from request_value import BASE,OLD
from update_calibration_experiment import hierarchical
import s0_audit_replay as s0

def main():
 out=BASE/'results/initial_speed_ranges';out.mkdir(exist_ok=True)
 w=pd.read_csv(s0.Q1.parent/'C题工作区/阶段交付/stage1_review/corrected_results/data/windows.csv').query("split=='train'");sc=json.loads((OLD/'results/S0/fold_scales.json').read_text());p=pd.read_csv(BASE/'results/validation_matched_ranges/training_oof_points.csv.gz').query("method=='baseline' and after==False").copy();cache={};features=[]
 for r in w.itertuples():
  key=(r.episode_uuid,r.side)
  if key not in cache:
   with np.load(s0.DATA/'records'/f'{r.episode_uuid}.npz',allow_pickle=False) as z:cache[key]=(z[r.side+'__t_s'],z[r.side+'__position'].mean(axis=1))
  t,x=cache[key];i=r.start_index;features.append(dict(window_id=r.window_id,speed=float(np.linalg.norm((x[i-1]-x[i-2])/(t[i-1]-t[i-2])))))
 p=p.merge(pd.DataFrame(features),on='window_id',validate='many_to_one');p['error']=p.nade*np.array([sc[e][s] for e,s in zip(p.environment_id,p.dataset)]);results=[];support=[]
 for held in sorted(w.environment_id.unique()):
  train=p[p.environment_id!=held].copy();ev=p[p.environment_id==held].copy();scale=sc[held]
  for f in [train,ev]:f['err']=f.error/np.array([scale[s] for s in f.dataset]);f['speed_normalized']=f.speed/np.array([scale[s] for s in f.dataset])
  cuts={src:np.quantile(g.drop_duplicates('window_id').speed_normalized,[1/3,2/3]) for src,g in train.groupby('dataset')}
  for f in [train,ev]:f['speed_group']=[int(np.searchsorted(cuts[s],v,side='right')) for s,v in zip(f.dataset,f.speed_normalized)]
  for (src,age,group),g in ev.groupby(['dataset','age_bin','speed_group']):
   same=train[train.dataset==src];base=same[same.age_bin==age];basefallback=len(base)<100
   if basefallback:base=same
   assert len(base)>0
   local=base[base.speed_group==group];fallback=len(local)<100 or local.environment_id.nunique()<2
   if fallback:local=base
   for method,cal in [('source_age',base),('source_age_speed',local)]:
    radius=float(cal.err.quantile(.9,interpolation='higher'));z=g.copy();delta=z.err-radius;z['method']=method;z['radius']=radius;z['coverage']=(delta<=0).astype(float);z['width']=2*radius;z['pinball']=np.maximum(.9*delta,-.1*delta);z['fallback']=float(fallback if method=='source_age_speed' else basefallback);results.append(z)
    support.append(dict(held=held,dataset=src,age_bin=int(age),speed_group=int(group),method=method,radius=radius,points=len(cal),environments=cal.environment_id.nunique(),fallback=fallback if method=='source_age_speed' else basefallback,cut1=float(cuts[src][0]),cut2=float(cuts[src][1])))
 result=pd.concat(results,ignore_index=True);result.to_csv(out/'points.csv.gz',index=False);pd.DataFrame(support).to_csv(out/'support.csv',index=False);metrics=['coverage','width','pinball','fallback'];q=result.groupby(['method','window_id','dataset','environment_id','episode_uuid','side'],as_index=False)[metrics].mean();summary=hierarchical(q,['method'],metrics);summary.to_csv(out/'summary.csv',index=False);e=q.groupby(['method','dataset','environment_id','episode_uuid','side'],as_index=False)[metrics].mean().groupby(['method','dataset','environment_id','episode_uuid'],as_index=False)[metrics].mean().groupby(['method','dataset','environment_id'],as_index=False)[metrics].mean();e.to_csv(out/'environments.csv',index=False);e.groupby(['method','dataset'],as_index=False)[metrics].mean().to_csv(out/'sources.csv',index=False)
 assert not result.duplicated(['method','window_id','index']).any();assert result.groupby('method').window_id.nunique().eq(len(w)).all();audit=dict(train_windows=len(w),training_records=len({k[0] for k in cache}),training_environments=w.environment_id.nunique(),point_rows=len(result),validation_reads=0,test_reads=0,formal_model_changed=False,scope='fixed three speed strata; before first return only; exploratory environment holdout');(out/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf8');print(summary.to_string(index=False));print(json.dumps(audit))
if __name__=='__main__':main()
