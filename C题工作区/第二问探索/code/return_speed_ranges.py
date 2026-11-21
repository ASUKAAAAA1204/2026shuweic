import json
import numpy as np,pandas as pd
from request_value import BASE,OLD
from candidate_replay_engine import replay,FixedPolicy
from update_calibration_experiment import hierarchical
import s0_audit_replay as s0

def main():
 out=BASE/'results/return_speed_ranges';out.mkdir(exist_ok=True);w=pd.read_csv(s0.Q1.parent/'C题工作区/阶段交付/stage1_review/corrected_results/data/windows.csv').query("split=='train'");sc=json.loads((OLD/'results/S0/fold_scales.json').read_text());cache={};rows=[]
 for r in w.itertuples():
  key=(r.episode_uuid,r.side)
  if key not in cache:
   with np.load(s0.DATA/'records'/f'{r.episode_uuid}.npz',allow_pickle=False) as z:cache[key]=(z[r.side+'__t_s'],z[r.side+'__position'].mean(axis=1))
  t,x=cache[key];i,j=r.start_index,r.end_index_exclusive;run=replay(t[i:j],x[i:j],t[i-2:i],x[i-2:i],FixedPolicy(),delay=.1,budget=2,window_end=float(t[j-1]));events=[a for a in run['actions'] if a['event']=='return'];speeds={};last=x[i-1];last_t=t[i-1]
  for stage,e in enumerate(events,1):
   sample=e['sample_time'];k=int(np.argmin(abs(t-sample)));assert abs(t[k]-sample)<1e-9;speeds[stage]=(float(np.linalg.norm((x[k]-last)/(sample-last_t))),sample,e['processed_at']);last=x[k];last_t=sample
  for index,p in enumerate(run['points']):
   if not p['returns']:continue
   stage=p['returns'];speed,sample,arrived=speeds[stage];assert sample<=arrived<=p['time']+1e-12
   rows.append(dict(**{k:getattr(r,k) for k in ['window_id','dataset','environment_id','episode_uuid','side']},index=index,stage=stage,time=p['time'],sample_time=sample,available_at=arrived,speed=speed,error=p['error'],age_bin=int(np.clip(np.ceil(p['age']*30),1,30))))
 raw=pd.DataFrame(rows);raw.to_csv(out/'raw_points.csv.gz',index=False);results=[];support=[]
 for held in sorted(w.environment_id.unique()):
  train=raw[raw.environment_id!=held].copy();ev=raw[raw.environment_id==held].copy()
  for f in [train,ev]:f['err']=f.error/f.dataset.map(sc[held]);f['speed_normalized']=f.speed/f.dataset.map(sc[held])
  cuts={s:np.quantile(g.drop_duplicates(['window_id','stage']).speed_normalized,[1/3,2/3]) for s,g in train.groupby('dataset')}
  for f in [train,ev]:f['speed_group']=[int(np.searchsorted(cuts[s],v,side='right')) for s,v in zip(f.dataset,f.speed_normalized)]
  for (src,age,group),g in ev.groupby(['dataset','age_bin','speed_group']):
   pool=train[train.dataset==src];base=pool[pool.age_bin==age];bf=len(base)<100
   if bf:base=pool
   local=base[base.speed_group==group];fallback=len(local)<100 or local.environment_id.nunique()<2
   if fallback:local=base
   for method,cal,fb in [('source_age',base,bf),('source_age_speed',local,fallback)]:
    radius=float(cal.err.quantile(.9,interpolation='higher'));z=g.copy();delta=z.err-radius;z['method']=method;z['radius']=radius;z['coverage']=(delta<=0).astype(float);z['width']=2*radius;z['pinball']=np.maximum(.9*delta,-.1*delta);z['fallback']=float(fb);results.append(z);support.append(dict(held=held,dataset=src,age_bin=int(age),speed_group=int(group),method=method,radius=radius,points=len(cal),environments=cal.environment_id.nunique(),fallback=fb,cut1=float(cuts[src][0]),cut2=float(cuts[src][1])))
 result=pd.concat(results,ignore_index=True);result.to_csv(out/'points.csv.gz',index=False);pd.DataFrame(support).to_csv(out/'support.csv',index=False);metrics=['coverage','width','pinball','fallback'];q=result.groupby(['method','window_id','dataset','environment_id','episode_uuid','side'],as_index=False)[metrics].mean();summary=hierarchical(q,['method'],metrics);summary.to_csv(out/'summary.csv',index=False);env=q.groupby(['method','dataset','environment_id','episode_uuid','side'],as_index=False)[metrics].mean().groupby(['method','dataset','environment_id','episode_uuid'],as_index=False)[metrics].mean().groupby(['method','dataset','environment_id'],as_index=False)[metrics].mean();env.to_csv(out/'environments.csv',index=False);env.groupby(['method','dataset'],as_index=False)[metrics].mean().to_csv(out/'sources.csv',index=False)
 audit=dict(train_windows=len(w),raw_points=len(raw),rows=len(result),validation_reads=0,test_reads=0,formal_changed=False,information_boundary='speed available only at processed return; sample age retains delay');(out/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf8');print(summary.to_string(index=False));print(json.dumps(audit))
if __name__=='__main__':main()
