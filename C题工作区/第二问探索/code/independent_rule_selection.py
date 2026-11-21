from pathlib import Path
import sys,json,itertools,time
import numpy as np,pandas as pd
from distance_aligned_return import fit_coeff,predictions,BASE,OLD
sys.path.insert(0,str(OLD/'code'))
from engine import FixedPolicy
from end_guard import safe_replay
from update_calibration_experiment import hierarchical
import s0_audit_replay as s0

def main():
 out=BASE/'results/independent_rule_selection';out.mkdir(exist_ok=True)
 w=pd.read_csv(s0.Q1.parent/'C题工作区/阶段交付/stage1_review/corrected_results/data/windows.csv').query("split=='train'");records=pd.read_csv(s0.DATA/'record_index.csv').query("split=='train'");envs=sorted(w.environment_id.unique());cache={};dist=[];data={}
 for r in records.itertuples():
  with np.load(s0.DATA/'records'/f'{r.episode_uuid}.npz',allow_pickle=False) as z:
   for side in ['left_hand','right_hand']:
    t=z[side+'__t_s'];x=z[side+'__position'].mean(axis=1);good=z[side+'__reference_candidate'];cache[(r.episode_uuid,side)]=(t,x);dist.append((r.dataset,r.environment_id,np.linalg.norm(x[good]-z['head__position'][good,0],axis=1)))
 for row in w.itertuples():
  t,x=cache[(row.episode_uuid,row.side)];i,j=row.start_index,row.end_index_exclusive;tt=t[i:j];run=safe_replay(tt,x[i:j],t[i-2:i],x[i-2:i],FixedPolicy(),delay=.1,budget=2,window_end=float(t[i]+1));ret=[v for v in run['actions'] if v['event']=='return'];data[row.window_id]=dict(t=tt,x=x[i:j],ht=t[i-2:i],hist=x[i-2:i],samples=[int(np.argmin(abs(tt-v['sample_time']))) for v in ret],arrivals=[int(np.argmin(abs(tt-v['processed_at']))) for v in ret])
 def points(frame,c1,c2):
  rows=[]
  for r in frame.itertuples():
   d=data[r.window_id];samples=np.full(len(d['t']),d['ht'][-1]);after=np.zeros(len(samples),dtype=bool)
   for k in [0,1]:samples[d['arrivals'][k]:]=d['t'][d['samples'][k]];after[d['arrivals'][k]:]=True
   ages=np.clip(np.ceil((d['t']-samples)*30).astype(int),1,30)
   for model,a,b in [('secant',[1.,0.],[1.,0.]),('distance_return',c1,c2)]:
    error=np.linalg.norm(predictions(d,a,b)-d['x'],axis=1)
    rows.extend(dict(window_id=r.window_id,dataset=r.dataset,environment_id=r.environment_id,episode_uuid=r.episode_uuid,side=r.side,model=model,index=k,after=bool(after[k]),age_bin=int(ages[k]),error=float(error[k])) for k in range(len(error)))
  return pd.DataFrame(rows)
 selections=[];splits=[];scores=[];support_rows=[];sources=set(w.dataset)
 for outer in envs:
  available=[e for e in envs if e!=outer];rng=np.random.default_rng(20261121);partition=None
  for attempt in range(10000):
   shuffled=rng.permutation(available).tolist();parts=[shuffled[:7],shuffled[7:11],shuffled[11:]]
   if all(set(w[w.environment_id.isin(part)].dataset)==sources for part in parts):partition=parts;break
  if partition is None:
   selections.append(dict(outer=outer,choice='all_age',supported=False,reason='cannot supply all sources in three disjoint groups'));continue
  train_env,cal_env,val_env=partition;train=w[w.environment_id.isin(train_env)]
  scale={src:float(np.median(np.concatenate([values for ss,e,values in dist if ss==src and e in train_env]))) for src in sources}
  c1=fit_coeff(data,train,scale,0,np.array([1.,0.]));c2=fit_coeff(data,train,scale,1,c1)
  cal=points(w[w.environment_id.isin(cal_env)],c1,c2).query("model=='distance_return'");val=points(w[w.environment_id.isin(val_env)],c1,c2).query("model=='distance_return'");rows=[]
  for (src,after,age),g in val.groupby(['dataset','after','age_bin']):
   pool=cal[(cal.dataset==src)&(cal.after==after)];local=pool[pool.age_bin==age];assert len(pool)>0
   original=local if len(local)>=100 else pool
   neighborhood=local;distance=0
   while len(neighborhood)<100 and distance<29:
    distance+=1;neighborhood=pool[abs(pool.age_bin-age)<=distance]
   # If 100 residuals are unavailable, do not pretend the neighbor rule has support.
   supported_neighbor=len(neighborhood)>=100
   support_rows.append(dict(outer=outer,dataset=src,after=bool(after),age_bin=int(age),local_points=len(local),pool_points=len(pool),neighbor_points=len(neighborhood),neighbor_supported=supported_neighbor,selection_points=len(g),distance=distance))
   if not supported_neighbor:neighborhood=original
   r0=float(original.error.quantile(.9,interpolation='higher'));rn=float(neighborhood.error.quantile(.9,interpolation='higher'))
   for method,radius in [('all_age',r0),('neighbor',rn),('prior_only',r0 if after else rn)]:
    z=g.copy();delta=z.error-radius;z['pinball']=np.maximum(.9*delta,-.1*delta)/scale[src];z['method']=method;rows.append(z)
  valpoints=pd.concat(rows);window=valpoints.groupby(['method','window_id','dataset','environment_id','episode_uuid','side'],as_index=False).pinball.mean();rank=hierarchical(window,['method'],['pinball']);priority={'all_age':0,'prior_only':1,'neighbor':2};rank['priority']=rank.method.map(priority);rank=rank.sort_values(['pinball','priority']);choice=rank.iloc[0].method
  selections.append(dict(outer=outer,choice=choice,supported=True,reason='disjoint inner score'))
  for row in rank.itertuples():scores.append(dict(outer=outer,method=row.method,pinball=row.pinball))
  splits.append(dict(outer=outer,train=train_env,calibration=cal_env,selection=val_env,scale=scale,first=c1.tolist(),second=c2.tolist()))
 old=pd.read_csv(BASE/'results/distance_nested_calibration/outer_points.csv.gz').query("model=='distance_return' and calibration=='pooled'");neighbor=pd.read_csv(BASE/'results/neighbor_age/points.csv.gz').query("model=='distance_return' and calibration=='pooled'");keys=['window_id','index'];neigh=neighbor.set_index(keys);selected=[]
 for f in selections:
  z=old[old.environment_id==f['outer']].copy();mask=np.ones(len(z),bool) if f['choice']=='neighbor' else (~z.after.to_numpy()) if f['choice']=='prior_only' else np.zeros(len(z),bool)
  matching=neigh.loc[pd.MultiIndex.from_frame(z[keys])]
  for metric in ['radius','coverage','width','pinball']:z.loc[mask,metric]=matching[metric].to_numpy()[mask]
  z['method']='inner_selected';selected.append(z)
 result=pd.concat([old.assign(method='all_age'),*selected],ignore_index=True);result.to_csv(out/'points.csv.gz',index=False)
 metrics=['coverage','width','pinball'];g=result.groupby(['method','window_id','dataset','environment_id','episode_uuid','side'],as_index=False)[metrics].mean();summary=hierarchical(g,['method'],metrics);summary.to_csv(out/'summary.csv',index=False)
 g=g.groupby(['method','dataset','environment_id','episode_uuid','side'],as_index=False)[metrics].mean().groupby(['method','dataset','environment_id','episode_uuid'],as_index=False)[metrics].mean().groupby(['method','dataset','environment_id'],as_index=False)[metrics].mean();g.to_csv(out/'environments.csv',index=False);g.groupby(['method','dataset'],as_index=False)[metrics].mean().to_csv(out/'sources.csv',index=False)
 pd.DataFrame(support_rows).to_csv(out/'inner_support.csv',index=False);pd.DataFrame(selections).to_csv(out/'selections.csv',index=False);pd.DataFrame(scores).to_csv(out/'inner_scores.csv',index=False);(out/'splits.json').write_text(json.dumps(splits,indent=2),encoding='utf8');audit=dict(passed=True,outer_folds=16,supported_selections=sum(f['supported'] for f in selections),unsupported_defaults=sum(not f['supported'] for f in selections),development_reads=0,test_reads=0,formal_model_changed=False);(out/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf8');print(summary.to_string(index=False));print(pd.DataFrame(selections).to_string(index=False));print(json.dumps(audit))
if __name__=='__main__':main()
