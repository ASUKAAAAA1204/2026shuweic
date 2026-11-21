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
 out=BASE/'results/distance_nested_calibration';out.mkdir(exist_ok=True)
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
 calibrations=[];fits=[]
 pairs=list(itertools.combinations(envs,2));start=time.perf_counter()
 for number,(a,b) in enumerate(pairs,1):
  train=w[~w.environment_id.isin([a,b])];scale={src:float(np.median(np.concatenate([values for s,e,values in dist if s==src and e not in {a,b}]))) for src in sorted(w.dataset.unique())}
  c1=fit_coeff(data,train,scale,0,np.array([1.,0.]));c2=fit_coeff(data,train,scale,1,c1)
  fits.append(dict(excluded=[a,b],fit_environments=sorted(train.environment_id.unique()),scale=scale,first=c1.tolist(),second=c2.tolist()))
  for outer,inner in [(a,b),(b,a)]:
   z=points(w[w.environment_id==inner],c1,c2);z['outer']=outer;calibrations.append(z)
  if number%20==0:print(f'pair fits {number}/{len(pairs)} elapsed {time.perf_counter()-start:.1f}s',flush=True)
 cal=pd.concat(calibrations,ignore_index=True);cal.to_csv(out/'inner_residuals.csv.gz',index=False);(out/'pair_fits.json').write_text(json.dumps(fits,indent=2),encoding='utf8')
 outerfits={f['held']:f for f in json.loads((BASE/'results/distance_aligned_return/folds.json').read_text())};sc=json.loads((OLD/'results/S0/fold_scales.json').read_text());evaluations=[];support=[]
 for outer in envs:
  f=outerfits[outer];ev=points(w[w.environment_id==outer],[f['first_a'],f['first_b']],[f['second_a'],f['second_b']]);fit=cal[cal.outer==outer];assert outer not in set(fit.environment_id)
  for (model,src,after,age),g in ev.groupby(['model','dataset','after','age_bin']):
   local=fit[(fit.model==model)&(fit.dataset==src)&(fit.after==after)&(fit.age_bin==age)];fallback=len(local)<100
   if fallback:local=fit[(fit.model==model)&(fit.dataset==src)&(fit.after==after)]
   assert len(local)>0
   pooled=float(local.error.quantile(.9,interpolation='higher'));ee=[float(z.error.quantile(.9,interpolation='higher')) for _,z in local.groupby('environment_id') if len(z)>=20];radius={'pooled':pooled,'env_mean':float(np.mean(ee)) if len(ee)>=2 else pooled}
   support.append(dict(outer=outer,model=model,dataset=src,after=after,age_bin=age,points=len(local),environments=local.environment_id.nunique(),age_fallback=fallback,env_mean_fallback=len(ee)<2))
   for method,r in radius.items():
    z=g.copy();z['calibration']=method;delta=z.error-r;z['radius']=r;z['coverage']=(delta<=0).astype(float);z['width']=2*r/sc[outer][src];z['pinball']=np.maximum(.9*delta,-.1*delta)/sc[outer][src];evaluations.append(z)
 result=pd.concat(evaluations,ignore_index=True);result.to_csv(out/'outer_points.csv.gz',index=False);pd.DataFrame(support).to_csv(out/'support.csv',index=False)
 summaries=[]
 for scope,group in [('all',result),('after_return',result[result.after])]:
  group=group.groupby(['model','calibration','window_id','dataset','environment_id','episode_uuid','side'],as_index=False)[['coverage','width','pinball']].mean();summary=hierarchical(group,['model','calibration'],['coverage','width','pinball']);summary['scope']=scope;summaries.append(summary)
  group=group.groupby(['model','calibration','dataset','environment_id','episode_uuid','side'],as_index=False)[['coverage','width','pinball']].mean().groupby(['model','calibration','dataset','environment_id','episode_uuid'],as_index=False)[['coverage','width','pinball']].mean().groupby(['model','calibration','dataset','environment_id'],as_index=False)[['coverage','width','pinball']].mean();group['scope']=scope;group.to_csv(out/(scope+'_environments.csv'),index=False);group.groupby(['model','calibration','dataset'],as_index=False)[['coverage','width','pinball']].mean().to_csv(out/(scope+'_sources.csv'),index=False)
 summary=pd.concat(summaries,ignore_index=True);summary.to_csv(out/'summary.csv',index=False);audit=dict(passed=True,pair_fits=len(fits),directed_inner_folds=2*len(fits),outer_environments=len(envs),inner_points=len(cal),outer_rows=len(result),test_reads=0,development_reads=0,formal_model_changed=False,coverage_guarantee=False,elapsed_s=time.perf_counter()-start);(out/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf8');print(summary.to_string(index=False));print(json.dumps(audit))
if __name__=='__main__':main()
