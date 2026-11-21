"""Precompute six fixed schedules; nested environment selection without fitting motion."""
from pathlib import Path
import json,time
import numpy as np
import pandas as pd
from engine import replay,FixedPolicy
from update_calibration_experiment import hierarchical,paired_ci
import s0_audit_replay as s0
BASE=Path(__file__).resolve().parents[1];OUT=BASE/'results/S4_fixed'

def main():
 OUT.mkdir(parents=True,exist_ok=True);start=time.perf_counter();reg=json.loads((BASE/'S4实验注册.json').read_text(encoding='utf-8-sig'));schedules=reg['fixed_schedules_s'];windows=pd.read_csv(s0.Q1.parent/'C题工作区/阶段交付/stage1_review/corrected_results/data/windows.csv').query("split=='train'")
 records=pd.read_csv(s0.DATA/'record_index.csv').query("split=='train'");cache={};distances=[]
 for r in records.itertuples():
  with np.load(s0.DATA/'records'/f'{r.episode_uuid}.npz',allow_pickle=False) as z:
   for side in ['left_hand','right_hand']:
    t=z[side+'__t_s'];x=z[side+'__position'].mean(axis=1);valid=z[side+'__reference_candidate'];head=z['head__position'][:,0];cache[(r.episode_uuid,side)]=(t,x)
    distances.append((r.dataset,r.environment_id,np.linalg.norm(x[valid]-head[valid],axis=1)))
 envs=sorted(windows.environment_id.unique());sources=sorted(windows.dataset.unique());scale_cache={}
 def scales(exclude):
  key=tuple(sorted(exclude))
  if key not in scale_cache:scale_cache[key]={s:float(np.median(np.concatenate([a for src,e,a in distances if src==s and e not in exclude]))) for s in sources}
  return scale_cache[key]
 rows=[]
 for w in windows.itertuples():
  t,x=cache[(w.episode_uuid,w.side)];i,j=w.start_index,w.end_index_exclusive
  for schedule in schedules:
   r=replay(t[i:j],x[i:j],t[i-2:i],x[i-2:i],FixedPolicy(schedule),delay=.1,window_end=float(t[i]+1))
   rows.append({**{k:getattr(w,k) for k in ['window_id','episode_uuid','dataset','environment_id','side']},'schedule':str(schedule),**r['metrics']})
 raw=pd.DataFrame(rows);raw.to_csv(OUT/'all_candidate_windows.csv',index=False)
 def evaluate(fit,exclude_outer):
  f=fit.copy();f['nade']=[a/scales(set(exclude_outer)|{e})[s] for a,e,s in zip(f.ade,f.environment_id,f.dataset)]
  return hierarchical(f,['schedule'],['nade']).sort_values(['nade','schedule'])
 folds=[];oof=[]
 for outer in envs:
  inner=raw[raw.environment_id!=outer];rank=evaluate(inner,{outer});schedule=rank.iloc[0].schedule
  f=raw[(raw.environment_id==outer)&(raw.schedule==schedule)].copy();f['method']='tuned_fixed';f['nade']=[a/scales({outer})[s] for a,s in zip(f.ade,f.dataset)];oof.append(f)
  base=raw[(raw.environment_id==outer)&(raw.schedule==str([.3,.6]))].copy();base['method']='fixed_03_06';base['nade']=[a/scales({outer})[s] for a,s in zip(base.ade,base.dataset)];oof.append(base)
  for rank_i,row in enumerate(rank.itertuples()):folds.append(dict(held=outer,rank=rank_i+1,schedule=row.schedule,nade=row.nade,inner_envs=len(envs)-1))
 result=pd.concat(oof,ignore_index=True);result.to_csv(OUT/'nested_oof_windows.csv',index=False);pd.DataFrame(folds).to_csv(OUT/'inner_selections.csv',index=False)
 metric=['nade','requests','waiting_s','strategy_compute_s'];summary=hierarchical(result,['method'],metric);summary.to_csv(OUT/'nested_oof_summary.csv',index=False)
 env=result.groupby(['method','dataset','environment_id','episode_uuid','side'],as_index=False)[metric].mean().groupby(['method','dataset','environment_id','episode_uuid'],as_index=False)[metric].mean().groupby(['method','dataset','environment_id'],as_index=False)[metric].mean();env.to_csv(OUT/'nested_oof_environments.csv',index=False)
 ci=paired_ci(env,'fixed_03_06','tuned_fixed');full=evaluate(raw,set());full.to_csv(OUT/'full_train_ranking.csv',index=False)
 # Champion must pass original promotion gate; otherwise keep 0.3/0.6 as formal reference.
 promoted=ci['gain_pct']>=3 and ci['lo95']>0
 frozen=dict(primary_reference_schedule=json.loads(full.iloc[0].schedule) if promoted else [.3,.6],training_best_schedule=json.loads(full.iloc[0].schedule),tuned_fixed_promoted=bool(promoted),paired=ci,parameter_combinations=len(schedules),nested_outer_envs=len(envs),development_replays=0,test_trajectory_reads=0,elapsed_s=time.perf_counter()-start)
 (OUT/'selection.json').write_text(json.dumps(frozen,indent=2),encoding='utf8');print(summary.to_string(index=False));print(json.dumps(frozen),flush=True)
if __name__=='__main__':main()
