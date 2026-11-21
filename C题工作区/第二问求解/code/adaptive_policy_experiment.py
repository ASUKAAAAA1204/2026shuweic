"""S4 nested global-environment adaptive-policy comparison; train only."""
from pathlib import Path
import json,time
import numpy as np
import pandas as pd
from engine import replay,FixedPolicy
from policies import RadiusPolicy,BudgetPolicy
from update_calibration_experiment import hierarchical,quantiles,paired_ci
import s0_audit_replay as s0
BASE=Path(__file__).resolve().parents[1];OUT=BASE/'results/S4_adaptive'
PARAMS=[('threshold',p) for p in [.02,.05,.1]]+[('delay_aware',p) for p in [.02,.05,.1]]+[('budget',p) for p in [0.,.002,.01,.03]]

def main():
 OUT.mkdir(parents=True,exist_ok=True);start=time.perf_counter();windows=pd.read_csv(s0.Q1.parent/'C题工作区/阶段交付/stage1_review/corrected_results/data/windows.csv').query("split=='train'")
 points=pd.read_csv(BASE/'results/S3/residual_points.csv.gz');points=points[points.method=='open'];envs=sorted(windows.environment_id.unique());sources=sorted(windows.dataset.unique());records=pd.read_csv(s0.DATA/'record_index.csv').query("split=='train'");cache={};dist=[];fitcache={}
 for rr in records.itertuples():
  with np.load(s0.DATA/'records'/f'{rr.episode_uuid}.npz',allow_pickle=False) as z:
   for side in ['left_hand','right_hand']:
    t=z[side+'__t_s'];x=z[side+'__position'].mean(axis=1);valid=z[side+'__reference_candidate'];head=z['head__position'][:,0];cache[(rr.episode_uuid,side)]=(t,x);dist.append((rr.dataset,rr.environment_id,np.linalg.norm(x[valid]-head[valid],axis=1)))
 def fit(exclude):
  key=tuple(sorted(exclude))
  if key not in fitcache:
   pp=points[~points.environment_id.isin(exclude)];qq=quantiles(pp);r={s:[qq[(s,i)] for i in range(1,31)] for s in sources};scale={s:float(np.median(np.concatenate([a for src,e,a in dist if src==s and e not in exclude]))) for s in sources};fitcache[key]=(r,scale)
  return fitcache[key]
 def evaluate(ws,exclude):
  radii,scale=fit(exclude);rows=[]
  for w in ws.itertuples():
   t,x=cache[(w.episode_uuid,w.side)];i,j=w.start_index,w.end_index_exclusive
   for family,param in PARAMS:
    policy=BudgetPolicy(radii[w.dataset],scale[w.dataset],param,.1) if family=='budget' else RadiusPolicy(radii[w.dataset],scale[w.dataset],param,.1,family=='delay_aware')
    result=replay(t[i:j],x[i:j],t[i-2:i],x[i-2:i],policy,delay=.1,window_end=float(t[i]+1))
    rows.append({**{k:getattr(w,k) for k in ['window_id','episode_uuid','dataset','environment_id','side']},'family':family,'parameter':param,**result['metrics'],'nade':result['metrics']['ade']/scale[w.dataset]})
  return pd.DataFrame(rows)
 allouter=[];selections=[];allinner=[]
 for outer in envs:
  checkpoint=OUT/f'inner_{outer}.csv.gz'
  if checkpoint.exists():inner=pd.read_csv(checkpoint)
  else:
   inner=pd.concat([evaluate(windows[windows.environment_id==held],{outer,held}) for held in envs if held!=outer],ignore_index=True);inner.to_csv(checkpoint,index=False)
  rank=hierarchical(inner,['family','parameter'],['nade','requests']).sort_values(['nade','requests','family','parameter'])
  best=rank.iloc[0];selections.append(dict(outer=outer,family=best.family,parameter=float(best.parameter),inner_nade=float(best.nade),inner_envs=15))
  checkpoint2=OUT/f'outer_{outer}.csv.gz'
  if checkpoint2.exists():outer_result=pd.read_csv(checkpoint2)
  else:
   outer_result=evaluate(windows[windows.environment_id==outer],{outer});outer_result.to_csv(checkpoint2,index=False)
  allouter.append(outer_result)
  print('completed outer',outer,'best',best.family,best.parameter,flush=True)
 outer=pd.concat(allouter,ignore_index=True);outer.to_csv(OUT/'all_candidate_oof_windows.csv.gz',index=False);pd.DataFrame(selections).to_csv(OUT/'nested_selections.csv',index=False)
 chosen=[]
 for row in selections:
  a=outer[(outer.environment_id==row['outer'])&(outer.family==row['family'])&np.isclose(outer.parameter,row['parameter'])].copy();a['method']='adaptive_nested';chosen.append(a)
 fixed=pd.read_csv(BASE/'results/S4_fixed/nested_oof_windows.csv').query("method=='fixed_03_06'").copy();chosen.append(fixed);m=pd.concat(chosen,ignore_index=True);m.to_csv(OUT/'primary_oof_windows.csv',index=False)
 cols=['nade','requests','waiting_s','strategy_compute_s'];hierarchical(m,['method'],cols).to_csv(OUT/'primary_summary.csv',index=False)
 env=m.groupby(['method','dataset','environment_id','episode_uuid','side'],as_index=False)[cols].mean().groupby(['method','dataset','environment_id','episode_uuid'],as_index=False)[cols].mean().groupby(['method','dataset','environment_id'],as_index=False)[cols].mean();env.to_csv(OUT/'primary_environments.csv',index=False);ci=paired_ci(env,'fixed_03_06','adaptive_nested')
 full=hierarchical(outer,['family','parameter'],['nade','requests']).sort_values(['nade','requests','family','parameter']);full.to_csv(OUT/'full_train_ranking.csv',index=False)
 src=env.groupby(['method','dataset']).nade.mean().unstack(0);sg=100*(1-src.adaptive_nested/src.fixed_03_06)
 summary=hierarchical(m,['method'],cols).set_index('method');costgain=100*(1-summary.loc['adaptive_nested','requests']/summary.loc['fixed_03_06','requests']);promote=ci['gain_pct']>=3 and ci['lo95']>0 and sg.min()>=-5
 frozen=dict(selected='adaptive' if promote else 'fixed',fixed_schedule=[.3,.6],adaptive_best_family=full.iloc[0].family,adaptive_best_parameter=float(full.iloc[0].parameter),adaptive_promoted=bool(promote),primary_comparison=ci,source_gain_pct=sg.to_dict(),cost_gain_pct=float(costgain),cost_tradeoff_qualifies=bool(costgain>=15 and ci['gain_pct']>=-3),development_used=False,test_trajectory_reads=0,elapsed_s=time.perf_counter()-start,inner_calibration_excludes_outer_and_inner=True)
 (OUT/'selection.json').write_text(json.dumps(frozen,indent=2),encoding='utf8');print(json.dumps(frozen),flush=True)
if __name__=='__main__':main()
