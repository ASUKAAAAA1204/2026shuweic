"""Registered 5-weight causal update trial; nested train environment selection."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
import pandas as pd
BASE=Path(__file__).resolve().parents[1];OLD=BASE.parent/'第二问求解';sys.path.insert(0,str(OLD/'code'))
from engine_fusion import replay,FixedPolicy,MotionState
from update_calibration_experiment import hierarchical,paired_ci
import s0_audit_replay as s0

def main():
 out=BASE/'results/fusion';out.mkdir(exist_ok=True);w=pd.read_csv(s0.Q1.parent/'C题工作区/阶段交付/stage1_review/corrected_results/data/windows.csv').query("split=='train'");scales=json.loads((OLD/'results/S0/fold_scales.json').read_text());rows=[];cache={};envs=sorted(w.environment_id.unique());dist=[]
 rec=pd.read_csv(s0.DATA/'record_index.csv').query("split=='train'")
 for rr in rec.itertuples():
  with np.load(s0.DATA/'records'/f'{rr.episode_uuid}.npz',allow_pickle=False) as z:
   for side in ['left_hand','right_hand']:
    t=z[side+'__t_s'];x=z[side+'__position'].mean(axis=1);valid=z[side+'__reference_candidate'];cache[(rr.episode_uuid,side)]=(t,x);dist.append((rr.dataset,rr.environment_id,np.linalg.norm(x[valid]-z['head__position'][valid,0],axis=1)))
 for a in w.itertuples():
  t,x=cache[(a.episode_uuid,a.side)];i,j=a.start_index,a.end_index_exclusive
  for alpha in [0.,.25,.5,.75,1.]:
   MotionState.fusion=alpha;r=replay(t[i:j],x[i:j],t[i-2:i],x[i-2:i],FixedPolicy(),delay=.1,window_end=float(t[i]+1));rows.append({**{k:getattr(a,k) for k in ['window_id','dataset','environment_id','episode_uuid','side']},'alpha':alpha,'ade':r['metrics']['ade'],'nade':r['metrics']['ade']/scales[a.environment_id][a.dataset]})
 m=pd.DataFrame(rows);m.to_csv(out/'all_weights_windows.csv',index=False);hierarchical(m,['alpha'],['nade']).to_csv(out/'weight_summary.csv',index=False);chosen=[];folds=[]
 for held in envs:
  inner=m[m.environment_id!=held].copy();inner_scale={}
  for e in envs:
   if e==held:continue
   for src in m.dataset.unique():inner_scale[(e,src)]=np.median(np.concatenate([v for s,h,v in dist if s==src and h not in {e,held}]))
  inner['nade']=[a/inner_scale[(e,s)] for a,e,s in zip(inner.ade,inner.environment_id,inner.dataset)]
  rank=hierarchical(inner,['alpha'],['nade']).sort_values(['nade','alpha'],ascending=[True,False]);alpha=float(rank.iloc[0].alpha);folds.append(dict(held=held,selected_alpha=alpha))
  for method,weight in [('fusion_nested',alpha),('secant',1.)]:
   a=m[(m.environment_id==held)&(m.alpha==weight)].copy();a['method']=method;chosen.append(a)
 oof=pd.concat(chosen);oof.to_csv(out/'nested_oof.csv',index=False);summary=hierarchical(oof,['method'],['nade']);summary.to_csv(out/'nested_summary.csv',index=False);pd.DataFrame(folds).to_csv(out/'fold_selection.csv',index=False)
 env=oof.groupby(['method','dataset','environment_id','episode_uuid','side'],as_index=False).nade.mean().groupby(['method','dataset','environment_id','episode_uuid'],as_index=False).nade.mean().groupby(['method','dataset','environment_id'],as_index=False).nade.mean();ci=paired_ci(env,'secant','fusion_nested');src=env.groupby(['method','dataset']).nade.mean().unstack(0);sg=100*(1-src.fusion_nested/src.secant);admit=ci['gain_pct']>=3 and ci['lo95']>0 and sg.min()>=-5
 audit=dict(admitted=bool(admit),comparison=ci,source_gain_pct=sg.to_dict(),train_windows=len(w),replays=len(m),test_reads=0,development_reads=0,formal_model_sha256=hashlib.sha256((OLD/'frozen_model.json').read_bytes()).hexdigest());(out/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf8');print(pd.read_csv(out/'weight_summary.csv').to_string(index=False));print(json.dumps(audit),flush=True)
if __name__=='__main__':main()
