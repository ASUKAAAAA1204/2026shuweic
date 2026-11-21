"""Train-only fixed-grid empirical resource curve; no test/dev data, no optimizer."""
from pathlib import Path
import sys
import numpy as np,pandas as pd
BASE=Path(__file__).resolve().parents[1];OLD=BASE.parent/'第二问求解';sys.path.insert(0,str(OLD/'code'))
from engine import FixedPolicy,NeverPolicy
from end_guard import safe_replay as replay
import s0_audit_replay as s0

def main():
 out=BASE/'results/resource_curve_safe';out.mkdir(exist_ok=True);w=pd.read_csv(s0.Q1.parent/'C题工作区/阶段交付/stage1_review/corrected_results/data/windows.csv').query("split=='train'");scales=__import__('json').loads((OLD/'results/S0/fold_scales.json').read_text());cache={};rows=[]
 for a in w.itertuples():
  key=(a.episode_uuid,a.side)
  if key not in cache:
   with np.load(s0.DATA/'records'/f'{a.episode_uuid}.npz',allow_pickle=False) as z:cache[key]=(z[a.side+'__t_s'],z[a.side+'__position'].mean(axis=1))
  t,x=cache[key];i,j=a.start_index,a.end_index_exclusive;scale=scales[a.environment_id][a.dataset]
  for length in [1.]:
   ix=np.arange(i,j)[t[i:j]-t[i]<=length+1e-9]
   for delay in [0,.1,.2]:
    for budget in range(9):
     schedule=tuple(length*(k+1)/(budget+1) for k in range(budget))
     policy=FixedPolicy(schedule) if budget else NeverPolicy();r=replay(t[ix],x[ix],t[i-2:i],x[i-2:i],policy,delay=delay,budget=budget,window_end=float(t[i]+length));e=np.array([p['error'] for p in r['points']])/scale
     rows.append(dict(window_id=a.window_id,dataset=a.dataset,environment_id=a.environment_id,episode_uuid=a.episode_uuid,side=a.side,length_s=length,delay_s=delay,budget_cap=budget,schedule='|'.join(f'{v:.3f}' for v in schedule),nade=e.mean(),exceed_02=np.mean(e>.02),exceed_05=np.mean(e>.05),exceed_10=np.mean(e>.1),requests=r['metrics']['requests'],waiting_s=r['metrics']['waiting_s'],nfde=r['metrics']['fde']/scale))
 m=pd.DataFrame(rows);m.to_csv(out/'all_windows.csv.gz',index=False);levels=[['dataset','environment_id','episode_uuid','side'],['dataset','environment_id','episode_uuid'],['dataset','environment_id'],['dataset'],[]];g=m.copy()
 for lev in levels:
  keys=['length_s','delay_s','budget_cap']+lev;g=g.groupby(keys,as_index=False)[['nade','exceed_02','exceed_05','exceed_10','requests','waiting_s','nfde']].mean()
 g.to_csv(out/'summary.csv',index=False);print(g.query('length_s==1')[['delay_s','budget_cap','nade','exceed_05','requests']].to_string(index=False))
if __name__=='__main__':main()
