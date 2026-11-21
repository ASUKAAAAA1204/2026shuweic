"""No-refit train diagnostic for query time, delay and tolerance feasibility."""
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
BASE=Path(__file__).resolve().parents[1];OLD=BASE.parent/'第二问求解';sys.path.insert(0,str(OLD/'code'))
from engine import replay,FixedPolicy,NeverPolicy
import s0_audit_replay as s0

def main():
 out=BASE/'results/feasibility';out.mkdir(exist_ok=True);w=pd.read_csv(s0.Q1.parent/'C题工作区/阶段交付/stage1_review/corrected_results/data/windows.csv').query("split=='train'");scales=json.loads((OLD/'results/S0/fold_scales.json').read_text());cache={};rows=[]
 for a in w.itertuples():
  key=(a.episode_uuid,a.side)
  if key not in cache:
   with np.load(s0.DATA/'records'/f'{a.episode_uuid}.npz',allow_pickle=False) as z:cache[key]=(z[a.side+'__t_s'],z[a.side+'__position'].mean(axis=1))
  t,x=cache[key];i,j=a.start_index,a.end_index_exclusive;scale=scales[a.environment_id][a.dataset]
  for length in [.2,.6,1.]:
   idx=np.arange(i,j)[t[i:j]-t[i]<=length+1e-9];now=t[idx]-t[i]
   schedules=[('none',()),('fixed_03_06',(.3,.6)),('fixed_02_04',(.2,.4)),('fixed_01_03',(.1,.3)),('fixed_02_05',(.2,.5)),('fixed_04_07',(.4,.7))]
   for delay in [0,.1,.2]:
    for name,schedule in schedules:
     q=FixedPolicy(schedule) if schedule else NeverPolicy();r=replay(t[idx],x[idx],t[i-2:i],x[i-2:i],q,delay=delay,window_end=float(t[i]+length));e=np.array([p['error'] for p in r['points']])/scale
     acts=[a for a in r['actions'] if a['event']=='request'];ret=[a for a in r['actions'] if a['event']=='return']
     rows.append(dict(window_id=a.window_id,dataset=a.dataset,environment_id=a.environment_id,episode_uuid=a.episode_uuid,side=a.side,length_s=length,delay_s=delay,schedule=name,nade=e.mean(),exceed_02=np.mean(e>.02),exceed_05=np.mean(e>.05),exceed_10=np.mean(e>.1),requests=r['metrics']['requests'],unreturned_paid=len(acts)-len(ret),terminal_requests=sum(z['sample_time']>=float(t[i]+length)-1e-9 for z in acts),effective_returns=len(ret)))
 m=pd.DataFrame(rows);m.to_csv(out/'all_schedules.csv.gz',index=False);levels=[['dataset','environment_id','episode_uuid','side'],['dataset','environment_id','episode_uuid'],['dataset','environment_id'],['dataset'],[]];g=m.copy()
 for lev in levels:
  keys=['length_s','delay_s','schedule']+lev;g=g.groupby(keys,as_index=False)[['nade','exceed_02','exceed_05','exceed_10','requests','unreturned_paid','terminal_requests','effective_returns']].mean()
 g.to_csv(out/'summary.csv',index=False);g.query('length_s==1').to_csv(out/'one_second.csv',index=False);print(g.query('length_s==1')[['delay_s','schedule','nade','exceed_05','requests','unreturned_paid','terminal_requests']].to_string(index=False))
if __name__=='__main__':main()
