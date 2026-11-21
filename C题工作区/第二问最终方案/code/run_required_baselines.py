"""Required baseline/horizon comparisons, not a candidate model search."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
from candidate_replay_engine import replay,FixedPolicy,NeverPolicy
from run_complete import aggregate
BASE=Path(__file__).resolve().parents[1]
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--data-root',type=Path,required=True);ap.add_argument('--windows',type=Path,required=True);args=ap.parse_args()
 cfg=json.loads((BASE/'model/model.json').read_text(encoding='utf8'));folds=json.loads((BASE/'model/fold_scales.json').read_text(encoding='utf8'));windows=pd.read_csv(args.windows);rows=[];cache={}
 for r in windows.itertuples():
  if r.episode_uuid not in cache:
   with np.load(args.data_root/'records'/f'{r.episode_uuid}.npz',allow_pickle=False) as z:cache[r.episode_uuid]={s:(z[s+'__t_s'],z[s+'__position'].mean(axis=1)) for s in ['left_hand','right_hand']}
  t,x=cache[r.episode_uuid][r.side];i=int(r.start_index);end=int(r.end_index_exclusive);scale=folds[r.environment_id][r.dataset] if r.split=='train' else cfg['scale'][r.dataset]
  for duration in [.2,.6,1.]:
   j=int(np.searchsorted(t,t[i]+duration+1e-9,side='right'));j=min(j,end)
   for method,motion,paid in [('hold','hold',False),('constant_velocity','cv',False),('damped_no_request','damped',False),('adopted_fixed','damped',True)]:
    run=replay(t[i:j],x[i:j],t[i-2:i],x[i-2:i],FixedPolicy() if paid else NeverPolicy(),motion=motion,delay=.1,budget=2,window_end=float(t[j-1]));m=run['metrics']
    rows.append({k:getattr(r,k) for k in ['window_id','split','dataset','environment_id','episode_uuid','side']}|dict(duration=duration,method=method,nade=m['ade']/scale,nfde=m['fde']/scale,exceed_005=np.mean([p['error']/scale>.05 for p in run['points']]),**m))
 frame=pd.DataFrame(rows);out=BASE/'results';frame.to_csv(out/'required_baseline_windows.csv.gz',index=False)
 metrics=['nade','nfde','exceed_005','requests','cost','waiting_s','no_request_wait_s','strategy_compute_s'];s=aggregate(frame,['split','duration','method'],metrics);s.to_csv(out/'required_baseline_summary.csv',index=False)
 aggregate(frame,['split','duration','method','dataset'],metrics).to_csv(out/'required_baseline_sources.csv',index=False)
 adopted=frame.query("duration==1 and method=='adopted_fixed'").set_index('window_id');original=pd.read_csv(out/'window_metrics.csv').set_index('window_id');assert set(adopted.index)==set(original.index)
 diff=float(abs(adopted.nade-original.nade).max());assert diff<1e-12
 (out/'required_baseline_audit.json').write_text(json.dumps(dict(windows=len(windows),replays=len(frame),methods=4,horizons=[.2,.6,1.],maximum_adopted_nade_difference=diff,delay_s=.1,budget_cap=2,cost_per_request=1,all_comparators_share_windows=True,uncertainty='no range coverage transferred to truncated horizons',evaluation='previously viewed descriptive comparisons'),indent=2),encoding='utf8')
 print(s.to_string(index=False))
if __name__=='__main__':main()
