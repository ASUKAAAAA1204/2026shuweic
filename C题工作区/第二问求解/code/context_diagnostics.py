"""Post-hoc task/annotation-progress/history evaluation, never policy inputs."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import s0_audit_replay as s0
BASE=Path(__file__).resolve().parents[1]
def main():
 windows=pd.read_csv(s0.Q1.parent/'C题工作区/阶段交付/stage1_review/corrected_results/data/windows.csv');segments=pd.read_csv(s0.DATA/'segments_private.csv');rows=[];cache={}
 for w in windows.itertuples():
  key=(w.episode_uuid,w.side)
  if key not in cache:
   with np.load(s0.DATA/'records'/f'{w.episode_uuid}.npz',allow_pickle=False) as z:cache[key]=(z[w.side+'__t_s'],z[w.side+'__position'].mean(axis=1))
  t,x=cache[key];i=w.start_index;lo=w.history_start;hist=x[lo:i];dt=np.diff(t[lo:i]);speed=np.linalg.norm(np.diff(hist,axis=0),axis=1)/dt
  hit=segments[(segments.episode_uuid==w.episode_uuid)&(segments.start_s<=t[i])&(t[i]<segments.end_s)]
  family='unannotated';phase='unannotated'
  if len(hit)==1:
   s=hit.iloc[0];family=s.family_id;phase=['early','middle','late'][min(2,int(3*(t[i]-s.start_s)/s.duration_s))]
  rows.append(dict(window_id=w.window_id,split=w.split,dataset=w.dataset,environment_id=w.environment_id,episode_uuid=w.episode_uuid,side=w.side,family=family,annotation_progress=phase,history_speed=float(np.median(speed))))
 meta=pd.DataFrame(rows);cut=meta[meta.split=='train'].groupby('dataset').history_speed.median().to_dict();meta['historical_motion']=np.where(meta.history_speed>=meta.dataset.map(cut),'high_speed','low_speed');out=BASE/'results/diagnostics';out.mkdir(exist_ok=True);meta.to_csv(out/'window_context_private_evaluation_only.csv',index=False)
 allrows=[]
 for split,path in [('train',BASE/'results/S2/windows.csv'),('validation',BASE/'results/S5_validation/windows.csv'),('test',BASE/'results/S5_test/windows.csv')]:
  a=pd.read_csv(path).query("method=='fixed' and length_s==1 and delay_s==0.1")
  b=a.merge(meta[['window_id','family','annotation_progress','historical_motion']],on='window_id',validate='one_to_one')
  for factor in ['family','annotation_progress','historical_motion']:
   keys=['dataset',factor,'environment_id','episode_uuid','side'];metric=['nade','nfde','exceed_0.05','coverage'];g=b.groupby(keys,as_index=False)[metric].mean().groupby(keys[:-1],as_index=False)[metric].mean().groupby(keys[:-2],as_index=False)[metric].mean().groupby(['dataset',factor],as_index=False)[metric].mean()
   count=b.groupby(['dataset',factor]).agg(windows=('window_id','size'),records=('episode_uuid','nunique'),environments=('environment_id','nunique')).reset_index();g=g.merge(count,on=['dataset',factor]);g['split']=split;g['factor']=factor;g=g.rename(columns={factor:'group'});allrows.append(g)
 pd.concat(allrows,ignore_index=True).to_csv(out/'phase_task_history_summary.csv',index=False)
 (out/'audit.json').write_text(json.dumps(dict(evaluation_only=True,model_refitted=False,source_endpoints_never_policy_inputs=True,speed_cutoffs_training_only=cut,warning='annotation thirds are post-hoc progress proxies, not verified reaching/grasping/placing labels; subgroup correlations are not causal effects'),indent=2),encoding='utf8');print('posthoc context diagnostic saved',flush=True)
if __name__=='__main__':main()
