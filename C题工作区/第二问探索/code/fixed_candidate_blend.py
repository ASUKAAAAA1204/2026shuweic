import json
import numpy as np,pandas as pd
from request_value import BASE,OLD
from candidate_replay_integration import DistanceState
from online_decay_selector import OnlineDecaySelector
from candidate_replay_engine import replay,FixedPolicy
from update_calibration_experiment import hierarchical,paired_ci
import s0_audit_replay as s0

class Blend(DistanceState):
 def __init__(self,*args,selector_kwargs,**kwargs):
  super().__init__(*args,**kwargs);self.selector=OnlineDecaySelector(*args,**selector_kwargs)
 def receive(self,sample_time,position):
  super().receive(sample_time,position);self.selector.receive(sample_time,position)
 def predict(self,now):return .5*super().predict(now)+.5*self.selector.predict(now)

def main():
 out=BASE/'results/fixed_candidate_blend';out.mkdir(exist_ok=True);w=pd.read_csv(s0.Q1.parent/'C题工作区/阶段交付/stage1_review/corrected_results/data/windows.csv').query("split=='train'");sc=json.loads((OLD/'results/S0/fold_scales.json').read_text());fits={f['held']:f for f in json.loads((BASE/'results/causal_decay_selector/fits.json').read_text())};co=pd.read_csv(BASE/'results/distance_aligned_return/coefficients.csv').set_index('held');sources=sorted(w.dataset.unique());cache={};rows=[]
 for r in w.itertuples():
  key=(r.episode_uuid,r.side)
  if key not in cache:
   with np.load(s0.DATA/'records'/f'{r.episode_uuid}.npz',allow_pickle=False) as z:cache[key]=(z[r.side+'__t_s'],z[r.side+'__position'].mean(axis=1))
  t,x=cache[key];i,j=r.start_index,r.end_index_exclusive;c=co.loc[r.environment_id]
  def factory(*args):return Blend(*args,coefficients=[(c.first_a,c.first_b),(c.second_a,c.second_b)],selector_kwargs=dict(fit=fits[r.environment_id],scale=sc[r.environment_id][r.dataset],source=r.dataset,sources=sources))
  run=replay(t[i:j],x[i:j],t[i-2:i],x[i-2:i],FixedPolicy(),delay=.1,budget=2,window_end=float(t[j-1]),state_factory=factory);e=np.array([p['error'] for p in run['points']])/sc[r.environment_id][r.dataset];rows.append(dict(**{k:getattr(r,k) for k in ['window_id','dataset','environment_id','episode_uuid','side']},method='blend_half',nade=e.mean(),nfde=e[-1],exceed05=float((e>.05).mean()),requests=run['metrics']['requests'],waiting_s=run['metrics']['waiting_s']))
 result=pd.DataFrame(rows);result.to_csv(out/'windows.csv.gz',index=False);old=pd.read_csv(BASE/'results/causal_decay_selector/windows.csv.gz');old=old.replace({'method':{'baseline':'baseline','causal_selector':'selector'}});allrows=pd.concat([old,result]);metrics=['nade','nfde','exceed05','requests','waiting_s'];summary=hierarchical(allrows,['method'],metrics);summary.to_csv(out/'summary.csv',index=False);env=allrows.groupby(['method','dataset','environment_id','episode_uuid','side'],as_index=False)[metrics].mean().groupby(['method','dataset','environment_id','episode_uuid'],as_index=False)[metrics].mean().groupby(['method','dataset','environment_id'],as_index=False)[metrics].mean();env.to_csv(out/'environments.csv',index=False);env.groupby(['method','dataset'],as_index=False)[metrics].mean().to_csv(out/'sources.csv',index=False);audit=dict(windows=len(w),comparison_baseline=paired_ci(env,'baseline','blend_half'),comparison_selector=paired_ci(env,'selector','blend_half'),weight=.5,weight_search=False,development_reads=0,test_reads=0,formal_model_changed=False);(out/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf8');print(summary.to_string(index=False));print(json.dumps(audit))
if __name__=='__main__':main()
