import json
import numpy as np,pandas as pd
from request_value import BASE,OLD
from update_calibration_experiment import hierarchical,paired_ci
from candidate_replay_engine import MotionState,replay,FixedPolicy
import s0_audit_replay as s0

class ReturnDecay(MotionState):
 def __init__(self,*args,tau,**kwargs):super().__init__(*args,**kwargs);self.return_tau=tau
 def predict(self,now):
  if not self.returns:return super().predict(now)
  age=now-self.last_time;return self.last-self.return_tau*np.expm1(-age/self.return_tau)*self.velocity

def main():
 out=BASE/'results/post_return_decay';out.mkdir(exist_ok=True);w=pd.read_csv(s0.Q1.parent/'C题工作区/阶段交付/stage1_review/corrected_results/data/windows.csv').query("split=='train'");sc=json.loads((OLD/'results/S0/fold_scales.json').read_text());cache={};rows=[]
 for r in w.itertuples():
  key=(r.episode_uuid,r.side)
  if key not in cache:
   with np.load(s0.DATA/'records'/f'{r.episode_uuid}.npz',allow_pickle=False) as z:cache[key]=(z[r.side+'__t_s'],z[r.side+'__position'].mean(axis=1))
  t,x=cache[key];i,j=r.start_index,r.end_index_exclusive
  for tau in [.1,.2,.4]:
   def factory(*args):return ReturnDecay(*args,tau=tau)
   run=replay(t[i:j],x[i:j],t[i-2:i],x[i-2:i],FixedPolicy(),delay=.1,budget=2,window_end=float(t[j-1]),state_factory=factory);e=np.array([p['error'] for p in run['points']])/sc[r.environment_id][r.dataset]
   rows.append(dict(window_id=r.window_id,dataset=r.dataset,environment_id=r.environment_id,episode_uuid=r.episode_uuid,side=r.side,tau=tau,raw_ade=run['metrics']['ade'],nade=e.mean(),nfde=e[-1],exceed05=float((e>.05).mean()),requests=run['metrics']['requests'],waiting_s=run['metrics']['waiting_s']))
 raw=pd.DataFrame(rows);raw.to_csv(out/'candidate_windows.csv.gz',index=False);selected=[];choices=[]
 # Scale depends on held fold. Re-express all candidate errors to the outer fold scales before selection.
 for held in sorted(w.environment_id.unique()):
  fit=raw[raw.environment_id!=held].copy();fit['nade']=fit.raw_ade/np.array([sc[held][src] for src in fit.dataset]);rank=hierarchical(fit,['tau'],['nade']);rank['priority']=rank.tau.map({.2:0,.1:1,.4:2});chosen=float(rank.sort_values(['nade','priority']).iloc[0].tau);z=raw[(raw.environment_id==held)&(raw.tau==chosen)].copy();z['method']='train_selected';selected.append(z);choices.append(dict(held=held,tau=chosen))
 base=raw[raw.tau==.2].copy();base['method']='baseline';result=pd.concat([base,*selected]);result.to_csv(out/'selected_windows.csv.gz',index=False);pd.DataFrame(choices).to_csv(out/'choices.csv',index=False);metrics=['nade','nfde','exceed05','requests','waiting_s'];summary=hierarchical(result,['method'],metrics);summary.to_csv(out/'summary.csv',index=False);env=result.groupby(['method','dataset','environment_id','episode_uuid','side'],as_index=False)[metrics].mean().groupby(['method','dataset','environment_id','episode_uuid'],as_index=False)[metrics].mean().groupby(['method','dataset','environment_id'],as_index=False)[metrics].mean();env.to_csv(out/'environments.csv',index=False);src=env.groupby(['method','dataset'],as_index=False)[metrics].mean();src.to_csv(out/'sources.csv',index=False);ci=paired_ci(env,'baseline','train_selected');old=pd.read_csv(BASE/'results/distance_aligned_return/windows.csv.gz').query("method=='secant'").set_index('window_id');delta=float(np.max(abs(base.set_index('window_id').nade-old.nade)));assert delta<1e-12
 audit=dict(passed=True,windows=len(w),candidate_replays=len(raw),original_baseline_max_difference=delta,comparison=ci,development_reads=0,test_reads=0,formal_model_changed=False,choices=pd.DataFrame(choices).tau.value_counts().to_dict());(out/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf8');print(summary.to_string(index=False));print(json.dumps(audit))
if __name__=='__main__':main()
