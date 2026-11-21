import json,time,math
import numpy as np,pandas as pd
from distance_aligned_return import BASE,OLD,predictions
import s0_audit_replay as s0
from candidate_replay_engine import MotionState,replay,FixedPolicy

class DistanceState(MotionState):
 def __init__(self,*args,coefficients,**kwargs):
  super().__init__(*args,**kwargs);self.coefficients=coefficients
 def receive(self,sample_time,position):
  assert sample_time>self.last_time and self.returns<2
  gap=sample_time-self.last_time;a,b=self.coefficients[self.returns]
  self.velocity=a*(position-self.last)/gap+b*self.velocity*math.exp(-gap/.2);self.last=np.asarray(position,float).copy();self.last_time=float(sample_time);self.returns+=1

def main():
 out=BASE/'results/candidate_replay_integration';out.mkdir(exist_ok=True);w=pd.read_csv(s0.Q1.parent/'C题工作区/阶段交付/stage1_review/corrected_results/data/windows.csv').query("split=='train'");co=pd.read_csv(BASE/'results/distance_aligned_return/coefficients.csv').set_index('held');cache={};rows=[];maxdiff=0.;prefixdiff=0.;checks=0
 for number,r in enumerate(w.itertuples()):
  key=(r.episode_uuid,r.side)
  if key not in cache:
   with np.load(s0.DATA/'records'/f'{r.episode_uuid}.npz',allow_pickle=False) as z:cache[key]=(z[r.side+'__t_s'],z[r.side+'__position'].mean(axis=1))
  t,x=cache[key];i,j=r.start_index,r.end_index_exclusive;tt=t[i:j];c=co.loc[r.environment_id];coefficients=[(c.first_a,c.first_b),(c.second_a,c.second_b)]
  def factory(*args):return DistanceState(*args,coefficients=coefficients)
  kwargs=dict(delay=.1,budget=2,window_end=float(tt[-1]));runs={}
  for method,f in [('secant',None),('distance_return',factory)]:
   run=replay(tt,x[i:j],t[i-2:i],x[i-2:i],FixedPolicy(),state_factory=f,**kwargs);runs[method]=run;rows.append(dict(window_id=r.window_id,method=method,**run['metrics']))
  assert runs['secant']['actions']==runs['distance_return']['actions'];ret=[v for v in runs['secant']['actions'] if v['event']=='return'];d=dict(t=tt,x=x[i:j],ht=t[i-2:i],hist=x[i-2:i],samples=[int(np.argmin(abs(tt-v['sample_time']))) for v in ret],arrivals=[int(np.argmin(abs(tt-v['processed_at']))) for v in ret]);p=np.array([v['prediction'] for v in runs['distance_return']['points']]);maxdiff=max(maxdiff,float(np.max(abs(p-predictions(d,*coefficients)))))
  if number%95==0:
   mutated=x[i:j]+np.random.default_rng(number).normal(0,1000,size=(j-i,3));run=replay(tt,mutated,t[i-2:i],x[i-2:i],FixedPolicy(),state_factory=factory,**kwargs);stop=d['arrivals'][0];prefixdiff=max(prefixdiff,float(np.max(abs(p[:stop]-np.array([v['prediction'] for v in run['points']])[:stop]))));assert run['actions']==runs['distance_return']['actions'];checks+=1
 assert maxdiff<1e-12 and prefixdiff==0
 m=pd.DataFrame(rows);m.to_csv(out/'window_metrics.csv',index=False);report={}
 for method,g in m.groupby('method'):
  report[method]=dict(total_strategy_compute_s=float(g.strategy_compute_s.sum()),median_window_compute_ms=float(g.strategy_compute_s.median()*1000),p95_window_compute_ms=float(g.strategy_compute_s.quantile(.95)*1000),requests=int(g.requests.sum()),returns=int(g.successful_returns.sum()))
 audit=dict(passed=True,train_windows=len(w),formula_max_difference=maxdiff,pre_return_mutation_checks=checks,pre_return_prediction_difference=prefixdiff,paired_actions_identical=True,metrics=report,development_reads=0,test_reads=0,formal_engine_changed=False,timing_scope='one pass engine-internal strategy loop including view/policy/predict/receive; excludes scorer, IO, calibration and transport; exploratory timing only');(out/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf8');print(json.dumps(audit))
if __name__=='__main__':main()
