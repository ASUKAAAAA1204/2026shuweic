import json
import numpy as np,pandas as pd
from request_value import BASE
from frozen_speed_range import FrozenSpeedRange
from candidate_replay_engine import replay,FixedPolicy
from candidate_replay_integration import DistanceState
from online_corrected_selector import OnlineCorrectedSelector
import s0_audit_replay as s0

def main():
 prior=pd.read_csv(BASE/'results/initial_speed_validation/frozen_ranges.csv');returned=pd.read_csv(BASE/'results/validation_matched_ranges/frozen_ranges.csv');models=json.loads((BASE/'results/validation_candidates/full_train_models.json').read_text());w=pd.read_csv(s0.Q1.parent/'C题工作区/阶段交付/stage1_review/corrected_results/data/windows.csv').query("split=='validation'");expected=pd.read_csv(BASE/'results/initial_speed_validation/points.csv.gz').query("variant=='speed_conditioned'");cache={};maximum=0.;checks=0
 for r in w.itertuples():
  key=(r.episode_uuid,r.side)
  if key not in cache:
   with np.load(s0.DATA/'records'/f'{r.episode_uuid}.npz',allow_pickle=False) as z:cache[key]=(z[r.side+'__t_s'],z[r.side+'__position'].mean(axis=1))
  t,x=cache[key];i=r.start_index;v=(x[i-1]-x[i-2])/(t[i-1]-t[i-2]);scale=models['scale'][r.dataset]
  for method,g in expected[expected.window_id==r.window_id].groupby('method'):
   interface=FrozenSpeedRange(source=r.dataset,scale=scale,initial_velocity=v,prior_table=prior,returned_table=returned,model=method)
   factory=None
   if method=='distance_return':factory=lambda *a:DistanceState(*a,coefficients=[models['first'],models['second']])
   if method=='crossfit_corrected':factory=lambda *a:OnlineCorrectedSelector(*a,fit=models['crossfit'],scale=scale,source=r.dataset,sources=models['sources'])
   run=replay(t[i:r.end_index_exclusive],x[i:r.end_index_exclusive],t[i-2:i],x[i-2:i],FixedPolicy(),delay=.1,budget=2,window_end=float(t[r.end_index_exclusive-1]),state_factory=factory)
   for point in g.itertuples():
    actual=run['points'][point.index];assert abs(actual['error']/scale-point.nade)<1e-12
    result=interface.assess(age=actual['age'],returns=int(actual['returns']));assert result['guaranteed'] is False and result['status']=='descriptive_only';maximum=max(maximum,abs(result['radius']/scale-point.radius_normalized));checks+=1
 assert maximum<1e-12
 interface=FrozenSpeedRange(source='EgoDemo',scale=models['scale']['EgoDemo'],initial_velocity=[0,0,0],prior_table=prior,returned_table=returned)
 rejected=0
 for args in [dict(age=-1,returns=0),dict(age=float('nan'),returns=0),dict(age=1.1,returns=0),dict(age=.2,returns=3),dict(age=.2,returns=True),dict(age=.2,returns=0,regime='natural_continuous'),dict(age=.2,returns=0,tolerance=0)]:
  try:interface.assess(**args)
  except ValueError:rejected+=1
 assert rejected==7
 out=BASE/'results/initial_speed_validation/interface_audit.json';audit=dict(passed=True,checked_points=checks,max_radius_difference=maximum,invalid_inputs_rejected=rejected,scope='actual replay state age and returns plus visible initial history; same upstream replay implementation',guaranteed=False);out.write_text(json.dumps(audit,indent=2),encoding='utf8');print(json.dumps(audit))
if __name__=='__main__':main()
