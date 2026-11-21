"""Research range for fixed .3/.6 requests, .1 delay, two returns, one-second replay."""
import math
import numpy as np,pandas as pd
from risk_contract import assess_range

class FrozenSpeedRange:
 def __init__(self,*,source,scale,initial_velocity,prior_table,returned_table,model='baseline'):
  if model not in {'baseline','distance_return','crossfit_corrected'}:raise ValueError('unknown frozen model')
  velocity=np.asarray(initial_velocity,float)
  if velocity.shape!=(3,) or not np.isfinite(velocity).all():raise ValueError('finite three-vector required')
  if not math.isfinite(scale) or scale<=0:raise ValueError('invalid scale')
  prior=prior_table[prior_table.dataset==source].copy();returned=returned_table[(returned_table.dataset==source)&(returned_table.method==model)&returned_table.after].copy()
  if len(prior)!=90 or len(returned)!=30 or prior.duplicated(['age_bin','speed_group']).any() or returned.age_bin.duplicated().any():raise ValueError('incomplete/duplicate calibration cells')
  if prior[['cut1','cut2']].drop_duplicates().shape[0]!=1:raise ValueError('inconsistent speed cuts')
  self.group=int(np.searchsorted(prior[['cut1','cut2']].iloc[0].to_numpy(float),np.linalg.norm(velocity)/scale,side='right'));self.prior=prior.set_index(['age_bin','speed_group']);self.returned=returned.set_index('age_bin');self.scale=float(scale)
 def assess(self,*,age,returns,tolerance=None,regime='paid_isolated'):
  if regime!='paid_isolated':raise ValueError('calibration only supports paid isolated replay')
  if not math.isfinite(age) or age<0 or age>1+1e-9:raise ValueError('age outside calibrated horizon')
  if isinstance(returns,bool) or not isinstance(returns,int) or returns not in [0,1,2]:raise ValueError('invalid successful return count')
  age_bin=int(np.clip(np.ceil(age*30),1,30))
  if returns:
   c=self.returned.loc[age_bin];radius=float(c.radius_normalized)*self.scale;fallback=bool(c.age_fallback)
  else:
   c=self.prior.loc[(age_bin,self.group)];radius=float(c.radius)*self.scale;fallback=bool(c.fallback)
  assessment=assess_range(observation_regime=regime,calibration_regime='paid_isolated',mode='paid_returned' if returns else 'paid_prior',radius=radius,tolerance=tolerance,supported=True)
  return dict(radius=assessment.radius,status=assessment.status,guaranteed=False,age_bin=age_bin,initial_speed_group=self.group,fallback=fallback,calibration_points=int(c.points),calibration_environments=int(c.environments),scope='fixed schedule .3/.6; delay .1; budget 2; one-second artificial mask')
