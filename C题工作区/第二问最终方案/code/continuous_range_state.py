"""Research baseline with continuous empirical ranges under a fixed paid protocol."""
import math
import numpy as np
from candidate_replay_engine import MotionState,replay,FixedPolicy

class ContinuousRangeState(MotionState):
 def __init__(self,history_t,history_x,motion='damped',update='secant',*,source,scale,prior_models,return_models,observation_protocol='fixed_paid'):
  if observation_protocol!='fixed_paid':raise ValueError('continuous ranges require fixed paid observations, not natural recovery')
  if motion!='damped' or update!='secant':raise ValueError('calibrated baseline only')
  ht=np.asarray(history_t,float);hx=np.asarray(history_x,float)
  if ht.shape!=(2,) or hx.shape!=(2,3) or not np.isfinite(ht).all() or not np.isfinite(hx).all() or not np.all(np.diff(ht)>0):raise ValueError('two finite ordered observations required')
  if not math.isfinite(scale) or scale<=0:raise ValueError('positive finite scale required')
  if source not in prior_models or source not in return_models:raise ValueError('unsupported source')
  self.models=[]
  for table in [prior_models,return_models]:
   f=table[source];mu=np.array(f['mu'],float);sd=np.array(f['sd'],float);coef=np.array(f['coefficient'],float)
   if mu.shape!=(4,) or sd.shape!=(4,) or coef.shape!=(5,) or not all(np.isfinite(v).all() for v in [mu,sd,coef]) or np.any(sd<=0):raise ValueError('invalid calibration coefficients')
   self.models.append((mu,sd,coef))
  super().__init__(ht,hx,motion,update);self.scale=float(scale);self.range_outputs=[];self.last_query=None
 def receive(self,sample_time,position):
  position=np.asarray(position,float)
  if not math.isfinite(sample_time) or sample_time<=self.last_time or position.shape!=(3,) or not np.isfinite(position).all() or self.returns>=2:raise ValueError('invalid return')
  super().receive(sample_time,position)
 def predict(self,now):
  if not math.isfinite(now) or now<self.last_time or (self.last_query is not None and now<self.last_query):raise ValueError('invalid prediction time')
  age=float(now-self.last_time)
  if age>1+1e-9:raise ValueError('outside calibrated age horizon')
  pred=super().predict(now);age_bin=int(np.clip(np.ceil(age*30),1,30));a=age_bin/30;v=float(np.log1p(np.linalg.norm(self.velocity)/self.scale));mu,sd,coef=self.models[int(self.returns>0)];x=np.array([a,a*a,v,a*v]);radius=max(0.,float(np.r_[1.,(x-mu)/sd]@coef))*self.scale
  self.range_outputs.append(dict(time=float(now),age=age,age_bin=age_bin,returns=self.returns,radius=radius,phase='returned' if self.returns else 'prior',status='descriptive_only',guaranteed=False,support='conditional_support_not_certified'));self.last_query=float(now);return pred

def fixed_joint_replay(times,reference,history_times,history_positions,*,source,scale,prior_models,return_models,observation_protocol='fixed_paid'):
 """Protocol fixed internally: .3/.6 requests, .1 delay, budget2, actual final frame."""
 if observation_protocol!='fixed_paid':raise ValueError('fixed paid protocol required')
 times=np.asarray(times,float);reference=np.asarray(reference,float);ht=np.asarray(history_times,float);hx=np.asarray(history_positions,float)
 if times.ndim!=1 or len(times)<2 or not np.isfinite(times).all() or not np.all(np.diff(times)>0) or times[-1]-times[0]>1+1e-9:raise ValueError('invalid scoring horizon')
 if reference.shape!=(len(times),3) or not np.isfinite(reference).all():raise ValueError('finite scoring reference with shape (n,3) required')
 if ht.shape!=(2,) or hx.shape!=(2,3) or not np.isfinite(ht).all() or not np.isfinite(hx).all() or not np.all(np.diff(ht)>0):raise ValueError('two finite ordered historical observations required')
 if ht[-1]>=times[0]:raise ValueError('history must strictly precede the hidden scoring interval')
 instances=[]
 def factory(*args):
  state=ContinuousRangeState(*args,source=source,scale=scale,prior_models=prior_models,return_models=return_models);instances.append(state);return state
 run=replay(times,reference,history_times,history_positions,FixedPolicy(),delay=.1,budget=2,window_end=float(times[-1]),state_factory=factory);run['ranges']=instances[0].range_outputs;return run
