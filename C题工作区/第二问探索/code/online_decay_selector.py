import math
import numpy as np
from candidate_replay_engine import MotionState

class OnlineDecaySelector(MotionState):
 def __init__(self,*args,fit,scale,source,sources,**kwargs):
  super().__init__(*args,**kwargs);self.fit=fit;self.scale=float(scale);self.source=source;self.sources=sources;self.tau=.2;self.decisions=[]
 def receive(self,sample_time,position):
  if self.returns>=2 or sample_time<=self.last_time:raise ValueError('invalid return')
  gap=float(sample_time-self.last_time);sec=(np.asarray(position)-self.last)/gap;old=self.velocity;oldnorm=float(np.linalg.norm(old));newnorm=float(np.linalg.norm(sec));features=np.array([oldnorm,newnorm,float(np.linalg.norm(sec-old)),float(np.dot(old,sec)/(oldnorm*newnorm+1e-12)),gap,float(self.returns)]);raw=features.copy();features[:3]/=self.scale;x=np.r_[features,[float(self.source==s) for s in self.sources]];x=np.r_[1.,(x-np.array(self.fit['mu']))/np.array(self.fit['sd'])];scores=x@np.array(self.fit['coefficient']);self.tau=[.1,.2,.4][int(np.argmin(scores))];self.decisions.append(dict(stage=self.returns,tau=self.tau,features=raw.tolist(),scores=scores.tolist()));self.velocity=sec;self.last=np.asarray(position,float).copy();self.last_time=float(sample_time);self.returns+=1
 def predict(self,now):
  if not self.returns:return super().predict(now)
  age=float(now-self.last_time)
  if age<0:raise ValueError('prediction precedes sample')
  return self.last-self.tau*math.expm1(-age/self.tau)*self.velocity
