import numpy as np

class SecondPurchasePolicy:
 def __init__(self,*,initial_velocity,initial_time,fit,scale,source,sources,cost_weight=.02):
  velocity=np.array(initial_velocity,float).copy();sources=tuple(sources);width=5+len(sources)
  mu=np.asarray(fit['mu'],float).copy();sd=np.asarray(fit['sd'],float).copy();coef=np.asarray(fit['coefficient'],float).copy()
  if velocity.shape!=(3,) or not np.isfinite(velocity).all() or not np.isfinite(initial_time):raise ValueError('invalid initial state')
  if source not in sources or len(set(sources))!=len(sources):raise ValueError('unsupported source')
  if not np.isfinite(scale) or scale<=0 or not np.isfinite(cost_weight) or cost_weight<0:raise ValueError('invalid scale/cost')
  if mu.shape!=(width,) or sd.shape!=(width,) or coef.shape!=(width+1,) or not all(np.isfinite(a).all() for a in [mu,sd,coef]) or (sd<=0).any():raise ValueError('invalid model')
  for a in [mu,sd,coef]:a.flags.writeable=False
  self.initial_velocity=velocity;self.initial_time=float(initial_time);self.fit={'mu':mu,'sd':sd,'coefficient':coef};self.scale=float(scale);self.source=source;self.sources=sources;self.cost_weight=float(cost_weight);self.first_sent=False;self.second_decided=False;self.gain=None;self.buy=None;self.features=None
 def request(self,view):
  if not self.first_sent:
   if view.relative+1e-12>=.3:self.first_sent=True;return True
   return False
  if view.successful_returns==1 and self.gain is None:
   old=self.initial_velocity;sec=np.array(view.velocity);a=float(np.linalg.norm(old));b=float(np.linalg.norm(sec));features=np.array([a,b,float(np.linalg.norm(sec-old)),float(np.dot(old,sec)/(a*b+1e-12)),float(view.last_time-self.initial_time)]);self.features=features.copy();features[:3]/=self.scale;x=np.r_[features,[float(self.source==s) for s in self.sources]];x=np.r_[1.,(x-np.array(self.fit['mu']))/np.array(self.fit['sd'])];self.gain=float(x@np.array(self.fit['coefficient']))
  if not self.second_decided and view.relative+1e-12>=.6:
   self.second_decided=True;self.buy=bool(self.gain is not None and self.gain>self.cost_weight);return self.buy
  return False
