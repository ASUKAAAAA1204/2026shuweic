"""State adapter for a past-only frozen rotation trajectory and paid single-point returns."""
import numpy as np

def factory(initial_times,initial_positions,motion_state_class):
 t=np.asarray(initial_times,float).copy();x=np.asarray(initial_positions,float).copy()
 if t.ndim!=1 or x.shape!=(len(t),3) or not len(t) or not np.isfinite(t).all() or not np.isfinite(x).all() or not np.all(np.diff(t)>0):raise ValueError('invalid frozen forecast')
 class RotationSwitch(motion_state_class):
  def predict(self,now):
   if self.returns:return super().predict(now)
   if now<t[0]-1e-9 or now>t[-1]+1e-9:raise ValueError('outside forecast horizon')
   return np.array([np.interp(now,t,x[:,k]) for k in range(3)])
 return RotationSwitch
