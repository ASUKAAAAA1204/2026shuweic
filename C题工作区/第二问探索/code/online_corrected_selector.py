import numpy as np
from online_decay_selector import OnlineDecaySelector

class OnlineCorrectedSelector(OnlineDecaySelector):
 def receive(self,sample_time,position):
  old=self.velocity.copy();gap=float(sample_time-self.last_time);stage=self.returns
  super().receive(sample_time,position)
  a,b=self.fit['first' if stage==0 else 'second'];self.velocity=a*self.velocity+b*old*np.exp(-gap/.2)
