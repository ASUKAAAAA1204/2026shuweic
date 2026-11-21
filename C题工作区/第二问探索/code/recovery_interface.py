"""Causal recovery interface. No reference values, future masks or end times."""
from dataclasses import dataclass
import math
import numpy as np

@dataclass(frozen=True)
class RiskOutput:
    position: tuple|None
    mode: str
    observation_age: float|None
    empirical_radius: float|None
    tolerance_status: str
    reliable_guarantee: bool=False

class RecoveryTracker:
    def __init__(self,max_gap_s=.12,tau=.2):
        self.max_gap=max_gap_s;self.tau=tau;self.last_t=None;self.last=None;self.v=None;self.mode='uninitialized';self.previous_good=None;self.last_arrival=-math.inf
    def predict(self,now,tolerance=None,radius=None,supported=False):
        if self.last_t is None:return RiskOutput(None,'uninitialized',None,None,'unavailable')
        if now<self.last_t:raise ValueError('prediction precedes known sample')
        age=now-self.last_t;mode=self.mode
        if age>self.max_gap:mode='missing'
        pred=self.last-self.tau*math.expm1(-age/self.tau)*self.v
        # Radius can be descriptive only. No empirical range implies guaranteed risk.
        status='uncertified'
        if radius is not None and tolerance is not None and supported and mode=='tracking':
            status='empirical_within' if radius<=tolerance else 'empirical_exceeds'
        return RiskOutput(tuple(pred),mode,age,radius,status)
    def receive(self,sample_time,position,valid,arrival_time):
        if arrival_time<self.last_arrival or sample_time>arrival_time:raise ValueError('illegal time')
        if self.last_t is not None and sample_time<=self.last_t:raise ValueError('duplicate/out-of-order sample')
        self.last_arrival=arrival_time
        if not valid:
            self.mode='missing';self.previous_good=None;return
        x=np.asarray(position,float)
        if x.shape!=(3,) or not np.isfinite(x).all():raise ValueError('invalid coordinate')
        continuous=self.previous_good is not None and sample_time-self.previous_good[0]<=self.max_gap
        if continuous:
            self.v=(x-self.previous_good[1])/(sample_time-self.previous_good[0]);self.mode='tracking'
        else:
            self.v=np.zeros(3) if self.last_t is None else self.v*math.exp(-(sample_time-self.last_t)/self.tau)
            self.mode='recovered_single'
        self.last=x.copy();self.last_t=sample_time;self.previous_good=(sample_time,x.copy())
