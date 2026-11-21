"""Causal origin marker and transactional validation; original interface preserved."""
import math
import numpy as np
from recovery_interface import RecoveryTracker

class OriginRecoveryTracker(RecoveryTracker):
    def __init__(self, max_gap_s=.12, tau=.2):
        if not math.isfinite(max_gap_s) or max_gap_s<=0 or not math.isfinite(tau) or tau<=0:
            raise ValueError('positive finite model parameters required')
        super().__init__(max_gap_s,tau)
        self.last_received_sample_time=-math.inf
        self.past_invalid_seen=False

    @property
    def origin(self):
        return 'after_observed_invalid' if self.past_invalid_seen else 'before_any_invalid'

    def predict(self,now,**kwargs):
        if not math.isfinite(now) or now<self.last_arrival:
            raise ValueError('prediction precedes received information or has nonfinite time')
        return super().predict(now,**kwargs)

    def receive(self,sample_time,position,valid,arrival_time):
        if not math.isfinite(sample_time) or not math.isfinite(arrival_time):
            raise ValueError('finite times required')
        if sample_time<=self.last_received_sample_time or arrival_time<self.last_arrival or sample_time>arrival_time:
            raise ValueError('duplicate, out-of-order or future sample')
        if valid:
            x=np.asarray(position,float)
            if x.shape!=(3,) or not np.isfinite(x).all():
                raise ValueError('finite three dimensional position required')
        super().receive(sample_time,position,valid,arrival_time)
        self.last_received_sample_time=sample_time
        if not valid:self.past_invalid_seen=True
