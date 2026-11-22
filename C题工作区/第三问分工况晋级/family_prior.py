"""No-feedback ablation of the frozen congestion/urgency policy."""
import numpy as np
from baseline import action
from deadline_policy import choose,pressure


def dispatch(obs,pools):
    if pressure(obs,pools,{}):
        return choose(obs,pools,{})
    return action(obs,{f:float(np.median(v)) for f,v in pools.items()},"density","minimum")
