"""Risk range contract: empirical descriptions must match the observation regime."""
from dataclasses import dataclass
import math

@dataclass(frozen=True)
class RangeAssessment:
    radius: float|None
    status: str
    guaranteed: bool=False


def assess_range(*,observation_regime,calibration_regime,mode,radius,tolerance,supported):
    allowed={'paid_isolated','natural_continuous'}
    if observation_regime not in allowed or calibration_regime not in allowed:
        raise ValueError('unknown observation regime')
    if tolerance is not None and (not math.isfinite(tolerance) or tolerance<=0):
        raise ValueError('positive finite tolerance required')
    if radius is not None and (not math.isfinite(radius) or radius<0):
        raise ValueError('nonnegative finite radius required')
    if observation_regime!=calibration_regime:return RangeAssessment(None,'regime_mismatch')
    if mode=='uninitialized':return RangeAssessment(None,'unavailable')
    if mode not in {'tracking','missing','recovered_single','paid_prior','paid_returned'}:
        raise ValueError('unknown state mode')
    legal={'paid_isolated':{'paid_prior','paid_returned'},'natural_continuous':{'tracking','missing','recovered_single'}}
    if mode not in legal[observation_regime]:return RangeAssessment(None,'mode_mismatch')
    if not supported or radius is None:return RangeAssessment(None,'unsupported')
    if observation_regime=='natural_continuous' and mode in {'missing','recovered_single'}:
        return RangeAssessment(radius,'recovery_uncertified')
    if tolerance is None:return RangeAssessment(radius,'descriptive_only')
    return RangeAssessment(radius,'empirical_within' if radius<=tolerance else 'empirical_exceeds')
