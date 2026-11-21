"""Evidence-aware range output. No trajectory or future reference is accepted."""
from dataclasses import dataclass
import math
from risk_contract import assess_range

@dataclass(frozen=True)
class BlockEvidence:
    regime:str
    phase:str
    nominal:float
    calibration_environments:int
    radius:float|None

def block_assessment(*,mode,origin,evidence,tolerance=None):
    if evidence.regime!='natural_continuous':raise ValueError('block evidence is natural-continuous only')
    if not 0<evidence.nominal<1 or not math.isfinite(evidence.nominal):raise ValueError('invalid nominal coverage')
    m=evidence.calibration_environments
    if not isinstance(m,int) or isinstance(m,bool) or m<0:raise ValueError('invalid environment count')
    if origin not in {'before_any_invalid','after_observed_invalid'}:raise ValueError('unknown causal origin')
    expected='single_'+origin if mode=='recovered_single' else mode
    if mode=='uninitialized':return dict(radius=None,status='unavailable',guaranteed=False,target='environment_block_all_points')
    if evidence.phase!=expected:raise ValueError('state/evidence mismatch')
    k=math.ceil((m+1)*evidence.nominal-1e-12)
    if k>m:
        if evidence.radius is not None:raise ValueError('finite radius supplied without rank support')
        output=assess_range(observation_regime='natural_continuous',calibration_regime=evidence.regime,mode=mode,radius=None,tolerance=tolerance,supported=False)
        return dict(radius=None,status='no_finite_range',guaranteed=False,target='environment_block_all_points',calibration_environments=m,rank=k,base_status=output.status)
    if evidence.radius is None:raise ValueError('supported range lacks radius')
    output=assess_range(observation_regime='natural_continuous',calibration_regime=evidence.regime,mode=mode,radius=evidence.radius,tolerance=tolerance,supported=True)
    # Empirical block rank is descriptive until exchangeability/collection assumptions are justified.
    return dict(radius=output.radius,status='descriptive_block_range',guaranteed=False,target='environment_block_all_points',calibration_environments=m,rank=k,base_status=output.status)
