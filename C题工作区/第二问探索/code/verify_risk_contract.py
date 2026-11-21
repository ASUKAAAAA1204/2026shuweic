from pathlib import Path
import json,hashlib
from risk_contract import assess_range
BASE=Path(__file__).resolve().parents[1]
def main():
 args=dict(observation_regime='paid_isolated',calibration_regime='paid_isolated',mode='paid_returned',radius=.02,tolerance=.05,supported=True)
 checks=[]
 def check(name,expected,**change):
  a=assess_range(**{**args,**change});assert a.status==expected and not a.guaranteed;checks.append(name)
 check('paid descriptive comparison','empirical_within')
 check('no absolute guarantee when too wide','empirical_exceeds',radius=.1)
 check('cross-regime blocked','regime_mismatch',calibration_regime='natural_continuous')
 check('wrong mode blocked','mode_mismatch',mode='tracking')
 check('missing calibration refused','unsupported',supported=False)
 check('no radius refused','unsupported',radius=None)
 check('uninitialized unavailable','unavailable',mode='uninitialized')
 for mode in ['missing','recovered_single']:
  check(mode+' never certified','recovery_uncertified',observation_regime='natural_continuous',calibration_regime='natural_continuous',mode=mode,radius=.0001)
 for change in [dict(radius=-1),dict(radius=float('nan')),dict(tolerance=0),dict(tolerance=float('inf'))]:
  try:assess_range(**{**args,**change})
  except ValueError:checks.append('invalid numeric input rejected')
  else:raise AssertionError('invalid range accepted')
 files=['results/environment_calibration/summary.csv','results/causal_calibration/summary.csv','results/origin_reproduction/summary.csv','results/resource_curve_safe/independent_audit.json','results/stage_diagnostic/independent_audit.json']
 evidence=[dict(path=f,sha256=hashlib.sha256((BASE/f).read_bytes()).hexdigest()) for f in files]
 frozen=hashlib.sha256((BASE.parent/'第二问求解/frozen_model.json').read_bytes()).hexdigest();assert frozen=='baf3ce889dc51d62b331a8d30aeee34242109397b1e27a1562df391e908a1af3'
 out=BASE/'results/risk_contract';out.mkdir(exist_ok=True);audit=dict(passed=True,checks=checks,evidence=evidence,formal_model_sha256=frozen,scope='risk labeling and evidence integrity; not coverage certification',development_reads=0,test_reads=0)
 (out/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf8');print(json.dumps(audit))
if __name__=='__main__':main()
