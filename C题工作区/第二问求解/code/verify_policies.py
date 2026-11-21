"""Public-policy tests: same visible state implies same decision, regardless of hidden reference."""
from pathlib import Path
import json
import numpy as np
from engine import replay
from policies import RadiusPolicy,BudgetPolicy

def main():
 out=Path(__file__).resolve().parents[1]/'results/S1';rng=np.random.default_rng(41);tests=[]
 for family in ['threshold','delay_aware','budget']:
  def policy():return BudgetPolicy(np.linspace(.01,.2,30),.5,.01,.1) if family=='budget' else RadiusPolicy(np.linspace(.01,.2,30),.5,.05,.1,family=='delay_aware')
  for trial in range(50):
   t=np.arange(31)/30;ht=np.array([-.06,-.03]);hx=rng.normal(size=(2,3));x=rng.normal(size=(31,3));a=replay(t,x,ht,hx,policy(),delay=.1)
   asked={round(p['sample_time'],10) for p in a['actions'] if p['event']=='request'};y=x.copy()
   for k,tt in enumerate(t):
    if round(tt,10) not in asked:y[k]+=rng.normal(size=3)*100
   b=replay(t,y,ht,hx,policy(),delay=.1)
   assert a['actions']==b['actions'] and [p['prediction'] for p in a['points']]==[p['prediction'] for p in b['points']]
   assert a['metrics']['cost']<=2 and all(p['pending_until'] is None or p['pending_until']>=p['time']-1e-12 for p in a['points'])
  tests.append(dict(family=family,perturbation_tests=50,passed=True))
 (out/'adaptive_policy_checks.json').write_text(json.dumps(tests,indent=2),encoding='utf8');print(json.dumps(tests))
if __name__=='__main__':main()
