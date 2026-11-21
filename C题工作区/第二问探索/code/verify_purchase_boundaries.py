import json
import numpy as np
from request_value import BASE
from candidate_replay_engine import replay
from online_second_purchase import SecondPurchasePolicy

def main():
 sources=['EgoDemo','EgoPro','EgoStandard'];fit=dict(mu=[0.]*8,sd=[1.]*8,coefficient=[.03]+[0.]*8);t=np.arange(31)/30;x=np.column_stack([t,t*t,np.zeros(len(t))]);ht=np.array([-2/30,-1/30]);hx=np.column_stack([ht,ht*ht,np.zeros(2)]);kwargs=dict(initial_velocity=(hx[-1]-hx[-2])/(ht[-1]-ht[-2]),initial_time=ht[-1],fit=fit,scale=1.,source='EgoDemo',sources=sources)
 def run(valid=None,budget=2,end=1.):
  p=SecondPurchasePolicy(**kwargs);mask=t<=end+1e-12;r=replay(t[mask],x[mask],ht,hx,p,delay=.1,budget=budget,valid=None if valid is None else valid[mask],window_end=float(t[mask][-1]));return p,r
 cases={};p,r=run();assert p.buy and r['metrics']['requests']==2 and r['metrics']['successful_returns']==2;cases['valid_buy']=True
 valid=np.ones(len(t),bool);valid[9]=False;p,r=run(valid);assert p.gain is None and p.buy is False and r['metrics']['requests']==1 and r['metrics']['successful_returns']==0;cases['invalid_first_no_untrained_retry']=True
 p,r=run(budget=0);assert r['metrics']['requests']==0;cases['zero_budget']=True
 p,r=run(budget=1);assert r['metrics']['requests']==1 and not p.second_decided;cases['one_budget']=True
 p,r=run(end=.65);assert r['metrics']['requests']==1 and not p.second_decided;cases['no_second_arrival_before_end']=True
 p,r=run(end=.7);assert r['metrics']['requests']==2 and r['metrics']['successful_returns']==2;cases['exact_final_arrival']=True
 # Model evidence is snapshotted before caller mutation.
 policy=SecondPurchasePolicy(**kwargs);fit['coefficient'][0]=-99;assert policy.fit['coefficient'][0]==.03;fit['coefficient'][0]=.03;cases['caller_model_mutation_isolated']=True
 rejected=0
 for patch in [dict(scale=0),dict(cost_weight=float('nan')),dict(source='unknown'),dict(initial_velocity=[0,float('inf'),0]),dict(fit=dict(mu=[0.]*8,sd=[0.]*8,coefficient=[.03]+[0.]*8))]:
  k=dict(kwargs);k.update(patch)
  try:SecondPurchasePolicy(**k)
  except ValueError:rejected+=1
  else:raise AssertionError(patch)
 audit=dict(passed=True,cases=cases,invalid_constructors_rejected=rejected,scope='synthetic protocol boundary tests, not invalid-observation performance evidence');out=BASE/'results/second_purchase_online';(out/'boundary_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf8');print(json.dumps(audit))
if __name__=='__main__':main()
