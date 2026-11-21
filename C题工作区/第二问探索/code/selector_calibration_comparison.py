import json
import numpy as np,pandas as pd
from request_value import BASE

def main():
 out=BASE/'results/selector_matched_calibration';rows=[];rng=np.random.default_rng(20261123)
 for scope in ['all','after_return']:
  data=pd.read_csv(out/(scope+'_environments.csv'));sources=sorted(data.dataset.unique());envs=sorted(data.environment_id.unique());metrics=['coverage','width','pinball'];models=['distance_return','causal_selector'];values=np.zeros((2,len(sources),len(envs),3));mask=np.zeros((len(sources),len(envs)))
  for r in data[data.model.isin(models)].itertuples():
   mi,si,ei=models.index(r.model),sources.index(r.dataset),envs.index(r.environment_id);values[mi,si,ei]=[getattr(r,m) for m in metrics];mask[si,ei]=1
  counts=rng.multinomial(len(envs),np.ones(len(envs))/len(envs),size=5000);den=counts@mask.T;valid=(den>0).all(axis=1);v=np.einsum('be,ksem->bksm',counts[valid],values)/den[valid,None,:,None];delta=(v[:,1]-v[:,0]).mean(axis=1);point=((values[1]-values[0]).sum(axis=1)/mask.sum(axis=1)[:,None]).mean(axis=0)
  for k,m in enumerate(metrics):rows.append(dict(scope=scope,metric=m,candidate_minus_distance=float(point[k]),lo95=float(np.quantile(delta[:,k],.025)),hi95=float(np.quantile(delta[:,k],.975)),valid_draws=int(valid.sum()),requested_draws=5000))
 result=pd.DataFrame(rows);result.to_csv(out/'paired_calibration_intervals.csv',index=False);(out/'paired_interval_scope.json').write_text(json.dumps(dict(unit='shared global environment',conditional_on_all_sources=True,refit_bootstrap=False,multiple_search_corrected=False,seed=20261123),indent=2),encoding='utf8');print(result.to_string(index=False))
if __name__=='__main__':main()
