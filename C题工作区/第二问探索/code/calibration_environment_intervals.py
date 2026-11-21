import json
import numpy as np,pandas as pd
from request_value import BASE

def main():
 out=BASE/'results/calibration_environment_intervals';out.mkdir(exist_ok=True)
 original=pd.read_csv(BASE/'results/distance_nested_calibration/all_environments.csv').query("calibration=='pooled'").copy();original['method']=original.model+'__pooled'
 weighted=pd.read_csv(BASE/'results/hierarchy_weighted_calibration/all_environments.csv').query("method!='pooled'")
 f=pd.concat([original,weighted],ignore_index=True);sources=sorted(f.dataset.unique());envs=sorted(f.environment_id.unique());methods=sorted(f.method.unique());metrics=['coverage','width','pinball'];rng=np.random.default_rng(20261122);counts=rng.multinomial(len(envs),np.ones(len(envs))/len(envs),size=5000);draws={};rows=[]
 for method in methods:
  z=f[f.method==method];values=np.zeros((len(sources),len(envs),len(metrics)));mask=np.zeros((len(sources),len(envs)))
  for r in z.itertuples():
   i,j=sources.index(r.dataset),envs.index(r.environment_id);assert mask[i,j]==0;mask[i,j]=1;values[i,j]=[getattr(r,m) for m in metrics]
  den=counts@mask.T;valid=(den>0).all(axis=1);num=np.einsum('be,sem->bsm',counts,values);bs=np.mean(num[valid]/den[valid,:,None],axis=1);draws[method]=bs
  point=np.mean(values.sum(axis=1)/mask.sum(axis=1)[:,None],axis=0)
  for k,metric in enumerate(metrics):rows.append(dict(method=method,metric=metric,estimate=point[k],lo95=float(np.quantile(bs[:,k],.025)),hi95=float(np.quantile(bs[:,k],.975)),valid_draws=int(valid.sum()),requested_draws=5000))
  for i,src in enumerate(sources):
   sv=den[:,i]>0;sb=num[sv,i,0]/den[sv,i];rows.append(dict(method=method,metric='coverage__'+src,estimate=float(values[i,:,0].sum()/mask[i].sum()),lo95=float(np.quantile(sb,.025)),hi95=float(np.quantile(sb,.975)),valid_draws=int(sv.sum()),requested_draws=5000))
 pd.DataFrame(rows).to_csv(out/'intervals.csv',index=False)
 paired=[]
 for a,b in [('secant__pooled','distance_return__pooled'),('point_cdf','hierarchy_weighted')]:
  delta=draws[b]-draws[a]
  for k,metric in enumerate(metrics):paired.append(dict(control=a,candidate=b,metric=metric,mean_delta=float(delta[:,k].mean()),lo95=float(np.quantile(delta[:,k],.025)),hi95=float(np.quantile(delta[:,k],.975))))
 pd.DataFrame(paired).to_csv(out/'paired.csv',index=False);audit=dict(global_environments=len(envs),source_environment_counts=f[f.method==methods[0]].groupby('dataset').environment_id.nunique().to_dict(),draws=5000,seed=20261122,conditional_on_all_sources_present=True,scope='fixed saved OOF environment scores, not full refit bootstrap, nominal exploratory intervals, no multiple-search correction',formal_model_changed=False);(out/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf8');print(pd.DataFrame(rows).query("metric=='coverage'").to_string(index=False));print(pd.DataFrame(paired).to_string(index=False))
if __name__=='__main__':main()
