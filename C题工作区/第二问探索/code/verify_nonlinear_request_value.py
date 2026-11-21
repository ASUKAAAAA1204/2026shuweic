from pathlib import Path
import json
import numpy as np,pandas as pd
from request_value import BASE,OLD,weights,CAND
p=BASE/'results/nonlinear_request_value';inp=BASE/'results/request_value';trees=json.loads((p/'trees.json').read_text());h=pd.read_csv(inp/'history_features.csv');raw=pd.read_csv(inp/'candidate_windows.csv.gz');pr=pd.read_csv(p/'oof_predictions.csv');sc=json.loads((OLD/'results/S0/fold_scales.json').read_text());pivot=raw.pivot(index='window_id',columns='method',values='raw_ade');sources=sorted(h.dataset.unique());leaf_count=0;maxerr=0.
for fold in trees:
 held=fold['held'];fit=h[h.environment_id!=held];ev=h[h.environment_id==held];assert held not in fold['fit_environments'];assert set(fit.environment_id)==set(fold['fit_environments'])
 def matrix(f):
  x=f[[f'f{k}' for k in range(7)]].to_numpy().copy();x[:,:4]/=np.array([sc[held][s] for s in f.dataset])[:,None]
  return np.column_stack([x,*[(f.dataset==s).astype(float).to_numpy() for s in sources]])
 x=matrix(fit);xe=matrix(ev);w=weights(fit);yy=pivot.loc[fit.window_id];y=np.column_stack([(yy.fixed-yy[c]).to_numpy() for c in ['early','late']])/np.array([sc[held][s] for s in fit.dataset])[:,None]
 def check(node,ix,depth):
  global leaf_count,maxerr
  assert depth<=2 and node['count']==len(ix)
  mean=(y[ix]*w[ix,None]).sum(axis=0)/w[ix].sum();err=float(np.abs(mean-node['mean']).max());maxerr=max(maxerr,err);assert err<1e-12
  if 'feature' in node:
   mask=x[ix,node['feature']]<=node['threshold'];assert mask.sum()>=60 and (~mask).sum()>=60
   check(node['left'],ix[mask],depth+1);check(node['right'],ix[~mask],depth+1)
  else:leaf_count+=1
 check(fold['tree'],np.arange(len(fit)),0)
 # Independent vectorized routing into leaves.
 expected=np.empty((len(ev),2))
 def route(node,ix):
  if 'feature' not in node:expected[ix]=node['mean'];return
  m=xe[ix,node['feature']]<=node['threshold'];route(node['left'],ix[m]);route(node['right'],ix[~m])
 route(fold['tree'],np.arange(len(ev)))
 saved=pr[pr.held==held].set_index('window_id').loc[ev.window_id];assert np.allclose(expected,saved[['early_gain','late_gain']],atol=1e-12)
 choices=np.array(CAND)[np.argmax(np.column_stack([expected[:,0],np.zeros(len(ev)),expected[:,1]]),axis=1)];assert (choices==saved.choice).all()
result=dict(passed=True,folds=len(trees),leaves=leaf_count,leaf_means_max_error=maxerr,held_exclusion_verified=True,depth_and_min_leaf_verified=True,independent_routing_verified=True,scope='saved tree nodes, weights, predictions and decisions; not independent split optimization')
(p/'independent_audit.json').write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps(result));print(pd.read_csv(p/'sources.csv').query("method in ['fixed','history_tree']").to_string(index=False));print(pd.read_csv(p/'choices.csv').to_string(index=False))
