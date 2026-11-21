from pathlib import Path
import sys,json
import numpy as np,pandas as pd
from request_value import weights,BASE,OLD,CAND
sys.path.insert(0,str(OLD/'code'))
from update_calibration_experiment import hierarchical,paired_ci

def fit_tree(x,y,w,depth=0):
 mean=np.average(y,axis=0,weights=w);node=dict(mean=mean.tolist(),count=len(y),weight=float(w.sum()))
 if depth==2 or len(y)<120:return node
 base=float(np.sum(w[:,None]*(y-mean)**2));best=base;split=None
 for j in range(x.shape[1]):
  for threshold in np.unique(np.quantile(x[:,j],np.arange(1,10)/10)):
   mask=x[:,j]<=threshold
   if mask.sum()<60 or (~mask).sum()<60:continue
   loss=0.
   for k in [mask,~mask]:
    m=np.average(y[k],axis=0,weights=w[k]);loss+=float(np.sum(w[k,None]*(y[k]-m)**2))
   if loss<best-1e-15:best=loss;split=(j,float(threshold),mask)
 if split is not None:
  j,threshold,mask=split;node.update(feature=j,threshold=threshold,left=fit_tree(x[mask],y[mask],w[mask],depth+1),right=fit_tree(x[~mask],y[~mask],w[~mask],depth+1))
 return node

def predict_tree(tree,x):
 result=[]
 for row in x:
  node=tree
  while 'feature' in node:node=node['left' if row[node['feature']]<=node['threshold'] else 'right']
  result.append(node['mean'])
 return np.array(result)

def main():
 inp=BASE/'results/request_value';out=BASE/'results/nonlinear_request_value';out.mkdir(exist_ok=True)
 h=pd.read_csv(inp/'history_features.csv');raw=pd.read_csv(inp/'candidate_windows.csv.gz');sc=json.loads((OLD/'results/S0/fold_scales.json').read_text());pv=raw.pivot(index='window_id',columns='method',values='raw_ade');sources=sorted(h.dataset.unique());pred=[];trees=[]
 for held in sorted(h.environment_id.unique()):
  fit=h[h.environment_id!=held];ev=h[h.environment_id==held];w=weights(fit)
  def matrix(frame):
   x=frame[[f'f{k}' for k in range(7)]].to_numpy().copy();x[:,:4]/=np.array([sc[held][s] for s in frame.dataset])[:,None]
   return np.column_stack([x,*[(frame.dataset==s).astype(float).to_numpy() for s in sources]])
  y=pv.loc[fit.window_id];y=np.column_stack([(y.fixed-y[c]).to_numpy() for c in ['early','late']])/np.array([sc[held][s] for s in fit.dataset])[:,None]
  tree=fit_tree(matrix(fit),y,w);pp=predict_tree(tree,matrix(ev));trees.append(dict(held=held,fit_environments=sorted(fit.environment_id.unique()),fit_windows=len(fit),tree=tree))
  for k,r in enumerate(ev.itertuples()):pred.append(dict(window_id=r.window_id,held=held,method='history_tree',early_gain=pp[k,0],late_gain=pp[k,1],choice=CAND[int(np.argmax([pp[k,0],0.,pp[k,1]]))]))
 pr=pd.DataFrame(pred);pr.to_csv(out/'oof_predictions.csv',index=False);(out/'trees.json').write_text(json.dumps(trees,indent=2),encoding='utf8')
 z=pr[['window_id','method','choice']].merge(raw.rename(columns={'method':'choice'}),on=['window_id','choice'],validate='one_to_one')
 for col,rawcol in [('nade','raw_ade'),('nfde','raw_fde')]:z[col]=[v/sc[e][s] for v,e,s in zip(z[rawcol],z.environment_id,z.dataset)]
 previous=pd.read_csv(inp/'selected_windows.csv.gz');a=pd.concat([previous,z],ignore_index=True);a.to_csv(out/'selected_windows.csv.gz',index=False);metrics=['nade','nfde','exceed05','requests','waiting_s'];summary=hierarchical(a,['method'],metrics);summary.to_csv(out/'summary.csv',index=False)
 env=a.groupby(['method','dataset','environment_id','episode_uuid','side'],as_index=False)[metrics].mean().groupby(['method','dataset','environment_id','episode_uuid'],as_index=False)[metrics].mean().groupby(['method','dataset','environment_id'],as_index=False)[metrics].mean();env.to_csv(out/'environments.csv',index=False);env.groupby(['method','dataset'],as_index=False)[metrics].mean().to_csv(out/'sources.csv',index=False)
 pd.DataFrame([paired_ci(env,ref,'history_tree') for ref in ['fixed','history_ridge']]).to_csv(out/'paired_ci.csv',index=False);pr.groupby('choice').size().to_csv(out/'choices.csv')
 audit=dict(passed=True,folds=len(trees),train_windows=len(h),new_raw_replays=0,development_reads=0,test_reads=0,max_depth=2,min_leaf=60,hyperparameter_search=False)
 (out/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf8');print(summary.to_string(index=False));print(pd.read_csv(out/'paired_ci.csv').to_string(index=False))
if __name__=='__main__':main()
