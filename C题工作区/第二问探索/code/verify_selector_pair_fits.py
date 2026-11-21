import json
import numpy as np,pandas as pd
from request_value import BASE

def main():
 out=BASE/'results/selector_matched_calibration';events=pd.read_csv(BASE/'results/causal_decay_selector/visible_features_and_labels.csv.gz');fits=json.loads((out/'pair_fits.json').read_text());sources=sorted(events.dataset.unique());maximum=0.;mudiff=0.;sddiff=0.
 for f in fits:
  z=events[~events.environment_id.isin(f['excluded'])];assert not set(f['excluded'])&set(z.environment_id);keys=['dataset','environment_id','episode_uuid','side'];weight=np.ones(len(z))/z.dataset.nunique()
  for k in range(1,len(keys)):weight/=z.groupby(keys[:k])[keys[k]].transform('nunique').to_numpy()
  weight/=z.groupby(keys).window_id.transform('size').to_numpy();weight*=z.fraction.to_numpy();weight/=weight.sum();x=z[[f'f{k}' for k in range(6)]].to_numpy();x[:,:3]/=np.array([f['scale'][s] for s in z.dataset])[:,None];x=np.column_stack([x,*[(z.dataset==s).astype(float).to_numpy() for s in sources]]);mu=np.sum(x*weight[:,None],axis=0);sd=np.sqrt(np.sum((x-mu)**2*weight[:,None],axis=0));sd[sd<1e-10]=1;mudiff=max(mudiff,float(abs(mu-f['mu']).max()));sddiff=max(sddiff,float(abs(sd-f['sd']).max()));x=np.column_stack([np.ones(len(x)),(x-mu)/sd]);y=z[['loss0','loss1','loss2']].to_numpy();y=(y-y[:,1,None])/np.array([f['scale'][s] for s in z.dataset])[:,None];reg=np.eye(x.shape[1])*np.sqrt(.1);reg[0,0]=0;a=np.vstack([np.sqrt(weight)[:,None]*x,reg]);target=np.vstack([np.sqrt(weight)[:,None]*y,np.zeros((x.shape[1],3))]);coef=np.linalg.lstsq(a,target,rcond=None)[0];maximum=max(maximum,float(abs(coef-f['coefficient']).max()))
 assert max(maximum,mudiff,sddiff)<1e-10
 audit=dict(passed=True,pair_fits_recomputed=len(fits),independent_group_count_weights=True,augmented_lstsq=True,coefficient_max_difference=maximum,mean_max_difference=mudiff,std_max_difference=sddiff,scope='independent fitting from saved causal event features and labels, not independent raw event extraction');(out/'independent_pair_fit_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf8');print(json.dumps(audit))
if __name__=='__main__':main()
