import json
import numpy as np,pandas as pd
from request_value import BASE,OLD

def main():
 out=BASE/'results/purchase_matched_calibration';events=pd.read_csv(BASE/'results/causal_decay_selector/visible_features_and_labels.csv.gz').query('stage==0');fits=json.loads((out/'pair_fits.json').read_text());raw=pd.read_csv(BASE/'results/second_request_value/candidate_windows.csv.gz');loss=raw.pivot(index='window_id',columns='method',values='raw_ade');sc=json.loads((OLD/'results/S0/fold_scales.json').read_text());sources=sorted(events.dataset.unique());maximum=0.
 for f in fits:
  scale=f['scale'];z=events[~events.environment_id.isin(f['excluded'])];assert not set(f['excluded'])&set(z.environment_id);keys=['dataset','environment_id','episode_uuid','side'];wt=np.ones(len(z))/z.dataset.nunique()
  for k in range(1,len(keys)):wt/=z.groupby(keys[:k])[keys[k]].transform('nunique').to_numpy()
  wt/=z.groupby(keys).window_id.transform('size').to_numpy();wt/=wt.sum();x=z[[f'f{k}' for k in range(5)]].to_numpy();x[:,:3]/=np.array([scale[s] for s in z.dataset])[:,None];x=np.column_stack([x,*[(z.dataset==s).astype(float).to_numpy() for s in sources]]);mu=np.sum(wt[:,None]*x,axis=0);sd=np.sqrt(np.sum(wt[:,None]*(x-mu)**2,axis=0));sd[sd<1e-10]=1;assert np.max(abs(mu-f['mu']))<1e-10 and np.max(abs(sd-f['sd']))<1e-10;x=np.column_stack([np.ones(len(x)),(x-mu)/sd]);y=np.array([(loss.loc[r.window_id,'first_only']-loss.loc[r.window_id,'fixed2'])/scale[r.dataset] for r in z.itertuples()]);reg=np.eye(x.shape[1])*np.sqrt(.1);reg[0,0]=0;coef=np.linalg.lstsq(np.vstack([np.sqrt(wt)[:,None]*x,reg]),np.r_[np.sqrt(wt)*y,np.zeros(x.shape[1])],rcond=None)[0];maximum=max(maximum,float(np.max(abs(coef-f['coefficient']))))
 assert maximum<1e-10
 audit=dict(passed=True,folds=len(fits),independent_group_count_weights=True,augmented_lstsq_coefficient_max_difference=maximum,both_excluded_environments_verified=True,scope='saved causal features and replay labels, not independent extraction');(out/'independent_pair_fit_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf8');print(json.dumps(audit))
if __name__=='__main__':main()
