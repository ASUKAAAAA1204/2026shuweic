import json
import numpy as np,pandas as pd
from request_value import BASE,OLD

def main():
 out=BASE/'results/crossfit_corrected_selector';fits=json.loads((out/'fits.json').read_text());choices=pd.read_csv(out/'choices.csv').set_index(['window_id','stage']);sc=json.loads((OLD/'results/S0/fold_scales.json').read_text());maximum=0.;count=0
 for f in fits:
  held=f['held']
  for provenance in f['event_provenance']:
   assert {held,provenance['target']}==set(provenance['excluded']) and not set(provenance['excluded'])&set(provenance['fit_environments'])
  events=pd.read_csv(out/'events'/f'{held}.csv.gz');sources=sorted(events.dataset.unique());z=events[events.environment_id!=held];keys=['dataset','environment_id','episode_uuid','side'];wt=np.ones(len(z))/z.dataset.nunique()
  for k in range(1,len(keys)):wt/=z.groupby(keys[:k])[keys[k]].transform('nunique').to_numpy()
  wt/=z.groupby(keys).window_id.transform('size').to_numpy();wt*=z.fraction.to_numpy();wt/=wt.sum()
  def matrix(frame):
   x=frame[[f'f{k}' for k in range(6)]].to_numpy().copy();x[:,:3]/=np.array([sc[held][s] for s in frame.dataset])[:,None];return np.column_stack([x,*[(frame.dataset==s).astype(float).to_numpy() for s in sources]])
  x=matrix(z);mu=np.sum(wt[:,None]*x,axis=0);sd=np.sqrt(np.sum(wt[:,None]*(x-mu)**2,axis=0));sd[sd<1e-10]=1;assert np.max(abs(mu-f['mu']))<1e-10 and np.max(abs(sd-f['sd']))<1e-10;x=np.column_stack([np.ones(len(x)),(x-mu)/sd]);y=z[['loss0','loss1','loss2']].to_numpy();y=(y-y[:,1,None])/np.array([sc[held][s] for s in z.dataset])[:,None];pen=np.eye(x.shape[1])*np.sqrt(.1);pen[0,0]=0;coef=np.linalg.lstsq(np.vstack([np.sqrt(wt)[:,None]*x,pen]),np.vstack([np.sqrt(wt)[:,None]*y,np.zeros((x.shape[1],3))]),rcond=None)[0];maximum=max(maximum,float(np.max(abs(coef-f['coefficient']))))
 assert maximum<1e-10
 audit=dict(passed=True,folds=16,independent_group_weights=True,augmented_lstsq_coefficient_max_difference=maximum,training_provenance_groups=240,choices_not_independently_replayed=True,scope='independent fit from saved cross-fit events and exclusion metadata; not independent raw event generation or online replay');(out/'independent_fit_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf8');print(json.dumps(audit))
if __name__=='__main__':main()
