import json
import pandas as pd,numpy as np
from request_value import BASE,OLD,weights

def main():
 out=BASE/'results/causal_decay_selector';events=pd.read_csv(out/'visible_features_and_labels.csv.gz');fits=json.loads((out/'fits.json').read_text());choices=pd.read_csv(out/'selection.csv');sc=json.loads((OLD/'results/S0/fold_scales.json').read_text());sources=sorted(events.dataset.unique());maxdiff=0.;checked=0
 for f in fits:
  held=f['held'];train=events[events.environment_id!=held];assert held not in f['fit_environments'];wt=weights(train)*train.fraction;wt=wt/wt.sum()
  def matrix(z):
   x=z[[f'f{k}' for k in range(6)]].to_numpy().copy();x[:,:3]/=np.array([sc[held][s] for s in z.dataset])[:,None];x=np.column_stack([x,*[(z.dataset==s).astype(float).to_numpy() for s in sources]]);return np.column_stack([np.ones(len(x)),(x-np.array(f['mu']))/np.array(f['sd'])])
  x=matrix(train);y=train[['loss0','loss1','loss2']].to_numpy();y=(y-y[:,1,None])/np.array([sc[held][s] for s in train.dataset])[:,None];reg=np.eye(x.shape[1])*np.sqrt(.1);reg[0,0]=0;a=np.vstack([np.sqrt(wt.to_numpy())[:,None]*x,reg]);b=np.vstack([np.sqrt(wt.to_numpy())[:,None]*y,np.zeros((x.shape[1],3))]);coef=np.linalg.lstsq(a,b,rcond=None)[0];maxdiff=max(maxdiff,float(np.max(abs(coef-np.array(f['coefficient'])))));ev=events[events.environment_id==held];pick=np.argmin(matrix(ev)@coef,axis=1);saved=choices.set_index(['window_id','stage']);assert all([.1,.2,.4][k]==saved.loc[(r.window_id,r.stage),'tau'] for r,k in zip(ev.itertuples(),pick));checked+=len(ev)
 assert maxdiff<1e-10
 baseline=pd.read_csv(out/'windows.csv.gz').query("method=='baseline'").set_index('window_id');old=pd.read_csv(BASE/'results/distance_aligned_return/windows.csv.gz').query("method=='secant'").set_index('window_id');delta=float(np.max(abs(baseline.nade-old.nade)));assert delta<1e-12
 audit=dict(passed=True,augmented_least_squares_coefficient_max_difference=maxdiff,choices_verified=checked,baseline_max_difference=delta,scope='independent ridge solve and saved-feature choices; raw feature causality and online integration pending');(out/'independent_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf8');print(json.dumps(audit))
if __name__=='__main__':main()
