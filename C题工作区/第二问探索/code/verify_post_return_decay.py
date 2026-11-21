import json
import pandas as pd,numpy as np
from request_value import BASE,OLD

def reduce(f,by,metric):
 z=f.groupby(by+['dataset','environment_id','episode_uuid','side'],as_index=False)[metric].mean()
 for keys in [['dataset','environment_id','episode_uuid'],['dataset','environment_id'],['dataset'],[]]:z=z.groupby(by+keys,as_index=False)[metric].mean()
 return z

def main():
 out=BASE/'results/post_return_decay';raw=pd.read_csv(out/'candidate_windows.csv.gz');choices=pd.read_csv(out/'choices.csv');selected=pd.read_csv(out/'selected_windows.csv.gz');sc=json.loads((OLD/'results/S0/fold_scales.json').read_text());assert not raw.duplicated(['window_id','tau']).any();checks=0
 for r in choices.itertuples():
  fit=raw[raw.environment_id!=r.held].copy();assert r.held not in set(fit.environment_id);fit['score']=fit.raw_ade/[sc[r.held][s] for s in fit.dataset];rank=reduce(fit,['tau'],'score');rank['priority']=rank.tau.map({.2:0,.1:1,.4:2});best=float(rank.sort_values(['score','priority']).iloc[0].tau);assert best==r.tau
  a=selected[(selected.method=='train_selected')&(selected.environment_id==r.held)].sort_values('window_id');b=raw[(raw.environment_id==r.held)&(raw.tau==best)].sort_values('window_id');assert list(a.window_id)==list(b.window_id);assert np.allclose(a.nade,b.nade,rtol=0,atol=1e-15);checks+=1
 fixed=reduce(raw,['tau'],'nade');fixed.to_csv(out/'fixed_tau_summary.csv',index=False)
 # Deliberately uses true outcomes and therefore is diagnostic only.
 oracle=raw.loc[raw.groupby('window_id').nade.idxmin()].copy();oracle['method']='posthoc_window_oracle';base=raw[raw.tau==.2].copy();base['method']='baseline';summary=reduce(pd.concat([base,oracle]),['method'],'nade');summary.to_csv(out/'oracle_diagnostic.csv',index=False);v=summary.set_index('method').nade;gain=100*(1-v['posthoc_window_oracle']/v['baseline']);audit=dict(passed=True,independent_choice_checks=checks,fixed_tau_summary=fixed.to_dict('records'),posthoc_oracle_gain_pct=gain,oracle_not_deployable=True,scope='independent hierarchical score reduction and selection provenance; upstream replay not independently reimplemented');(out/'independent_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf8');print(json.dumps(audit))
if __name__=='__main__':main()
