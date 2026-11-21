from pathlib import Path
import sys,json
import numpy as np,pandas as pd
BASE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(BASE.parent/'第二问求解/code'))
from update_calibration_experiment import hierarchical

def main():
 inp=BASE/'results/distance_nested_calibration';out=BASE/'results/neighbor_age';out.mkdir(exist_ok=True)
 cal=pd.read_csv(inp/'inner_residuals.csv.gz');ev=pd.read_csv(inp/'outer_points.csv.gz');support=pd.read_csv(inp/'support.csv');sc=json.loads((BASE.parent/'第二问求解/results/S0/fold_scales.json').read_text());groups={k:g for k,g in cal.groupby(['outer','model','dataset','after'])};rows=[];proof=[];maxunchanged=0.
 for r in support.itertuples():
  pool=groups[(r.outer,r.model,r.dataset,r.after)];assert r.outer not in set(pool.environment_id)
  distance=0;local=pool[pool.age_bin==r.age_bin]
  while len(local)<100 and distance<29:
   distance+=1;local=pool[abs(pool.age_bin-r.age_bin)<=distance]
  supported=len(local)>=100
  pr=float(local.error.quantile(.9,interpolation='higher')) if supported else np.nan
  ee=[float(g.error.quantile(.9,interpolation='higher')) for _,g in local.groupby('environment_id') if len(g)>=20]
  radii={'pooled':pr,'env_mean':float(np.mean(ee)) if supported and len(ee)>=2 else pr}
  proof.append(dict(outer=r.outer,model=r.model,dataset=r.dataset,after=r.after,age_bin=r.age_bin,distance=distance,points=len(local),supported=supported,original_fallback=r.age_fallback))
  select=ev[(ev.environment_id==r.outer)&(ev.model==r.model)&(ev.dataset==r.dataset)&(ev.after==r.after)&(ev.age_bin==r.age_bin)]
  for method,rad in radii.items():
   z=select[select.calibration==method].copy();assert len(z)>0
   if not r.age_fallback:
    d=float(abs(z.radius-rad).max());maxunchanged=max(maxunchanged,d);assert d<1e-12
   z['original_fallback']=bool(r.age_fallback);z['supported']=supported;z['radius']=rad;delta=z.error-rad;z['coverage']=(delta<=0).astype(float) if supported else np.nan;z['width']=2*rad/sc[r.outer][r.dataset];z['pinball']=np.maximum(.9*delta,-.1*delta)/sc[r.outer][r.dataset];z['variant']='neighbor';rows.append(z)
 result=pd.concat(rows,ignore_index=True);assert len(result)==len(ev)
 original=ev.merge(support.rename(columns={'outer':'environment_id','age_fallback':'original_fallback'})[['environment_id','model','dataset','after','age_bin','original_fallback']],on=['environment_id','model','dataset','after','age_bin'],validate='many_to_one');original['variant']='all_age';original['supported']=True
 a=pd.concat([original,result],ignore_index=True);summaries=[]
 for scope,z in [('all',a),('after_return',a[a.after]),('original_fallback',a[a.original_fallback])]:
  z=z.groupby(['variant','model','calibration','window_id','dataset','environment_id','episode_uuid','side'],as_index=False)[['coverage','width','pinball','supported']].mean();s=hierarchical(z,['variant','model','calibration'],['coverage','width','pinball','supported']);s['scope']=scope;summaries.append(s)
 summary=pd.concat(summaries,ignore_index=True);summary.to_csv(out/'summary.csv',index=False);result.to_csv(out/'points.csv.gz',index=False);pd.DataFrame(proof).to_csv(out/'support.csv',index=False)
 audit=dict(passed=True,rows=len(result),support_groups=len(proof),unsupported=sum(not r['supported'] for r in proof),adequate_age_radius_max_difference=maxunchanged,development_reads=0,test_reads=0,formal_model_changed=False)
 (out/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf8');print(summary.query("model=='distance_return' and calibration=='pooled'").to_string(index=False));print(json.dumps(audit))
if __name__=='__main__':main()
