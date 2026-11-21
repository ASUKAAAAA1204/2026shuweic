from pathlib import Path
import sys,json
import numpy as np,pandas as pd
BASE=Path(__file__).resolve().parents[1];OLD=BASE.parent/'第二问求解';sys.path.insert(0,str(OLD/'code'))
from update_calibration_experiment import quantiles,hierarchical,paired_ci

def main():
 out=BASE/'results/environment_calibration';out.mkdir(exist_ok=True);p=pd.read_csv(OLD/'results/S3/residual_points.csv.gz');op=p[p.method=='open'];post=p[(p.method=='secant')&p.after];rows=[];support=[]
 for held in sorted(post.environment_id.unique()):
  fit=post[post.environment_id!=held];prior=quantiles(op[op.environment_id!=held]);ev=post[post.environment_id==held].copy();assert held not in set(fit.environment_id)
  maps={m:{} for m in ['open','pooled','env_mean','env_q75']}
  for src,age in prior:
   b=fit[(fit.dataset==src)&(fit.age_bin==age)];pooled=float(b.error.quantile(.9,interpolation='higher')) if len(b)>=100 else prior[(src,age)]
   values=[float(g.error.quantile(.9,interpolation='higher')) for _,g in b.groupby('environment_id') if len(g)>=20];support.append(dict(held=held,dataset=src,age_bin=age,eligible_envs=len(values),points=len(b)))
   maps['open'][(src,age)]=prior[(src,age)];maps['pooled'][(src,age)]=pooled;maps['env_mean'][(src,age)]=float(np.mean(values)) if len(values)>=2 else pooled;maps['env_q75'][(src,age)]=float(np.quantile(values,.75,method='higher')) if len(values)>=2 else pooled
  for method,rr in maps.items():
   z=ev.copy();rad=np.array([rr[(s,int(a))] for s,a in zip(z.dataset,z.age_bin)]);d=z.error-rad;z['coverage']=(d<=0).astype(float);z['width']=2*rad/z.scale;z['pinball']=np.maximum(.9*d,-.1*d)/z.scale;z['method']=method;z=z.groupby(['method','window_id','dataset','environment_id','episode_uuid','side'],as_index=False)[['coverage','width','pinball']].mean();rows.append(z)
 m=pd.concat(rows,ignore_index=True);m.to_csv(out/'windows.csv',index=False);s=hierarchical(m,['method'],['coverage','width','pinball']);s.to_csv(out/'summary.csv',index=False)
 e=m.groupby(['method','dataset','environment_id','episode_uuid','side'],as_index=False)[['coverage','width','pinball']].mean().groupby(['method','dataset','environment_id','episode_uuid'],as_index=False)[['coverage','width','pinball']].mean().groupby(['method','dataset','environment_id'],as_index=False)[['coverage','width','pinball']].mean();e.to_csv(out/'environments.csv',index=False);src=e.groupby(['method','dataset'],as_index=False)[['coverage','width','pinball']].mean();src.to_csv(out/'sources.csv',index=False);pd.DataFrame(support).to_csv(out/'support.csv',index=False)
 ref=s.set_index('method').loc['open'];dec=[]
 for method in ['pooled','env_mean','env_q75']:
  row=s.set_index('method').loc[method];ci=paired_ci(e,'open',method,metric='pinball');growth=100*(row.width/ref.width-1);minimum=src[src.method==method].coverage.min();passed=bool(.87<=row.coverage<=.93 and minimum>=.85 and ci['gain_pct']>=5 and growth<=25 and ci['lo95']>0);dec.append(dict(method=method,coverage=row.coverage,min_source_coverage=minimum,width_growth_pct=growth,admitted=passed,**ci))
 pd.DataFrame(dec).to_csv(out/'decisions.csv',index=False);(out/'audit.json').write_text(json.dumps(dict(train_only=True,held_global_env_excluded=True,development_reads=0,test_reads=0,formal_model_changed=False,candidates=4,promotion=False,notes='exploratory fixed grid, not independent confirmation'),indent=2),encoding='utf8');print(s.to_string(index=False));print(pd.DataFrame(dec)[['method','min_source_coverage','width_growth_pct','gain_pct','lo95','admitted']].to_string(index=False))
if __name__=='__main__':main()
