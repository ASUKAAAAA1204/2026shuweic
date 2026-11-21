import json
import numpy as np,pandas as pd
from request_value import BASE,OLD,weights
from update_calibration_experiment import hierarchical

def main():
 out=BASE/'results/hierarchy_weighted_calibration';out.mkdir(exist_ok=True)
 cal=pd.read_csv(BASE/'results/distance_nested_calibration/inner_residuals.csv.gz').query("model=='distance_return'");ev=pd.read_csv(BASE/'results/distance_nested_calibration/outer_points.csv.gz').query("model=='distance_return' and calibration=='pooled'");sc=json.loads((OLD/'results/S0/fold_scales.json').read_text());rows=[];support=[]
 for outer,g in ev.groupby('environment_id'):
  c=cal[cal.outer==outer].copy();assert outer not in set(c.environment_id)
  meta=c[['window_id','dataset','environment_id','episode_uuid','side']].drop_duplicates();meta['weight']=weights(meta);wm=meta.set_index('window_id').weight
  c['weight']=c.window_id.map(wm)/c.groupby('window_id').window_id.transform('size')
  assert abs(c.weight.sum()-1)<1e-12
  for (src,after,age),z in g.groupby(['dataset','after','age_bin']):
   pool=c[(c.dataset==src)&(c.after==after)];local=pool[pool.age_bin==age];fallback=len(local)<100
   if fallback:local=pool
   ordered=local.sort_values('error',kind='stable');point_r=float(ordered.error.iloc[int(np.ceil(.9*len(ordered)))-1]);control=z.copy();delta0=control.error-point_r;control['radius']=point_r;control['coverage']=(delta0<=0).astype(float);control['width']=2*point_r/sc[outer][src];control['pinball']=np.maximum(.9*delta0,-.1*delta0)/sc[outer][src];control['method']='point_cdf';rows.append(control)
   cw=np.cumsum(ordered.weight.to_numpy());idx=int(np.searchsorted(cw,.9*cw[-1],side='left'));r=float(ordered.error.iloc[idx]);q=z.copy();delta=q.error-r;q['radius']=r;q['coverage']=(delta<=0).astype(float);q['width']=2*r/sc[outer][src];q['pinball']=np.maximum(.9*delta,-.1*delta)/sc[outer][src];q['method']='hierarchy_weighted';rows.append(q)
   support.append(dict(outer=outer,dataset=src,after=after,age_bin=age,points=len(local),environments=local.environment_id.nunique(),fallback=fallback,radius=r,total_weight=float(cw[-1]),cdf_below=float(cw[idx-1]/cw[-1]) if idx else 0.,cdf_at=float(cw[idx]/cw[-1])))
 result=pd.concat([ev.assign(method='pooled'),*rows]);result.to_csv(out/'points.csv.gz',index=False);pd.DataFrame(support).to_csv(out/'support.csv',index=False)
 summaries=[]
 for scope,f in [('all',result),('after_return',result[result.after])]:
  w=f.groupby(['method','window_id','dataset','environment_id','episode_uuid','side'],as_index=False)[['coverage','width','pinball']].mean();s=hierarchical(w,['method'],['coverage','width','pinball']);s['scope']=scope;summaries.append(s)
  e=w.groupby(['method','dataset','environment_id','episode_uuid','side'],as_index=False)[['coverage','width','pinball']].mean().groupby(['method','dataset','environment_id','episode_uuid'],as_index=False)[['coverage','width','pinball']].mean().groupby(['method','dataset','environment_id'],as_index=False)[['coverage','width','pinball']].mean();e.to_csv(out/(scope+'_environments.csv'),index=False);e.groupby(['method','dataset'],as_index=False)[['coverage','width','pinball']].mean().to_csv(out/(scope+'_sources.csv'),index=False)
 summary=pd.concat(summaries);summary.to_csv(out/'summary.csv',index=False);print(summary.to_string(index=False))
 assert all(v['cdf_below']<.9+1e-12 and v['cdf_at']>=.9-1e-12 for v in support)
 (out/'audit.json').write_text(json.dumps(dict(outer_folds=ev.environment_id.nunique(),groups=len(support),cdf_rank_checks=True,development_reads=0,test_reads=0,formal_model_changed=False,coverage_guarantee=False),indent=2),encoding='utf8')
if __name__=='__main__':main()
