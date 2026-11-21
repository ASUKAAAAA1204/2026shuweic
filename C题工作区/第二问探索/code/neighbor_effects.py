from pathlib import Path
import sys,json
import numpy as np,pandas as pd
B=Path(__file__).resolve().parents[1];sys.path.insert(0,str(B.parent/'第二问求解/code'));from update_calibration_experiment import paired_ci
p=B/'results/neighbor_effects';p.mkdir(exist_ok=True)
a=pd.read_csv(B/'results/neighbor_age/points.csv.gz');old=pd.read_csv(B/'results/distance_nested_calibration/outer_points.csv.gz');keys=['window_id','index','model','calibration'];z=a.merge(old,on=keys,suffixes=('_new','_old'),validate='one_to_one');assert len(z)==188148
base=['model','calibration','dataset_new','environment_id_new','episode_uuid_new','side_new','window_id'];metrics=['coverage','width','pinball']
for metric in metrics:z[metric+'_delta']=z[metric+'_new']-z[metric+'_old']
z['stage']=np.where(z.after_new,'after_return','before_return');z['age_band']=np.select([z.age_bin_new<=3,z.age_bin_new<=6,z.age_bin_new<=9],['<=.1','(.1,.2]','(.2,.3]'],default='>.3')
for factor in ['stage','age_band']:
 g=z.groupby(base+[factor],as_index=False)[[m+'_delta' for m in metrics]].mean()
 for lev in [['dataset_new','environment_id_new','episode_uuid_new','side_new'],['dataset_new','environment_id_new','episode_uuid_new'],['dataset_new','environment_id_new'],['dataset_new']]:g=g.groupby(['model','calibration',factor]+lev,as_index=False)[[m+'_delta' for m in metrics]].mean()
 g.to_csv(p/(factor+'_source_deltas.csv'),index=False)
# Candidate mixed rule is only diagnostic and registered after seeing whole-rule result.
# Prior-only neighbor avoids altering the known post-return rule; no current error is consulted.
mixed=a.copy();mask=mixed.after.to_numpy()
lookup=old.set_index(keys).loc[pd.MultiIndex.from_frame(mixed[keys])]
for metric in ['radius','coverage','width','pinball']:mixed.loc[mask,metric]=lookup[metric].to_numpy()[mask]
variants=[old.assign(variant='all_age'),a.assign(variant='neighbor'),mixed.assign(variant='prior_only_diagnostic')];summaries=[];intervals=[];envs=[]
for variant in variants:
 for scope,frame in [('all',variant),('after_return',variant[variant.after])]:
  g=frame.groupby(['variant','model','calibration','window_id','dataset','environment_id','episode_uuid','side'],as_index=False)[metrics].mean()
  for lev in [['dataset','environment_id','episode_uuid','side'],['dataset','environment_id','episode_uuid'],['dataset','environment_id']]:g=g.groupby(['variant','model','calibration']+lev,as_index=False)[metrics].mean()
  g['scope']=scope;envs.append(g)
e=pd.concat(envs,ignore_index=True);e.to_csv(p/'environments.csv',index=False)
s=e.groupby(['variant','model','calibration','scope','dataset'],as_index=False)[metrics].mean();s.to_csv(p/'sources.csv',index=False);s.groupby(['variant','model','calibration','scope'],as_index=False)[metrics].mean().to_csv(p/'summary.csv',index=False)
for (model,calibration,scope),frame in e.groupby(['model','calibration','scope']):
 if model!='distance_return' or calibration!='pooled':continue
 frame=frame.copy();frame['method']=frame.variant
 for method in ['neighbor','prior_only_diagnostic']:
  ci=paired_ci(frame,'all_age',method,metric='pinball');intervals.append(dict(model=model,calibration=calibration,scope=scope,**ci))
pd.DataFrame(intervals).to_csv(p/'paired_ci.csv',index=False)
assert np.allclose(mixed.loc[mask,'pinball'],lookup.pinball.to_numpy()[mask]);audit=dict(passed=True,paired_rows=len(z),mixed_rule='diagnostic after original experiment; not independently confirmed',post_return_mixed_exact_original=True,current_error_used_for_selection=False,development_reads=0,test_reads=0)
(p/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf8');print(s.query("model=='distance_return' and calibration=='pooled'").to_string(index=False));print(pd.DataFrame(intervals).to_string(index=False))
if __name__=='__main__':pass
