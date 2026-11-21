from pathlib import Path
import json
import numpy as np,pandas as pd
BASE=Path(__file__).resolve().parents[1];p=BASE/'results/stage_diagnostic';lab=pd.read_csv(p/'labels.csv');seg=pd.read_csv(BASE.parents[1]/'分支测试/q1-ding/q1/C_Problem_Data/v1/v1/segments_private.csv').query("split=='train'")
rounding_boundary_count=0
for r in lab.itertuples():
 a=seg[seg.episode_uuid==r.episode_uuid].sort_values('start_s');idx=np.searchsorted(a.start_s.to_numpy(),r.start_s,side='right')-1;assert idx>=0;s=a.iloc[idx]
 assert s.start_s<=r.start_s<s.end_s and s.family_id==r.family and int(s.source_segment_index)==r.source_segment_index
 assert r.boundary==('crosses_end' if r.last_score_s>=s.end_s else 'internal')
 q=(r.start_s-s.start_s)/(s.end_s-s.start_s);expected=['early','middle','late'][min(2,int(3*q))]
 if r.progress_third!=expected:
  assert min(abs(q-1/3),abs(q-2/3))<1e-12
  rounding_boundary_count+=1
old=pd.read_csv(BASE.parent/'第一问重构/实验比较_v3/full_semantic_coverage/window_mapping.csv');old=old.rename(columns={'id':'window_id','family':'prior_family'});cross=lab.merge(old[['window_id','prior_family']],on='window_id',validate='one_to_one');assert len(cross)>0 and (cross.family==cross.prior_family).all()
m=pd.read_csv(BASE.parent/'第二问求解/results/S3/update_windows.csv').query("method=='secant'").merge(lab[['window_id','boundary']],on='window_id',validate='one_to_one')
g=m.groupby(['boundary','dataset','environment_id','episode_uuid','side'],as_index=False).nade.mean().groupby(['boundary','dataset','environment_id','episode_uuid'],as_index=False).nade.mean().groupby(['boundary','dataset','environment_id'],as_index=False).nade.mean()
pair=g.pivot(index=['dataset','environment_id'],columns='boundary',values='nade').dropna().reset_index();pair['difference']=pair.crosses_end-pair.internal;pair.to_csv(p/'matched_environment_boundary.csv',index=False)
source=pair.groupby('dataset')[['crosses_end','internal','difference']].mean();source.to_csv(p/'matched_source_boundary.csv');overall=source.mean();overall.to_csv(p/'matched_boundary_summary.csv')
audit=dict(passed=True,independently_mapped=len(lab),progress_float_boundary_cases=rounding_boundary_count,prior_mapping_family_matches=len(cross),short_semantic_windows=int((lab.third_question_eligible==0).sum()),boundary_windows=int((lab.boundary=='crosses_end').sum()),matched_source_environment_pairs=len(pair),unique_matched_global_environments=pair.environment_id.nunique(),offline_future_end_used_only_for_diagnosis=True,phase_controls_confounding=False)
(p/'independent_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf8');print(json.dumps(audit));print(source.to_string());print(overall.to_string())

