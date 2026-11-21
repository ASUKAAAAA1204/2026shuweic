from pathlib import Path
import sys,json
import numpy as np,pandas as pd
BASE=Path(__file__).resolve().parents[1];OLD=BASE.parent/'第二问求解';sys.path.insert(0,str(OLD/'code'));import s0_audit_replay as s0
from update_calibration_experiment import hierarchical

def main():
 inp=BASE/'results/distance_nested_calibration';out=BASE/'results/calibration_gap_diagnostic';out.mkdir(exist_ok=True)
 w=pd.read_csv(s0.Q1.parent/'C题工作区/阶段交付/stage1_review/corrected_results/data/windows.csv').query("split=='train'");e=pd.read_csv(inp/'outer_points.csv.gz');support=pd.read_csv(inp/'support.csv');labels=[];cache={}
 for r in w.itertuples():
  key=(r.episode_uuid,r.side)
  if key not in cache:
   with np.load(s0.DATA/'records'/f'{r.episode_uuid}.npz',allow_pickle=False) as z:cache[key]=z[r.side+'__t_s']
  t=cache[key][r.start_index:r.end_index_exclusive];q=[int(np.flatnonzero(t-t[0]+1e-12>=a)[0]) for a in [.3,.6]];arr=[int(np.flatnonzero(t+1e-12>=t[k]+.1)[0]) for k in q]
  for k in range(len(t)):labels.append(dict(window_id=r.window_id,index=k,return_phase=0 if k<arr[0] else 1 if k<arr[1] else 2,relative=float(t[k]-t[0])))
 lab=pd.DataFrame(labels);lab.to_csv(out/'point_phase_labels.csv.gz',index=False)
 a=e.merge(lab,on=['window_id','index'],validate='many_to_one');assert len(a)==len(e) and ((a.return_phase>0)==a.after).all()
 a=a.merge(support.rename(columns={'outer':'environment_id'}),on=['environment_id','model','dataset','after','age_bin'],validate='many_to_one');assert len(a)==len(e)
 for c in ['age_fallback','env_mean_fallback']:a[c]=a[c].astype(float)
 a['age_pool_status']=np.where(a.age_fallback>0,'pooled_all_ages','age_specific');a['age_band']=np.select([a.age_bin<=3,a.age_bin<=6,a.age_bin<=9],['<=.1','(.1,.2]','(.2,.3]'],default='>.3');a['excess_shortfall']=np.maximum(.9-a.coverage,0)
 metrics=['coverage','width','pinball','age_fallback','env_mean_fallback']
 for factor in ['return_phase','age_band','age_pool_status']:
  g=a.groupby(['model','calibration',factor,'window_id','dataset','environment_id','episode_uuid','side'],as_index=False)[metrics].mean();hierarchical(g,['model','calibration',factor],metrics).to_csv(out/(factor+'_summary.csv'),index=False)
  source=g.groupby(['model','calibration',factor,'dataset','environment_id','episode_uuid','side'],as_index=False)[metrics].mean().groupby(['model','calibration',factor,'dataset','environment_id','episode_uuid'],as_index=False)[metrics].mean().groupby(['model','calibration',factor,'dataset','environment_id'],as_index=False)[metrics].mean().groupby(['model','calibration',factor,'dataset'],as_index=False)[metrics].mean();source.to_csv(out/(factor+'_sources.csv'),index=False)
 a.groupby(['model','calibration','return_phase','age_bin','dataset','age_pool_status']).size().to_csv(out/'counts.csv');a.to_csv(out/'diagnostic_points.csv.gz',index=False)
 audit=dict(passed=True,outer_rows=len(a),base_timestamps=len(lab),phase_zero_matches_before_return=True,phase_is_known_from_arrivals=True,support_merge_complete=True,model_refitted=False,calibration_refitted=False,development_reads=0,test_reads=0)
 (out/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf8');print(json.dumps(audit))
 for factor in ['return_phase','age_band','age_pool_status']:print(factor);print(pd.read_csv(out/(factor+'_summary.csv')).query("model=='distance_return'").to_string(index=False))
if __name__=='__main__':main()
