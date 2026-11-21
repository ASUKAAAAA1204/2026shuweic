from pathlib import Path
import json,math
import numpy as np,pandas as pd
BASE=Path(__file__).resolve().parents[1]
def rank(n,c):return math.ceil((n+1)*c-1e-12)
def main():
 out=BASE/'results/environment_block_calibration';out.mkdir(exist_ok=True)
 p=pd.read_csv(BASE/'results/initialization_split/labeled_residuals.csv.gz');p['phase']=np.where(p['mode']=='recovered_single','single_'+p.origin,p['mode'])
 sc=json.loads((BASE.parent/'第二问求解/results/S0/fold_scales.json').read_text());blocks=p.groupby(['dataset','phase','environment_id'],as_index=False).agg(max_raw_error=('raw_error','max'),points=('raw_error','size'),records=('episode_uuid','nunique'))
 blocks.to_csv(out/'environment_scores.csv',index=False);rows=[]
 for (src,phase),g in blocks.groupby(['dataset','phase']):
  for held in g.itertuples():
   fit=g[g.environment_id!=held.environment_id];assert held.environment_id not in set(fit.environment_id)
   for c in [.8,.9]:
    k=rank(len(fit),c);finite=k<=len(fit);r=float(np.sort(fit.max_raw_error)[k-1]) if finite else np.nan
    obs=p[(p.dataset==src)&(p.phase==phase)&(p.environment_id==held.environment_id)]
    rows.append(dict(dataset=src,phase=phase,held=held.environment_id,nominal=c,calibration_environments=len(fit),rank=k,finite=finite,radius_raw=r,width_normalized=2*r/sc[held.environment_id][src] if finite else np.nan,environment_all_covered=float(held.max_raw_error<=r) if finite else np.nan,point_coverage=float((obs.raw_error<=r).mean()) if finite else np.nan,held_points=len(obs),status='empirical_block_range' if finite else 'no_finite_range',guaranteed=False))
 a=pd.DataFrame(rows);a.to_csv(out/'oof_blocks.csv',index=False)
 summary=a.groupby(['dataset','phase','nominal'],as_index=False).agg(held_groups=('held','size'),calibration_environments=('calibration_environments','min'),finite_output_fraction=('finite','mean'),finite_environment_coverage=('environment_all_covered','mean'),finite_point_coverage=('point_coverage','mean'),finite_mean_width=('width_normalized','mean'));summary.to_csv(out/'summary.csv',index=False)
 audit=dict(passed=True,points=len(p),environment_blocks=len(blocks),evaluations=len(a),cross_source_fallback=False,fit_in_raw_units=True,unsupported_range_not_replaced_by_finite_max=True,formal_coverage_guarantee=False,development_reads=0,test_reads=0)
 (out/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf8');print(summary.query('nominal==.9').to_string(index=False));print(json.dumps(audit))
if __name__=='__main__':main()
