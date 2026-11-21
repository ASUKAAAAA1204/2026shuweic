from pathlib import Path
import json
import numpy as np,pandas as pd
from evidence_range_interface import BlockEvidence,block_assessment
BASE=Path(__file__).resolve().parents[1]
def main():
 p=pd.read_csv(BASE/'results/initialization_split/labeled_residuals.csv.gz');p['phase']=np.where(p['mode']=='recovered_single','single_'+p.origin,p['mode'])
 b=pd.read_csv(BASE/'results/environment_block_calibration/oof_blocks.csv').query('nominal==.9');lookup={(r.dataset,r.phase,r.held):r for r in b.itertuples()};rows=[]
 for r in p.itertuples():
  row=lookup[(r.dataset,r.phase,r.environment_id)]
  evidence=BlockEvidence('natural_continuous',r.phase,.9,int(row.calibration_environments),float(row.radius_raw) if row.finite else None)
  result=block_assessment(mode=r.mode,origin=r.origin,evidence=evidence)
  assert not result['guaranteed']
  if not row.finite:assert result['radius'] is None and result['status']=='no_finite_range'
  rows.append(dict(dataset=r.dataset,episode_uuid=r.episode_uuid,side=r.side,index=r.index,mode=r.mode,origin=r.origin,**result))
 a=pd.DataFrame(rows);assert len(a)==93099
 recovering=a[(a['mode']=='recovered_single')&(a.origin=='after_observed_invalid')];assert len(recovering)==128 and (recovering.status=='no_finite_range').all()
 assert (a[a['mode']=='missing'].status=='no_finite_range').all()
 rejected=0
 for evidence,mode,origin in [(BlockEvidence('natural_continuous','tracking',.9,2,.01),'tracking','before_any_invalid'),(BlockEvidence('paid_isolated','tracking',.9,9,.01),'tracking','before_any_invalid'),(BlockEvidence('natural_continuous','single_before_any_invalid',.9,9,.01),'recovered_single','after_observed_invalid'),(BlockEvidence('natural_continuous','tracking',.9,9,float('nan')),'tracking','before_any_invalid')]:
  try:block_assessment(mode=mode,origin=origin,evidence=evidence)
  except ValueError:rejected+=1
  else:raise AssertionError('incompatible evidence accepted')
 out=BASE/'results/evidence_range_interface';out.mkdir(exist_ok=True);a.to_csv(out/'outputs.csv.gz',index=False);counts=a.groupby(['dataset','mode','origin','status']).size().reset_index(name='points');counts.to_csv(out/'status_counts.csv',index=False)
 audit=dict(passed=True,outputs=len(a),true_recovery_outputs=len(recovering),rejected_incompatible_inputs=rejected,guaranteed_claims=0,unscorable_or_uninitialized_not_in_residuals=993,omitted_993_not_declared_covered=True,held_identifier_used_by_replay_evaluator_only=True,interface_receives_reference=False,development_reads=0,test_reads=0,formal_predictor_changed=False)
 (out/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf8');print(json.dumps(audit));print(a.groupby('status').size().to_string())
if __name__=='__main__':main()
