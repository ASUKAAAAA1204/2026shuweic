import json
from paid_joint_output import PaidRange
from request_value import BASE

def main():
 table={'EgoDemo|0|30':dict(radius=.2,points=100,environments=2,age_fallback=False)};unit=PaidRange(table);table['EgoDemo|0|30']['radius']=999
 assert unit.output([0,0,0],now=1.,last_sample_time=0.,returns=0,source='EgoDemo')['radius']==.2
 outside=unit.output([0,0,0],now=1.001,last_sample_time=0.,returns=0,source='EgoDemo');assert outside['radius'] is None and outside['status']=='outside_calibration_domain'
 assert unit.output([0,0,0],now=1.,last_sample_time=0.,returns=1,source='EgoDemo')['radius'] is None
 rejected=0
 for patch in [dict(radius=-1),dict(radius=float('nan')),dict(radius=float('inf')),dict(points=0),dict(environments=101),dict(environments=0),dict(age_fallback='False')]:
  value=dict(radius=.2,points=100,environments=2,age_fallback=False);value.update(patch)
  try:PaidRange({'EgoDemo|0|30':value})
  except ValueError:rejected+=1
  else:raise AssertionError(patch)
 try:unit.table['EgoDemo|0|30']['radius']=3
 except TypeError:pass
 else:raise AssertionError('mutable evidence')
 audit=dict(passed=True,invalid_tables_rejected=rejected,external_mutation_isolated=True,internal_evidence_immutable=True,age_one_second_accepted=True,age_over_one_second_abstained=True,return_state_mismatch_abstained=True)
 (BASE/'results/paid_joint_output/boundary_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf8');print(json.dumps(audit))
if __name__=='__main__':main()
