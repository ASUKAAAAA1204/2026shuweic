"""One-command reproducibility checks; green execution never implies quality gates pass."""
from pathlib import Path
import sys,subprocess,json,time,hashlib
import pandas as pd
BASE=Path(__file__).resolve().parents[1];ROOT=BASE.parent.parent

def main():
 latest='--latest' in sys.argv
 out=BASE/'results'/('latest_delivery_checks' if latest else 'delivery_checks');out.mkdir(exist_ok=True);formal=BASE.parent/'第二问求解/frozen_model.json';expected='baf3ce889dc51d62b331a8d30aeee34242109397b1e27a1562df391e908a1af3';before=hashlib.sha256(formal.read_bytes()).hexdigest();assert before==expected
 checks=['candidate_replay_integration.py','verify_paid_joint_output.py','verify_paid_range_boundaries.py','verify_evidence_range_interface.py','verify_independent_rule_selection.py','verify_hierarchy_weights.py','verify_post_return_decay.py','verify_causal_decay_fit.py','verify_online_decay_selector.py','verify_selector_calibration.py','verify_selector_pair_fits.py','verify_second_purchase_fit.py','verify_second_purchase_online.py','verify_purchase_boundaries.py','verify_purchase_calibration.py','verify_purchase_pair_fits.py'];runs=[]
 additions=['verify_validation_candidates.py','verify_validation_matched_ranges.py','verify_initial_history_candidates.py','verify_initial_speed_ranges.py','verify_initial_speed_validation.py','verify_return_speed_ranges.py','verify_frozen_speed_range.py']
 checks=additions if latest else checks+additions
 for script in checks:
  start=time.perf_counter();r=subprocess.run([sys.executable,'-B',str(BASE/'code'/script)],cwd=ROOT,capture_output=True,text=True,encoding='utf8',errors='replace');(out/(script+'.log')).write_text(r.stdout+'\n'+r.stderr,encoding='utf8');runs.append(dict(script=script,exit_code=r.returncode,elapsed_s=time.perf_counter()-start));print(script,r.returncode,flush=True)
 after=hashlib.sha256(formal.read_bytes()).hexdigest();assert before==after
 pred=json.loads((BASE/'results/distance_aligned_return/audit.json').read_text());s=pd.read_csv(BASE/'results/distance_nested_calibration/summary.csv');c=s.query("model=='distance_return' and calibration=='pooled' and scope=='all'").iloc[0];reference=s.query("model=='secant' and calibration=='pooled' and scope=='all'").iloc[0];src=pd.read_csv(BASE/'results/distance_nested_calibration/all_sources.csv').query("model=='distance_return' and calibration=='pooled'")
 selector=json.loads((BASE/'results/causal_decay_selector/audit.json').read_text());purchase=json.loads((BASE/'results/second_request_value/audit.json').read_text())
 gates=dict(prediction_registered_gate=bool(pred['admitted'] or selector['comparison']['gain_pct']>=3 and selector['comparison']['lo95']>0),calibration_engineering_numeric_gate=bool(.87<=c.coverage<=.93 and src.coverage.min()>=.85 and c.width<=reference.width and c.pinball<=reference.pinball),formal_reliability_established=False,adaptive_advantage_established=False,new_independent_generalization_established=False)
 validation=pd.read_csv(BASE/'results/initial_speed_validation/summary.csv').query("variant=='speed_conditioned' and method=='baseline' and scope=='all'").iloc[0]
 validation_sources=pd.read_csv(BASE/'results/initial_speed_validation/all_sources.csv').query("variant=='speed_conditioned' and method=='baseline'")
 validation_reference=pd.read_csv(BASE/'results/initial_speed_validation/summary.csv').query("variant=='source_age_control' and method=='baseline' and scope=='all'").iloc[0]
 gates['training_calibration_engineering_numeric_gate']=gates['calibration_engineering_numeric_gate']
 gates['validation_calibration_engineering_numeric_gate']=bool(.87<=validation.coverage<=.93 and validation_sources.coverage.min()>=.85 and validation.width<=validation_reference.width and validation.pinball<=validation_reference.pinball)
 gates['calibration_engineering_numeric_gate']=bool(gates['training_calibration_engineering_numeric_gate'] and gates['validation_calibration_engineering_numeric_gate'])
 audit=dict(execution_checks_passed=all(r['exit_code']==0 for r in runs),check_mode='latest_additions_only' if latest else 'full',checks=runs,formal_model_sha256=after,formal_model_unchanged=before==after,quality_gates=gates,objective_complete=False,research_candidates=dict(selector_gain_pct=selector['comparison']['gain_pct'],purchase_preference_weight=purchase['lambda_cost'],purchase_utility_gain_pct=purchase['comparison_utility']['gain_pct'],purchase_local_error_cost_disclosed=True),validation_conditional_baseline=dict(coverage=float(validation.coverage),minimum_source_coverage=float(validation_sources.coverage.min()),width=float(validation.width),pinball=float(validation.pinball)),scope='current code and saved training/previously-viewed validation evidence; no new test evaluation, no raw double-nested refit; latest mode does not rerun older checks');(out/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf8');print(json.dumps(audit));assert audit['execution_checks_passed']
if __name__=='__main__':main()
