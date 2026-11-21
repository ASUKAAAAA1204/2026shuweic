"""Read-only completion and manifest verification. Never refits a model."""
from pathlib import Path
import json,hashlib,csv,sys
BASE=Path(__file__).resolve().parents[1]
def main():
 checks=[]
 def require(name,ok):
  checks.append(dict(name=name,passed=bool(ok)))
  assert ok,name
 def read(path):return json.loads((BASE/path).read_text(encoding='utf-8-sig'))
 for p in (BASE/'code').glob('*.py'):compile(p.read_bytes(),str(p),'exec')
 require('all source files compile',True)
 f=read('frozen_model.json');require('Q1 retained',f['q1_commit']=='bf380c4' and f['selected']=='fixed')
 for name,digest in f['code_sha256'].items():require('frozen inference/source hash '+name,hashlib.sha256((BASE/'code'/name).read_bytes()).hexdigest()==digest)
 require('protocol hash',hashlib.sha256((BASE/'protocol.json').read_bytes()).hexdigest()==f['protocol_sha256'])
 a=read('results/S0/audit.json');require('S0 input/window audit',a['record_index']==69 and a['qualified_windows_checked']==1739 and a['test_trajectory_reads']==0)
 a=read('results/S1/audit.json');require('S1 causal engine checks',a['passed'] and a['checks']>=16 and a['future_perturbation_tests']>=200 and a['real_training_replays']==27486 and a['max_metric_difference']<=1e-9)
 a=read('results/S1/adaptive_policy_checks.json');require('all adaptive policies noninterfering',len(a)==3 and all(v['passed'] and v['perturbation_tests']==50 for v in a))
 a=read('results/S2/audit.json');require('S2 full train conditions',a['rows']==27486 and a['radius_fit_excludes_held_global_env'] and a['test_trajectory_reads']==0)
 a=read('results/S3/independent_audit.json');require('S3 independent quantiles and nesting',a['passed'] and a['outer_folds']==16 and a['quantiles_independently_recomputed']==1950 and a['max_quantile_difference']<1e-9 and a['inner_results']==229050)
 a=read('results/S4_fixed/selection.json');require('S4 fixed candidates',a['parameter_combinations']==6 and a['nested_outer_envs']==16)
 a=read('results/S4_adaptive/selection.json');require('S4 selection before development',not a['development_used'] and a['inner_calibration_excludes_outer_and_inner'] and a['test_trajectory_reads']==0)
 for split,count in [('validation',9540),('test',17550)]:
  a=read(f'results/S5_{split}/independent_audit.json');require('S5 '+split+' independent audit',a['passed'] and a['window_results']==count and a['max_independent_score_error']<=1e-9 and a['budget_violations']==0 and a['early_returns']==0 and a['frozen_code_hashes_valid'])
  a=read(f'results/S5_{split}/audit.json');require('S5 '+split+' unchanged frozen model',not a['model_refitted'] and a['frozen_model_sha256']==hashlib.sha256((BASE/'frozen_model.json').read_bytes()).hexdigest())
 a=read('results/diagnostics/audit.json');require('stage and history evaluation only',a['evaluation_only'] and a['source_endpoints_never_policy_inputs'] and not a['model_refitted'])
 needed=['第二问完整解答与实验报告.md','执行计划与量化目标.md','README.md','paired_final_comparisons.csv','runtime_summary.csv','empirical_prediction_duration.csv','results/S0/natural_reason_summary.csv','results/S0/natural_duration_summary.csv','results/S5_test/worst_fixed_windows.csv','results/S5_test/points.csv.gz','results/S5_test/actions.csv.gz','results/diagnostics/phase_task_history_summary.csv']
 for name in needed:require('required deliverable '+name,(BASE/name).is_file() and (BASE/name).stat().st_size>0)
 manifest=BASE/'manifest.csv'
 if '--build-manifest' in sys.argv:
  with manifest.open('w',encoding='utf8',newline='') as file:
   w=csv.DictWriter(file,fieldnames=['path','bytes','sha256']);w.writeheader()
   for p in sorted(BASE.rglob('*')):
    if p.is_file() and p.name not in ['manifest.csv','交付验收.json'] and '__pycache__' not in p.parts:w.writerow(dict(path=p.relative_to(BASE).as_posix(),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
 with manifest.open(encoding='utf-8-sig') as file:entries=list(csv.DictReader(file))
 for r in entries:
  p=BASE/r['path'];require('manifest '+r['path'],p.is_file() and p.stat().st_size==int(r['bytes']) and hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256'])
 final=dict(engineering_delivery_passed=True,check_count=len(checks),manifest_files=len(entries),manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),performance_targets=dict(adaptive_ade_gain_3pct=False,cost_saving_15pct_at_3pct_ade_degradation=False,nominal_90pct_coverage_guarantee=False,tolerance_5pct_reliable_operation=False),selection='fixed damping tau .2 + secant update + requests at .3/.6',final_test_used=True,test_used_for_refitting=False,claim='finite candidate study delivered; performance improvement targets not achieved; no global optimality claim',checks=checks)
 (BASE/'交付验收.json').write_text(json.dumps(final,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps({k:v for k,v in final.items() if k!='checks'},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
