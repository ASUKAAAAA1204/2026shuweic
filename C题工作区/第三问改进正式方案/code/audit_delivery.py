from pathlib import Path
import json,hashlib
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]

def main():
    frozen=json.loads((ROOT/'selection_before_validation.json').read_text());original=json.loads((ROOT/'audit/original_selection_before_validation.json').read_text())
    assert hashlib.sha256((ROOT/'code/deadline_policy.py').read_bytes()).hexdigest()==original['hashes']['deadline_policy.py']
    for f,h in frozen['hashes'].items():assert hashlib.sha256((ROOT/'code'/f).read_bytes()).hexdigest()==h
    assert hashlib.sha256((ROOT/'model/training_family_samples.json').read_bytes()).hexdigest()==frozen['model_hash']
    d=pd.read_csv(ROOT/'results/validation_frozen/trials.csv');keys=['case_id','seed','alpha','beta','method'];assert len(d)==900 and not d.duplicated(keys).any()
    assert d.invalid_actions.eq(0).all()
    reference=pd.read_csv(ROOT/'audit/original_frozen_trials.csv');aa=d.set_index(keys).sort_index();bb=reference.set_index(keys).sort_index()
    assert aa.index.equals(bb.index);metrics=['weighted_completion_rate','weighted_on_time_rate','weighted_tardiness_s','cost','observations','elapsed_s','observed_overrun_s','reserved_idle_s','censored_units','invalid_actions']
    delta=float((aa[metrics]-bb[metrics]).abs().max().max());assert delta<1e-9
    nominal=d[(d.alpha==1)&(d.beta==.5)].groupby(['case_id','method']).weighted_on_time_rate.mean().unstack();gain=nominal.pressure_gated-nominal.density
    assert gain.mean()>=.03 and gain.min()>=-.01
    independent=json.loads((ROOT/'results/validation_frozen/independent_audit.json').read_text());assert independent['passed'] and independent['trials']==900 and independent['maximum_metric_error']<=1e-9
    for name in ['README.md','第三问完整解答.md','交付验收.md']:assert (ROOT/name).stat().st_size>100
    files=[p.name for p in (ROOT/'code').glob('*.py')];assert not set(files)&{'planner.py','rollout.py','scenario_planner.py','robust_policy.py','guarded_policy.py'}
    audit=dict(passed=True,matched_research_trials=900,maximum_repackaged_metric_difference=delta,nominal_gain=float(gain.mean()),per_case_gain=gain.to_dict(),adopted_policy_matches_original_freeze=True,failed_policy_implementations_excluded=True,official_final_test_evaluated=False)
    (ROOT/'audit/delivery_checks.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf8')
    manifest={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='manifest.json'}
    (ROOT/'audit/manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8');print(audit)
if __name__=='__main__':main()
