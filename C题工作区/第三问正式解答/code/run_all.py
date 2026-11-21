"""Reproduce the adopted Q3 answer; rejected exploratory policies are not rerun."""
from pathlib import Path
import hashlib,json,platform,subprocess,sys
import numpy,pandas
BASE=Path(__file__).resolve().parents[1]
def main():
 stages=['audit_protocol.py','baseline_train.py','duration_information.py','validation.py','heldout_duration.py','verify_delivery.py','write_report.py']
 frozen=hashlib.sha256((BASE/'model/family_duration.json').read_bytes()).hexdigest()
 for name in stages:
  print('RUN',name,flush=True);subprocess.run([sys.executable,str(BASE/'code'/name)],check=True)
 assert hashlib.sha256((BASE/'model/family_duration.json').read_bytes()).hexdigest()==frozen
 files={str(p.relative_to(BASE)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(BASE.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.log' and p.name!='delivery_manifest.json'}
 result=dict(passed=True,stages=stages,model_unchanged=True,environment=dict(python=platform.python_version(),numpy=numpy.__version__,pandas=pandas.__version__),files=files,official_final_test_status='organizer additional seeds unavailable; not represented as completed',scope='complete official validation and report; no global optimality or physical performance claim')
 (BASE/'audit/delivery_manifest.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8');print('ALL STAGES PASSED',flush=True)
if __name__=='__main__':main()
