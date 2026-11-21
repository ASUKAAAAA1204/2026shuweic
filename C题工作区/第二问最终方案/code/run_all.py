"""Run the adopted solution and required descriptive evaluations in sequence."""
from pathlib import Path
import argparse,hashlib,json,subprocess,sys,time
BASE=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--data-root',type=Path,required=True);ap.add_argument('--windows',type=Path,required=True);a=ap.parse_args();data=a.data_root.resolve();windows=a.windows.resolve();start=time.time()
 models={p.name:sha(p) for p in (BASE/'model').glob('*.json')}
 stages=[('run_complete.py',['--data-root',str(data),'--windows',str(windows)]),('verify_outputs.py',[]),('run_required_baselines.py',['--data-root',str(data),'--windows',str(windows)]),('tolerance_horizons.py',[]),('audit_subgroups.py',[]),('natural_quality_analysis.py',['--data-root',str(data)])];finished=[]
 for script,args in stages:
  print('RUN',script,flush=True);subprocess.run([sys.executable,str(BASE/'code'/script),*args],check=True);finished.append(script)
 script='cost_error_frontier.py';subprocess.run([sys.executable,str(BASE/'code'/script),'--data-root',str(data),'--windows',str(windows)],check=True);finished.append(script)
 assert all(sha(BASE/'model'/k)==v for k,v in models.items())
 files={str(p.relative_to(BASE)).replace('\\','/'):sha(p) for p in sorted(BASE.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.name!='complete_delivery_manifest.json'}
 result=dict(passed=True,completed_stages=finished,frozen_model_unchanged=True,elapsed_s=time.time()-start,files=files,scope='adopted frozen solution and original-question evaluation; not confirmation that all research goals are solved',remaining=['phase-specific undercoverage','internal 3 percent predictor improvement unmet','adaptive advantage unproven','natural invalid-interval position truth unavailable','new independent evaluation unavailable','official missing observation conventions unavailable'])
 (BASE/'results/complete_delivery_manifest.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8');print('ALL',len(finished),'STAGES PASSED; MODEL UNCHANGED',flush=True)
if __name__=='__main__':main()
