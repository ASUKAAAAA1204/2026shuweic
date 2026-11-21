from pathlib import Path
import subprocess,sys
ROOT=Path(__file__).resolve().parent
for name in ['run_complete.py','verify.py','perturb_payload.py']:
 subprocess.run([sys.executable,str(ROOT/name)],check=True)
print('REPLAY AND INDEPENDENT VERIFICATION PASSED')
