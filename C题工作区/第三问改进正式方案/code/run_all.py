"""Reproduce adopted policy and independently audit all replay metrics."""
from pathlib import Path
import subprocess,sys
ROOT=Path(__file__).resolve().parent
for name in ['run.py','verify.py','duration_audit.py','audit_delivery.py']:
    subprocess.run([sys.executable,str(ROOT/name)],check=True,cwd=ROOT.parent)
print('ADOPTED DELIVERY REPRODUCED AND AUDITED')
