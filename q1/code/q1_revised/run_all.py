"""Reproduce the revised Q1 experiment in order, stopping on any failed check."""
import os
import json
import platform
from importlib.metadata import version
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
def main():
    env=os.environ.copy()
    for key in ("OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","OMP_NUM_THREADS","NUMEXPR_NUM_THREADS"):
        env[key]="1"
    stages=[("audit_position.py",),("quality_replay.py",),("source_check.py",),
            ("position_fit.py",),("check_curve_dev.py",),("score_position.py",),
            ("recovery_fit.py",),("natural_reference.py",),("score_recovery.py",),
            ("calibration.py","continuous"),("calibration.py","natural"),("verify_delivery.py",)]
    for name,*args in stages:
        subprocess.run([sys.executable,"-B",str(HERE/name),*args],cwd=ROOT,env=env,check=True)
    runtime=dict(python=sys.version,executable=Path(sys.executable).name,os=platform.platform(),
                 processor=platform.processor(),packages={k:version(k) for k in
                 ("numpy","pandas","scipy","scikit-learn","threadpoolctl")})
    (ROOT/"results/q1_revised/runtime.json").write_text(json.dumps(runtime,indent=2),encoding="utf8")
if __name__=="__main__":main()
