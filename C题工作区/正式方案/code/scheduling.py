"""Replay the frozen scheduling package without writing into its reference outputs."""
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
import pandas as pd


def run(processed, config, output):
    package = Path(__file__).resolve().parents[2]/"第三问改进正式方案"
    target = output/"q3"
    target.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory() as tmp:
        work = Path(tmp)
        task = work/package.name
        shutil.copytree(package,task,ignore=shutil.ignore_patterns("results","__pycache__"))
        cfg = work/"Restocking_Problem_and_Data_EN/Restocking_Tasks_and_Constraints"
        cfg.mkdir(parents=True)
        for name in ["tasks.csv","service_zones.csv","source_segment_index.csv","common_conditions.json"]:
            shutil.copyfile(config/name,cfg/name)
        data = work/"处理结果/v1"
        data.mkdir(parents=True)
        shutil.copyfile(processed/"segments_private.csv",data/"segments_private.csv")
        subprocess.run([sys.executable,"-B",str(task/"code/run_all.py")],check=True)
        shutil.copytree(task/"results/validation_frozen",target,dirs_exist_ok=True)
        shutil.copytree(task/"results/duration_uncertainty",target/"duration_uncertainty",dirs_exist_ok=True)
        shutil.copyfile(task/"audit/delivery_checks.json",target/"delivery_checks.json")
        previous = package.parent/"第三问正式解答"
        legacy = work/previous.name
        shutil.copytree(previous/"code",legacy/"code")
        subprocess.run([sys.executable,"-B",str(legacy/"code/validation.py")],check=True)
        legacy_trials=pd.read_csv(legacy/"results/validation/trials.csv")
        matched=pd.read_csv(previous/"results/validation/trials.csv")
        index=["case_id","seed","alpha","beta","method"]
        metrics=["weighted_on_time_rate","weighted_completion_rate","weighted_tardiness_s","cost","observations"]
        left,right=[t.set_index(index).sort_index() for t in (legacy_trials,matched)]
        assert len(left)==1800 and left.index.equals(right.index)
        assert (left[metrics]-right[metrics]).abs().max().max()<1e-9
        old=legacy_trials.query("alpha==1 and beta==0.5 and method=='density_median'")
        old.groupby("case_id")[metrics].mean().to_csv(target/"workspace_previous_nominal.csv")
    scores = pd.read_csv(target/"summary.csv")
    nominal = scores.query("alpha==1 and beta==0.5")
    nominal.to_csv(target/"nominal_cases.csv",index=False)
    nominal.groupby("method")[["weighted_on_time_rate","weighted_completion_rate","weighted_tardiness_s","cost","observations"]].mean().to_csv(target/"nominal_overall.csv")
