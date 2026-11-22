"""Replay the committed scheduler and pair it with a fresh formal nominal replay."""
import argparse
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
import zipfile

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT
REV = "c6a63a50767fbec6e0969e52880d84e59dab945c"
OLD = Path("C题工作区/第三问改进正式方案")
CONFIG = Path("赛题分析报告")
METRICS = ["weighted_on_time_rate", "weighted_completion_rate", "weighted_tardiness_s", "cost"]


def git(*args):
    return subprocess.check_output(["git", "-C", str(REPO), *args])


def archived_csv(name):
    return pd.read_csv(io.BytesIO(git("show", f"{REV}:C题工作区/正式方案/results/{name}")))


def replay_commit(work):
    paths = [OLD/"code", OLD/"model", OLD/"selection_before_validation.json"]
    inputs = ["common_conditions.json", "source_segment_index.csv", "tasks.csv", "service_zones.csv"]
    data = git("archive", "--format=zip", REV, *(p.as_posix() for p in paths),
               *((CONFIG/name).as_posix() for name in inputs))
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        archive.extractall(work)
    cfg = work/"C题工作区/Restocking_Problem_and_Data_EN/Restocking_Tasks_and_Constraints"
    cfg.mkdir(parents=True)
    for name in inputs:
        source = work/CONFIG/name
        if source.read_text(encoding="utf-8-sig") != (REPO/CONFIG/name).read_text(encoding="utf-8-sig"):
            raise ValueError(f"Frozen inputs differ: {name}")
        shutil.copyfile(source, cfg/name)
    private = Path("C题工作区/处理结果/v1/segments_private.csv")
    fields = ["family_id","unit_id","duration_s"]
    def pool(p):
        return pd.read_csv(p).query("split=='validation' and eligible==1")[fields].reset_index(drop=True)
    pd.testing.assert_frame_equal(pool(REPO/private), pool(cfg/"source_segment_index.csv"), check_exact=True)
    (work/private).parent.mkdir(parents=True)
    shutil.copyfile(REPO/private, work/private)
    solver = work/OLD
    subprocess.run([sys.executable, "-B", "-X", "utf8", str(solver/"code/run.py")], check=True, cwd=solver)
    return pd.read_csv(solver/"results/validation_frozen/trials.csv")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path, required=True, help="Output of run_current_solution.py --q3-only")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    current_dir = args.replay/"q3/formal_release"
    if not json.loads((current_dir/"verification.json").read_bytes())["passed"]:
        raise ValueError("Current replay did not pass verification")
    new = pd.read_csv(current_dir/"trials.csv.gz").query("split=='official' and version=='tier4'")
    with TemporaryDirectory() as work:
        old = replay_commit(Path(work))
    key = ["case_id","seed"]
    nominal = "alpha==1 and beta==0.5 and method=='pressure_gated'"
    def indexed(frame):
        result = frame.set_index(key)[METRICS].sort_index()
        if not result.index.is_unique:
            raise ValueError("Duplicate case and seed")
        return result
    old = indexed(old.query(nominal))
    pd.testing.assert_frame_equal(old, indexed(archived_csv("q3/trials.csv").query(nominal)), atol=1e-9,rtol=0)
    new = indexed(new)
    if len(old) != 90 or not old.index.equals(new.index):
        raise ValueError("Expected 90 matched case/seed pairs")
    reference = indexed(pd.read_csv(REPO/"C题工作区/正式方案/results/q3/formal_release/trials.csv.gz")
                        .query("split=='official' and version=='tier4'"))
    pd.testing.assert_frame_equal(new, reference, atol=1e-9,rtol=0)
    rows = []
    for case in ["S1","S2","S3","overall"]:
        a,b = (df.groupby("seed").mean() if case=="overall" else df.loc[case] for df in (old,new))
        for m in METRICS:
            rows.append(dict(case=case,metric=m,repo=a[m].mean(),current=b[m].mean(),delta=(b[m]-a[m]).mean()))
    args.output.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output/"official_30.csv",index=False)
    old.add_prefix("repo_").join(new.add_prefix("current_")).to_csv(args.output/"paired_90.csv")
    frozen = REPO/"C题工作区/正式方案/results"
    for q in ["q1","q2"]:
        name = f"{q}_summary.csv"
        pd.testing.assert_frame_equal(archived_csv(name), pd.read_csv(frozen/name),check_exact=True)
    print(pd.DataFrame(rows).query("case=='overall'").to_string(index=False))


if __name__ == "__main__":
    main()
