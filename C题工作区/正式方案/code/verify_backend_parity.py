"""Compare two complete Q3 runs by trial key, including nominal event histories."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


NOMINAL_KEY = ["split", "case_id", "seed", "version"]
MAPPING_KEY = ["case_id", "seed", "alpha", "beta", "method"]


def paired(left, right, filename, keys, n, *, ignore=(), exact=False):
    def load(directory):
        frame = pd.read_csv(directory/filename).drop(columns=list(ignore)).set_index(keys).sort_index()
        return frame
    a, b = load(left), load(right)
    if len(a) != n or len(b) != n or not a.index.is_unique or not b.index.is_unique:
        raise ValueError(f"Expected {n} unique trial keys in {filename}")
    pd.testing.assert_index_equal(a.index, b.index)
    pd.testing.assert_index_equal(a.columns, b.columns)
    pd.testing.assert_frame_equal(a, b, check_exact=exact, atol=1e-9, rtol=0)
    numeric = list(a.select_dtypes(include="number").columns)
    x, y = (v[numeric].to_numpy(dtype=float) for v in (a,b))
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError(f"Nonfinite values in {filename}")
    return a, float(np.max(np.abs(x-y)))


def compare(reference, run):
    if reference.resolve() == run.resolve():
        raise ValueError("Use a fresh --output directory distinct from the reference run")
    left, right = reference/"q3", run/"q3"
    sub = Path("formal_release")
    trials, nominal_diff = paired(left,right,sub/"trials.csv.gz",NOMINAL_KEY,12468)
    traces, _ = paired(left,right,sub/"trace_digests.csv.gz",NOMINAL_KEY,12468,exact=True)
    pd.testing.assert_index_equal(traces.index,trials.index)
    if not traces.events_sha256.str.fullmatch(r"[0-9a-f]{64}").all() or not (traces.event_count>0).all():
        raise ValueError("Missing nominal event hashes")
    for case in ("S1","S2"):
        frozen = traces.xs(case,level="case_id")
        pd.testing.assert_frame_equal(frozen.xs("tier3",level="version"),frozen.xs("tier4",level="version"),check_exact=True)
    _, tier2_diff = paired(left,right,sub/"official_tier2.csv",["case_id","seed"],90)
    mapping, mapping_diff = paired(left,right,Path("trials.csv"),MAPPING_KEY,900,ignore=("seconds",))
    for folder in (left,right):
        verified = json.loads((folder/sub/"verification.json").read_bytes())
        mapping_audit = json.loads((folder/"independent_audit.json").read_bytes())
        if not verified["passed"] or verified["invalid_actions"] or not mapping_audit["passed"]:
            raise ValueError(f"Failed replay audit in {folder}")
    group = mapping.reset_index()
    split = group.query("alpha!=1 or beta!=0.5")
    if len(split)!=720 or len(split[["alpha","beta"]].drop_duplicates())!=4:
        raise ValueError("Incomplete non-nominal replay")
    return dict(passed=True,reference_backend=json.loads((left/sub/"backend.json").read_bytes())["backend"],
                backend=json.loads((right/sub/"backend.json").read_bytes())["backend"],
                official_seeds=30,promotion_seeds=2048,paired_nominal_trials=len(trials),
                paired_nominal_events=int(traces.event_count.sum()),frozen_cases=["S1","S2"],
                official_tier2_trials=90,mapping_trials=len(mapping),non_nominal_trials=len(split),
                invalid_actions=int(trials.invalid_actions.sum()+mapping.invalid_actions.sum()),
                max_abs_delta=dict(nominal=nominal_diff,tier2=tier2_diff,mappings=mapping_diff))


if __name__=="__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run",type=Path)
    p.add_argument("--reference",type=Path,default=Path(__file__).resolve().parents[1]/"results")
    p.add_argument("--output",type=Path)
    args = p.parse_args()
    report = json.dumps(compare(args.reference,args.run),indent=2)
    print(report)
    if args.output:
        args.output.write_text(report+"\n",encoding="utf8")
