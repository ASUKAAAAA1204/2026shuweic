"""Run the adopted trajectory, re-observation and B_K64 scheduling policies."""
import argparse
from pathlib import Path

import pandas as pd

from trajectory import run as replay_positions
from subgroups import run as score_subgroups
from scheduling import run as replay_historical_scheduling
from b_replay import METHOD, METRICS, run as replay_b


def merge_methods(previous_path, current, destination):
    previous = pd.read_csv(previous_path)
    merged = pd.concat([previous[previous.method != METHOD], current], ignore_index=True)
    merged.to_csv(destination, index=False)
    return merged


def main():
    package = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed", type=Path, default=package.parent / "处理结果/v1")
    parser.add_argument("--constraints", type=Path, default=package.parents[1] / "赛题分析报告")
    parser.add_argument("--output", type=Path, default=package / "results")
    parser.add_argument("--q3-only", action="store_true", help="Replay B_K64 and use frozen Q1/Q2 summaries.")
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    frozen = package / "results"
    historical = pd.read_csv(frozen / "q3/nominal_cases.csv")
    cfg = package.parents[1] / "赛题分析报告"
    for name in ("common_conditions.json","source_segment_index.csv","tasks.csv","service_zones.csv"):
        if (args.constraints/name).read_text(encoding="utf-8-sig") != (cfg/name).read_text(encoding="utf-8-sig"):
            raise ValueError(f"Official replay requires frozen {name}")
    if not args.q3_only:
        replay_positions(args.processed, out)
        score_subgroups(args.processed, out)
        # Historical schedulers also produce the task-duration diagnostics.
        # Keep their output distinct from the adopted policy.
        replay_historical_scheduling(args.processed, args.constraints, out)

    trials, paired = replay_b(args.constraints, out)
    q3 = out / "q3"
    reference = (frozen if args.q3_only else out) / "q3"
    common = trials.assign(seconds=trials.decision_total_ms / 1000)
    merge_methods(reference / "trials.csv", common, q3 / "trials.csv")
    by_case = common.groupby(["case_id","alpha","beta","method"], as_index=False)[METRICS+["seconds"]].mean()
    merge_methods(reference / "summary.csv", by_case, q3 / "summary.csv")
    nominal = by_case.query("alpha==1 and beta==0.5")
    # Keep the frozen tier comparators, including when a full run regenerates the old baseline table.
    core = METRICS[:5]
    cases = pd.concat([historical[historical.method != METHOD],nominal],ignore_index=True)
    cases.to_csv(q3 / "nominal_cases.csv",index=False)
    cases.groupby("method")[core].mean().to_csv(q3 / "nominal_overall.csv")
    rows = []
    for question in ("q1","q2"):
        src = frozen if args.q3_only else out
        table = pd.read_csv(src / f"{question}_summary.csv").pivot(index="split", columns="model", values="nade")
        for split, values in table.iterrows():
            for baseline, old in [("previous_formal",values.B),("historical_damping",values.original)]:
                rows.append(dict(question=question, split=split, baseline=baseline, method="B",
                                 metric="NADE", old=old, new=values.B, gain_pct=100*(1-values.B/old)))
    for row in paired.query("case_id=='overall'").itertuples():
        rows.append(dict(question="q3", split="validation_"+row.scope, baseline="previous_local",
                         method=METHOD, metric=row.metric, old=row.previous_local,new=row.B_K64,
                         change_pp=100*row.delta if row.metric.endswith("_rate") or row.metric=="valid_protocol" else None))
    pd.DataFrame(rows).to_csv(out / "comparison.csv",index=False)


if __name__ == "__main__":
    main()
