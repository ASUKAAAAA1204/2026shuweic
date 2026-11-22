"""Run the adopted three-question solution against the frozen previous version."""
from pathlib import Path
import argparse
import subprocess
import sys
import pandas as pd
from trajectory import run as replay_positions
from scheduling import run as replay_scheduling
from subgroups import run as score_subgroups


def main():
    package=Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser()
    parser.add_argument("--processed",type=Path,default=package.parent/"处理结果/v1")
    parser.add_argument("--constraints",type=Path,default=package.parents[1]/"赛题分析报告")
    parser.add_argument("--output",type=Path,default=package/"results")
    parser.add_argument("--q3-only", action="store_true", help="Replay Q3 including non-nominal mappings, using frozen Q1/Q2 results.")
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    nominal = package.parent/"第三问分工况晋级"
    if not args.q3_only:
        replay_positions(args.processed,args.output)
        score_subgroups(args.processed,args.output)
    replay_scheduling(args.processed,args.constraints,args.output)
    if not args.q3_only:
        subprocess.run([sys.executable,"-B",str(nominal/"confirm.py"),"--output",
                        str(args.output/"q3/nominal_frozen"),"--constraints",str(args.constraints)],check=True)
        subprocess.run([sys.executable,"-B",str(nominal/"续轮优化/verify_release.py"),"--output",
                        str(args.output/"q3/nominal_promotions")],check=True)
    # The published nominal engine resolves its four input tables from the frozen package.
    fixed = package.parents[1]/"赛题分析报告"
    for name in ("common_conditions.json","source_segment_index.csv","tasks.csv","service_zones.csv"):
        if (args.constraints/name).read_text(encoding="utf-8-sig").splitlines() != (fixed/name).read_text(encoding="utf-8-sig").splitlines():
            raise ValueError(f"Official nominal replay requires frozen {name}")
    target = args.output/"q3/formal_release"
    subprocess.run([sys.executable,"-B",str(package/"code/nominal_replay.py"),"--output",str(target)],check=True)
    finish_release(package,args.output,fixed,package/"results" if args.q3_only else args.output)


def finish_release(package,output,conditions,frozen):
    target = output/"q3/formal_release"
    verification = subprocess.run([sys.executable,"-B",str(package/"code/verify_nominal_replay.py"),
                                  str(target/"trials.csv.gz"),"--conditions",str(conditions/"common_conditions.json")],
                                 check=True, capture_output=True, text=True)
    (target/"verification.json").write_text(verification.stdout,encoding="utf8")
    rows = []
    for question in ("q1","q2"):
        table = pd.read_csv(frozen/f"{question}_summary.csv").pivot(index="split",columns="model",values="nade")
        for split,row in table.iterrows():
            for baseline,old in [("previous_formal",row.B),("historical_damping",row.original)]:
                rows.append(dict(question=question,split=split,baseline=baseline,metric="NADE",
                                 old=old,new=row.B,gain_pct=100*(1-row.B/old)))
    scores = pd.read_csv(target/"comparison.csv")
    formal = pd.read_csv(target/"official_versions.csv")
    for row in scores.query("case_id=='overall'").itertuples():
        rows.append(dict(question="q3",split=row.split,baseline="tier3",metric=row.metric,
                         old=row.tier3,new=row.tier4,change_pp=100*row.delta if row.metric.endswith("_rate") else None))
    for row in formal.query("case_id=='overall'").itertuples():
        rows.append(dict(question="q3",split="validation_previous_release",baseline="previous_formal_tier2",
                         metric=row.metric,old=row.tier2,new=row.tier4,
                         change_pp=100*row.change_from_tier2 if row.metric.endswith("_rate") else None))
    q3 = output/"q3"
    old_results = q3
    trials = pd.read_csv(target/"trials.csv.gz").query("split=='official'").rename(columns={"version":"method"})
    tier2 = pd.read_csv(target/"official_tier2.csv").assign(method="tier2")
    official = pd.concat([trials,tier2],ignore_index=True)
    metrics = ["weighted_on_time_rate","weighted_completion_rate","weighted_tardiness_s","cost","observations"]
    cases = official.groupby(["case_id","method"],as_index=False)[metrics].mean().assign(alpha=1.,beta=.5)
    prior = pd.read_csv(old_results/"nominal_cases.csv")
    prior = prior[~prior.method.isin(["tier2","tier3","tier4"])]
    combined = pd.concat([prior,cases],ignore_index=True)
    combined.to_csv(q3/"nominal_cases.csv",index=False)
    totals = combined.groupby("method")[metrics].mean()
    totals.to_csv(q3/"nominal_overall.csv")
    previous_workspace = pd.read_csv(old_results/"workspace_previous_nominal.csv").weighted_on_time_rate.mean()
    for name,old in [("density_baseline",totals.loc["density",metrics[0]]),
                     ("pressure_gated",totals.loc["pressure_gated",metrics[0]]),
                     ("previous_workspace",previous_workspace)]:
        new = totals.loc["tier4",metrics[0]]
        rows.append(dict(question="q3",split="validation_"+name,baseline=name,metric=metrics[0],
                         old=old,new=new,change_pp=100*(new-old)))
    pd.DataFrame(rows,columns=["question","split","metric","old","new","gain_pct","change_pp","baseline"]).to_csv(
        output/"comparison.csv",index=False)


if __name__=="__main__":
    main()
