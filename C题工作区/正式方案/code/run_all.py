"""Run the adopted three-question solution against the frozen previous version."""
from pathlib import Path
import argparse
import pandas as pd
from trajectory import run as replay_positions
from scheduling import run as replay_scheduling
from subgroups import run as score_subgroups


def main():
    package=Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser()
    parser.add_argument("--processed",type=Path,default=package.parent/"处理结果/v1")
    parser.add_argument("--constraints",type=Path,default=package.parent/"Restocking_Problem_and_Data_EN/Restocking_Tasks_and_Constraints")
    parser.add_argument("--output",type=Path,default=package/"results")
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    replay_positions(args.processed,args.output)
    score_subgroups(args.processed,args.output)
    replay_scheduling(args.processed,args.constraints,args.output)
    rows=[]
    for question in ("q1","q2"):
        table=pd.read_csv(args.output/f"{question}_summary.csv").pivot(index="split",columns="model",values="nade")
        for split,row in table.iterrows():
            rows.append(dict(question=question,split=split,metric="NADE",old=row.original,new=row.B,
                             gain_pct=100*(1-row.B/row.original),change_pp=None))
    q3=pd.read_csv(args.output/"q3/nominal_overall.csv").set_index("method")
    new=q3.loc["pressure_gated","weighted_on_time_rate"]
    density=q3.loc["density","weighted_on_time_rate"]
    # The immediately preceding release already used the same pressure-gated policy.
    # The scheduling package independently checks every trial against its freeze.
    archive=pd.read_csv(package.parent/"第三问改进正式方案/audit/original_frozen_trials.csv")
    previous=archive.query("alpha==1 and beta==0.5 and method=='pressure_gated'").groupby("case_id").weighted_on_time_rate.mean().mean()
    rows.append(dict(question="q3",split="validation_previous_release",metric="weighted_on_time_rate",
                     old=previous,new=new,gain_pct=None,change_pp=100*(new-previous)))
    workspace_previous=pd.read_csv(args.output/"q3/workspace_previous_nominal.csv").weighted_on_time_rate.mean()
    rows.append(dict(question="q3",split="validation_previous_workspace",metric="weighted_on_time_rate",
                     old=workspace_previous,new=new,gain_pct=None,change_pp=100*(new-workspace_previous)))
    rows.append(dict(question="q3",split="validation_density_baseline",metric="weighted_on_time_rate",
                     old=density,new=new,gain_pct=None,change_pp=100*(new-density)))
    pd.DataFrame(rows).to_csv(args.output/"comparison.csv",index=False)


if __name__=="__main__":
    main()
