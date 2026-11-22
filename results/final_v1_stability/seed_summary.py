"""Separate variability between internal streams from variability between tasks."""
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
KEY = ["case_id","seed","alpha","beta"]
DEFAULT = 20261122
RATE = "weighted_on_time_rate"


def main():
    raw=pd.read_csv(OUT/"seed_trials.csv.gz")
    assert raw.valid_protocol.all() and not raw.invalid_actions.any()
    assert raw.internal_seed.nunique()==20 and raw[KEY].drop_duplicates().shape[0]==450
    profile=[]
    for key,g in raw.groupby(KEY):
        g=g.set_index("internal_seed").sort_index()
        base=g.loc[DEFAULT]
        alts=g.drop(DEFAULT)
        counts=g.first_action.value_counts()
        rates=g[RATE]
        limits=[rates.idxmin(),rates.idxmax()]
        profile.append(dict(zip(KEY,key),first_choices=len(counts),path_count=g.path_hash.nunique(),
            first_default_match=(alts.first_action==base.first_action).mean(),
            first_pair_match=(counts*(counts-1)).sum()/(len(g)*(len(g)-1)),
            path_default_match=(alts.path_hash==base.path_hash).mean(),default_rate=base[RATE],
            rate_default_match=(alts[RATE]==base[RATE]).mean(),rate_mean=rates.mean(),sd_pp=rates.std(ddof=1)*100,
            span_pp=(rates.max()-rates.min())*100,min_rate=rates.min(),max_rate=rates.max(),
            min_internal_seed=limits[0],max_internal_seed=limits[1]))
    profile=pd.DataFrame(profile)
    profile.to_csv(OUT/"fixed_instance_stability.csv",index=False)
    summary=[]
    means=[]
    for scope,subset in [("nominal",profile.query("alpha==1 and beta==.5")),("all_five",profile)]:
        for case in ["overall","S1","S2","S3"]:
            sample=subset if case=="overall" else subset[subset.case_id==case]
            rows=raw.merge(sample[KEY],on=KEY,validate="many_to_one")
            per_seed=rows.groupby("internal_seed")[RATE].mean().rename("on_time_rate")
            means.extend(dict(scope=scope,case_id=case,internal_seed=s,on_time_rate=v) for s,v in per_seed.items())
            summary.append(dict(scope=scope,case_id=case,instances=len(sample),
                first_default_match=sample.first_default_match.mean(),first_pair_match=sample.first_pair_match.mean(),
                path_default_match=sample.path_default_match.mean(),rate_default_match=sample.rate_default_match.mean(),
                instances_first_varies=int(sample.first_choices.gt(1).sum()),instances_path_varies=int(sample.path_count.gt(1).sum()),
                instances_outcome_varies=int(sample.span_pp.gt(0).sum()),mean_within_sd_pp=sample.sd_pp.mean(),
                median_within_span_pp=sample.span_pp.median(),p90_within_span_pp=sample.span_pp.quantile(.9),
                max_within_span_pp=sample.span_pp.max(),mean_within_span_pp=sample.span_pp.mean(),
                seed_mean_min=per_seed.min(),seed_mean_max=per_seed.max(),seed_mean=per_seed.mean(),
                default_mean=per_seed.loc[DEFAULT]))
    pd.DataFrame(summary).to_csv(OUT/"seed_summary.csv",index=False)
    pd.DataFrame(means).to_csv(OUT/"seed_averages.csv",index=False)
    first=raw.groupby(["case_id","alpha","beta","internal_seed"]).first_action.nunique()
    assert first.eq(1).all() # initial observation is shared by the 30 hidden draws
    firsts=raw.groupby(["case_id","alpha","beta","internal_seed"]).first_action.first().reset_index()
    firsts.groupby(["case_id","alpha","beta","first_action"]).size().rename("internal_seed_count").to_csv(OUT/"first_choices.csv")


if __name__=="__main__":
    main()
