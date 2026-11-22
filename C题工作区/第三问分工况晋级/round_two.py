"""A single predefined S3 ablation with a grouped, training-only diagnosis."""
import json
import pandas as pd
from evaluate import OUT,CFG,OLD,settings,jobs_for,sample_hidden,replay,evaluate

def main():
    case="S3"
    if (OUT/f"frozen_{case}.json").exists():return
    r=OUT/"round_2"
    r.mkdir(exist_ok=True)
    model=dict(case=case,round=2,kind="family_prior")
    (r/"S3.json").write_text(json.dumps(model,indent=2),encoding="utf8")
    c,tasks,zones,_=settings(case)
    jobs=jobs_for(tasks,case)
    segments=pd.read_csv(CFG/"source_segment_index.csv").query("split=='train' and eligible==1")
    folds=json.loads((OLD/"results/training_evidence/grouped_stress/folds.json").read_bytes())
    rows=[]
    for k,envs in enumerate(folds):
        fit=segments[~segments.environment_id.isin(envs)]
        test=segments[segments.environment_id.isin(envs)]
        pools={f:g.duration_s.to_numpy() for f,g in fit.groupby("family_id")}
        hidden_pool={f:list(g.itertuples()) for f,g in test.groupby("family_id")}
        for seed in range(2026112200,2026112230):
            hidden=sample_hidden((jobs,hidden_pool),seed)
            for method,param in [("old",None),("new",model)]:
                m,_=replay(c,jobs,zones,pools,param,hidden)
                rows.append(dict(fold=k,seed=seed,method=method,**m))
    pd.DataFrame(rows).to_csv(r/"S3_grouped_training.csv",index=False)
    evaluate(case,2)

if __name__=="__main__":main()
