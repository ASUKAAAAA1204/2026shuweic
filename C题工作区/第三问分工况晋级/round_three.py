import json
import pandas as pd
from evaluate import OUT,CFG,OLD,settings,jobs_for,sample_hidden,replay,evaluate

def main():
    case="S3"
    if (OUT/f"frozen_{case}.json").exists():return
    r=OUT/"round_3";r.mkdir(exist_ok=True)
    c,tasks,zones,_=settings(case)
    jobs=jobs_for(tasks,case)
    segments=pd.read_csv(CFG/"source_segment_index.csv").query("split=='train' and eligible==1")
    folds=json.loads((OLD/"results/training_evidence/grouped_stress/folds.json").read_bytes())
    models=[dict(case=case,round=3,kind="rolling",completion_weight=w,prior_count=v,scenarios=24,steps=42)
            for w,v in [(0.25,2),(0.55,5),(0.85,10)]]
    rows=[]
    for fold,held in enumerate(folds):
        fit=segments[~segments.environment_id.isin(held)]
        test=segments[segments.environment_id.isin(held)]
        pools={f:g.duration_s.to_numpy() for f,g in fit.groupby("family_id")}
        hidden_pool={f:list(g.itertuples()) for f,g in test.groupby("family_id")}
        for seed in range(2026112200,2026112212):
            hidden=sample_hidden((jobs,hidden_pool),seed)
            for i,model in enumerate(models):
                m,_=replay(c,jobs,zones,pools,model,hidden)
                rows.append(dict(fold=fold,seed=seed,model=i,**m))
    d=pd.DataFrame(rows);d.to_csv(r/"S3_training.csv",index=False)
    s=d.groupby("model").mean(numeric_only=True)
    best=int((s.weighted_on_time_rate+.35*s.weighted_completion_rate).idxmax())
    (r/"S3.json").write_text(json.dumps(models[best],indent=2),encoding="utf8")
    print(s[["weighted_on_time_rate","weighted_completion_rate"]].to_string(),flush=True)
    evaluate(case,3)

if __name__=="__main__":main()
