"""Choose from the completion-protected shortlist using a second, previously used block."""
import argparse
import json
import pandas as pd
from study import HERE,OUT,DEV,MET,bench,measure,save

def check(design):
    if any(OUT.glob("round_*/passed.lock")):raise RuntimeError("Already promoted")
    f=OUT/f"design_{design}"
    table=pd.read_csv(f/"protected_shortlist.csv")
    models=json.loads((HERE/f"grid_{design}.json").read_bytes())
    eligible=table.query("completion_margin>=.002-1e-12").head(4)
    reference=pd.read_csv(HERE.parent/"results/round_2/paired_trials.csv").query("method=='candidate'").set_index("seed")
    rows=[];b=bench()
    for i in eligible.id.to_list():
        path=f/f"historical_{i}.csv"
        if not path.exists():measure(b,models[i],reference.index.to_list()).to_csv(path)
        trial=pd.read_csv(path).set_index("seed")
        on=trial[MET[0]].mean()
        on_gain=(trial[MET[0]]-reference[MET[0]]).mean()
        done_gain=(trial[MET[1]]-reference[MET[1]]).mean()
        d=table.set_index("id").loc[i]
        # Reject a model if either historical block loses completion or on-time performance.
        safe=min(done_gain,d.completion_margin)>=0 and min(on_gain,d[MET[0]]-.8522135416666666)>0
        score=min(on,d[MET[0]])+.35*min(done_gain,d.completion_margin)
        rows.append(dict(id=i,historical_on=on,historical_gain=on_gain,historical_completion_gain=done_gain,
                         first_on=d[MET[0]],first_completion_gain=d.completion_margin,eligible=safe,score=score))
    if not rows:print("No model eligible for historical review");return
    out=pd.DataFrame(rows).sort_values("score",ascending=False)
    out.to_csv(f/"historical_summary.csv",index=False)
    choices=out[out.eligible]
    if len(choices):save(f/"nominee.json",models[int(choices.iloc[0].id)])
    print(out.round(5).to_string(index=False),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("design",type=int)
    check(p.parse_args().design)
