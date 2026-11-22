"""Refine the post-round-one shortlist using historical development seeds only."""
import argparse
import json
import pandas as pd
from study import OUT,HERE,DEV,MET,bench,measure,save


def refine(design):
    if any(OUT.glob("round_*/passed.lock")):raise RuntimeError("Already promoted")
    folder=OUT/f"design_{design}"
    grid=json.loads((HERE/f"grid_{design}.json").read_bytes())
    small=pd.read_csv(folder/"screen.csv")
    b=bench()
    rows=[]
    ref=pd.read_csv(OUT/"dev_tier3.csv")[MET[1]].mean()
    # Also check three candidates with the strongest completion records on the small screen.
    extra=small.nlargest(3,MET[1]).id.to_list()
    finalists=pd.read_csv(folder/"finalists.csv").id.to_list()
    for i in sorted(set(extra+finalists)):
        path=folder/f"dev_{i}.csv"
        if not path.exists():measure(b,grid[i],DEV).to_csv(path)
        result=pd.read_csv(path)[MET].mean()
        rows.append(dict(id=i,**result,completion_margin=result[MET[1]]-ref))
    table=pd.DataFrame(rows).sort_values(MET[0],ascending=False)
    table.to_csv(folder/"protected_shortlist.csv",index=False)
    eligible=table[table.completion_margin>=.002-1e-12]
    if eligible.empty:
        print("No candidate with the development completion buffer")
        return
    chosen=eligible.iloc[0]
    save(folder/"nominee.json",grid[int(chosen.id)])
    print(chosen.to_string(),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("design",type=int)
    refine(p.parse_args().design)
