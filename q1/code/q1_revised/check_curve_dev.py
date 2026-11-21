"""Independent all-training inner-grid check on the corrected development protocol."""
from audit_position import DATA, ROOT, OUT,HORIZONS
import sys
sys.path.insert(0,str(ROOT/"code/q1_curves"))
from experiment import Curve, split_bank, features, PRIMARY, np, pd, threadpool_limits
from data import Window, Obs

def main():
    m=pd.read_csv(OUT/"window_manifest.csv")
    m["start_s"]=m.last_visible # Only used to order windows within a record/hand.
    ws=[None]*len(m);dist={}
    for (uid,side),g in m.groupby(["record","side"]):
        with np.load(DATA/"records"/f"{uid}.npz") as z:
            t=z[side+"__t_s"];x=z[side+"__position"].mean(1)
            if g.split.iloc[0]=="train":
                ok=z[side+"__reference_candidate"] & z["head__finite"] & z["head__quaternion_unit_1e3"] & z["head__time_increasing"]
                dist[(uid,side)]=np.linalg.norm(x[ok]-z["head__position"][ok,0],axis=1)
            for r in g.itertuples():
                i,j=r.last_visible,r.end_exclusive
                ws[r.Index]=Window({},Obs(t[r.history_start:i+1],x[r.history_start:i+1]),t[i:j]-t[i],x[i:j])
    cache=[features(w.obs) for w in ws]
    train=m[m.split=="train"];dev=m[m.split=="validation"]
    envs=sorted(train.env.unique());rows=[]
    def scores(fit,target,orders,fold):
        for src,block in target.groupby("source"):
            ids=fit.loc[fit.source==src,"record"].unique()
            scale=np.median(np.concatenate([dist[uid,side] for uid in ids for side in ["left_hand","right_hand"]]))
            bank=split_bank(fit,src,ws,cache)
            for order in orders:
                model=Curve("C",order,ws,bank.ids,bank,scale,PRIMARY)
                for ix in block.index:
                    w=ws[ix];e=np.linalg.norm(model.forecast(w.obs,w.ages[1:])-w.reference[1:],axis=1)
                    score=np.mean([e[w.ages[1:]<=h+1e-12].mean() for h in HORIZONS])
                    rows.append(dict(ix=ix,window_id=m.at[ix,"window_id"],order=order,fold=fold,
                                     source=src,env=m.at[ix,"env"],record=m.at[ix,"record"],nade=score/scale))
    for j in range(4):
        outer=train.env.isin(envs[j::4]);scores(train[~outer],train[outer],[2,3],str(j))
    grid=pd.DataFrame(rows).groupby(["order","source","env","record"]).nade.mean().groupby(level=[0,1,2]).mean().groupby(level=[0,1]).mean().groupby(level=0).mean()
    chosen=int(grid.idxmin());scores(train,dev,[chosen],"development")
    pd.DataFrame(rows).to_csv(OUT/"curve_dev_independent.csv.gz",index=False)
    print("training inner scores",grid.to_dict(),"development order",chosen)
if __name__=="__main__":
    with threadpool_limits(limits=1):main()
