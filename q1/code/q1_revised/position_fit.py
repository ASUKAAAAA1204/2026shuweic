"""Refit the archived low-frequency residual on official Q1 windows and time against CV."""
from time import perf_counter_ns
from functools import lru_cache
import sys
from audit_position import DATA, ROOT, OUT,HORIZONS
sys.path.insert(0,str(ROOT/"code/q1_curves"))
from experiment import Curve,split_bank,features,PRIMARY,SEEDS,np,pd,threadpool_limits
from data import Window,Obs

def main():
    dest=OUT/"curve";dest.mkdir(exist_ok=True)
    meta=pd.read_csv(OUT/"window_manifest.csv")
    ws=[None]*len(meta);distance={};meta["start_s"]=0.
    for (record,side),g in meta.groupby(["record","side"]):
        with np.load(DATA/"records"/f"{record}.npz") as z:
            t=z[side+"__t_s"];x=z[side+"__position"].mean(1)
            if g.split.iloc[0]=="train":
                good=z[side+"__reference_candidate"] & z["head__finite"] & z["head__time_increasing"] & z["head__quaternion_unit_1e3"]
                distance[record,side]=np.linalg.norm(x[good]-z["head__position"][good,0],axis=1)
            for r in g.itertuples():
                i,j=r.last_visible,r.end_exclusive
                ws[r.Index]=Window({},Obs(t[r.history_start:i+1],x[r.history_start:i+1]),t[i:j]-t[i],x[i:j])
                meta.at[r.Index,"start_s"]=t[i]
    cache=[features(w.obs) for w in ws]
    @lru_cache(maxsize=512)
    def scale(ids):
        return float(np.median(np.concatenate([distance[u,s] for u in ids for s in ("left_hand","right_hand")])))
    def bundles(fit,query):
        for src,b in query.groupby("source"):
            ids=tuple(sorted(fit.loc[fit.source==src,"record"].unique()))
            yield src,b,split_bank(fit,src,ws,cache),scale(ids)
    def base(n,o,h):
        if n=="Hold":return np.broadcast_to(o.x[-1],(len(h),3)).copy()
        v=(o.x[-1]-o.x[-2])/(o.t[-1]-o.t[-2])
        return o.x[-1]+(h if n=="CV" else -.2*np.expm1(-h/.2))[:,None]*v
    met=[];fits=[];inner=[];timing=[];fallback=[];selected=[]
    offsets=np.r_[0,np.cumsum([len(w.ages)-1 for w in ws])]
    coords={n:np.full((offsets[-1],3),np.nan) for n in ["B","CV","Hold",*[f"C{s}" for s in SEEDS]]}
    rng=np.random.default_rng(761)
    train=meta[meta.split=="train"]
    for held in [*sorted(train.env.unique()),"development"]:
        fit,query=(train,meta[meta.split=="validation"]) if held=="development" else (train[train.env!=held],train[train.env==held])
        assert not set(query.env)&set(fit.env)
        envs=sorted(fit.env.unique());scores=[]
        for k in range(4):
            mask=fit.env.isin(envs[k::4]);sub,val=fit[~mask],fit[mask]
            for src,b,bank,unit in bundles(sub,val):
                for order in (2,3):
                    start=perf_counter_ns()
                    net=Curve("C",order,ws,bank.ids,bank,unit,PRIMARY)
                    inner.append(dict(outer=held,inner=k,source=src,order=order,scale=unit,fit_windows=len(bank.ids),
                                      held_envs="|".join(envs[k::4]),fit_ms=(perf_counter_ns()-start)/1e6))
                    for i in b.index:
                        w=ws[i];h=w.ages[1:]
                        e=np.linalg.norm(net.forecast(w.obs,h)-w.reference[1:],axis=1)
                        scores.append((order,src,meta.at[i,"env"],meta.at[i,"record"],np.mean([e[h<=H+1e-12].mean() for H in HORIZONS])/unit))
        s=pd.DataFrame(scores,columns=["order","source","env","record","loss"]).groupby(["order","source","env","record"]).loss.mean()
        s=s.groupby(level=[0,1,2]).mean().groupby(level=[0,1]).mean().groupby(level=0).mean()
        order=int(s.idxmin());selected.extend(dict(outer=held,order=q,loss=l,chosen=q==order) for q,l in s.items())
        for src,b,bank,unit in bundles(fit,query):
            for seed in SEEDS:
                start=perf_counter_ns()
                net=Curve("C",order,ws,bank.ids,bank,unit,seed)
                fits.append(dict(outer=held,source=src,order=order,scale=unit,seed=seed,nfit=len(bank.ids),threshold=bank.threshold,
                                 parameters=net.parameters,fit_ms=(perf_counter_ns()-start)/1e6))
                def call(n,o,h):
                    return net.forecast(o,h) if n=="C" else base(n,o,h)
                names=["CV","Hold","B","C"] if seed==PRIMARY else ["C"]
                for r in b.itertuples():
                    i=r.Index;w=ws[i];h=w.ages[1:]
                    for n in names:
                        pred=call(n,w.obs,h)
                        key=f"C{seed}" if n=="C" else n
                        coords[key][offsets[i]:offsets[i+1]]=pred
                        e=np.linalg.norm(pred-w.reference[1:],axis=1)
                        for H in HORIZONS:
                            take=h<=H+1e-12;ade=e[take].mean();fde=e[take][-1]
                            met.append((r.window_id,r.split,src,r.env,r.record,r.side,H,n,seed,ade,fde,ade/unit,fde/unit))
                    if seed==PRIMARY:
                        fallback.append((r.window_id,r.split,len(w.obs.t),not net.active(cache[i][1])))
                        sequence=list(rng.permutation(names));measure={n:[] for n in names}
                        for k in range(5):
                            rot=k%len(sequence)
                            for n in sequence[rot:]+sequence[:rot]:
                                start=perf_counter_ns();call(n,w.obs,h);measure[n].append((perf_counter_ns()-start)/1000)
                        timing.extend((r.window_id,r.split,n,np.median(tt)) for n,tt in measure.items())
        print(held,order,flush=True)
    assert all(np.isfinite(p).all() for p in coords.values())
    np.savez_compressed(dest/"predictions.npz",offsets=offsets,ages=np.concatenate([w.ages[1:] for w in ws]),**coords)
    outputs={"window_metrics.csv.gz":pd.DataFrame(met,columns="window_id split source env record side horizon model seed ade fde nade nfde".split()),
             "timing.csv":pd.DataFrame(timing,columns=["window_id","split","model","us"]),
             "fallback.csv":pd.DataFrame(fallback,columns=["window_id","split","history_points","fallback"]),
             "fit.csv":pd.DataFrame(fits),"inner.csv":pd.DataFrame(inner),"choices.csv":pd.DataFrame(selected)}
    for name,d in outputs.items():d.to_csv(dest/name,index=False)
if __name__=="__main__":
    with threadpool_limits(limits=1):main()
