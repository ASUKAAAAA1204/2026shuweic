"""Equal-source scores, global-environment paired bootstrap and sign-randomization tests."""
import numpy as np
import pandas as pd

def contrasts(env, pairs):
    models=sorted(set(m for pair in pairs for m in pair))
    env=env.pivot(index="env",columns="source",values=models)
    a=np.stack([env.xs(s,level=1,axis=1)[models].to_numpy() for s in sorted(env.columns.levels[1])])
    n=a.shape[1]
    assert np.array_equal(np.isfinite(a),np.repeat(np.isfinite(a[:,:,:1]),len(models),axis=2))
    rng=np.random.default_rng(20261121)
    boot=[];remaining=5000;rejected=0
    while remaining:
        ids=rng.integers(n,size=(remaining,n))
        good=np.isfinite(a[:,ids,0]).any(axis=-1).all(0)
        if good.any():
            boot.extend(np.nanmean(a[:,ids[good],:],axis=2).mean(0))
        rejected+=int((~good).sum());remaining-=int(good.sum())
    boot=np.asarray(boot)
    point=np.nanmean(a,axis=1).mean(0)
    signs=1-2*((np.arange(2**n)[:,None]>>np.arange(n))&1)
    out=[]
    for m,r in pairs:
        j,k=models.index(m),models.index(r)
        g=100*(1-boot[:,j]/boot[:,k]);lo,hi=np.quantile(g,[.025,.975])
        d=a[:,:,k]-a[:,:,j]
        contributions=np.nansum(d/np.isfinite(d).sum(1)[:,None],axis=0)/a.shape[0]
        pdiff=abs(contributions.sum())
        p=(np.abs(signs@contributions)>=pdiff-1e-15).mean()
        out.append(dict(model=m,reference=r,gain_pct=100*(1-point[j]/point[k]),lo95=lo,hi95=hi,
                        p=float(p),draws=len(g),rejected=rejected,envs=n))
    order=sorted(out,key=lambda x:x["p"]);prior=0.
    for i,it in enumerate(order):
        prior=min(1.,max(prior,it["p"]*(len(out)-i)))
        it["holm"]=prior
    return pd.DataFrame(out)

