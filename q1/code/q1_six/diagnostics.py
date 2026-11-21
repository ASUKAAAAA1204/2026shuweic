"""Audit saved predictions with hidden query futures and summarize paired effects."""
import json
import numpy as np
import pandas as pd
from scipy.integrate import quad_vec
from run import MODELS, Bank, predict
from data import OUT, Window, Obs, basic, features, load_windows, turn


def verify_turn():
    v=np.array([.6,-1.,.2])
    axis=np.array([0.,0.,1.])
    times=np.array([0.,.02,.1,.2,1.])
    np.testing.assert_array_equal(turn(v,times,axis,0),basic(v,times))
    diffs=[]
    for omega in (.5,10.,80.):
        stop=min(.1,np.pi/(6*omega))
        def velocity(t):
            angle=omega*min(t,stop)
            return np.exp(-5*t)*(v*np.cos(angle)+np.cross(axis,v)*np.sin(angle)+axis*(axis@v)*(1-np.cos(angle)))
        oracle=np.array([quad_vec(velocity,0,h,points=[stop] if stop<h else [])[0] for h in times])
        diffs.append(abs(oracle-turn(v,times,axis,omega)).max())
    assert max(diffs)<1e-10
    return float(max(diffs))


def verify_blind(ws,meta,folds):
    count=0
    largest=0.
    with np.load(OUT/'predictions.npz') as z:
        offsets=z['offsets']
        saved={m:z[m] for m in MODELS}
    for env in sorted(meta.env.unique()):
        train=list(meta.index[meta.env!=env])
        # Redact all held environment future coordinates before reconstructing
        # any bank or querying any of its windows. Keep timestamps for scoring grid.
        redacted=[w if i in train else Window(w.key,Obs(w.obs.t.copy(),w.obs.x.copy()),
                    w.ages.copy(),np.full_like(w.reference,np.nan)) for i,w in enumerate(ws)]
        cache=[features(w.obs) for w in redacted]
        for src,block in meta[meta.env==env].groupby('source'):
            fit=folds[(folds.env==env)&(folds.source==src)].iloc[0]
            ids=meta[(meta.env!=env)&(meta.source==src)].sort_values(['record','side','start_s']).index
            bank=Bank(ids,redacted,cache,fit.speed_q10)
            for i in block.index:
                w=redacted[i]
                for m in MODELS:
                    p,_=predict(m,w.obs,w.ages,bank,fit.c)
                    d=float(abs(p-saved[m][offsets[i]:offsets[i+1]]).max())
                    assert d<1e-12
                    largest=max(largest,d)
                    count+=1
    return count,largest


def main():
    meta=pd.read_csv(OUT/'windows.csv')
    folds=pd.read_json(OUT/'folds.json',precise_float=True)
    metrics=pd.read_csv(OUT/'metrics.csv')
    actions=pd.read_csv(OUT/'inference.csv')
    active=actions[actions.neighbors.notna()].copy()
    pairs=active.assign(neighbor=active.neighbors.str.split('|',regex=False)).explode('neighbor')
    linked=pairs.merge(meta[['window_id','env','source']],left_on='neighbor',right_on='window_id',suffixes=('','_lib'),validate='many_to_one')
    assert len(pairs)==len(linked) and (linked.env!=linked.env_lib).all() and (linked.source==linked.source_lib).all()
    assert linked.groupby(['window_id','model']).neighbor.nunique().eq(5).all()
    ws=load_windows()
    assert [w.key['window_id'] for w in ws]==list(meta.window_id)
    blind_n,blind_error=verify_blind(ws,meta,folds)
    wide=metrics.pivot(index=['window_id','horizon'],columns='model',values='ade')
    counts=[]
    for span,part in [('three_horizons_mean',wide.groupby(level=0).mean()),*[(str(h),g.droplevel(1)) for h,g in wide.groupby(level=1)]]:
        for m in MODELS[1:]:
            diff=part[m]-part.B
            counts.append(dict(horizon=span,model=m,improved=int((diff < -1e-12).sum()),
                               tied=int((diff.abs() <= 1e-12).sum()),worse=int((diff>1e-12).sum())))
    pd.DataFrame(counts).to_csv(OUT/'win_tie_loss.csv',index=False)
    groups=['model','active','horizon','source','env','record','side']
    v=metrics.groupby(groups)[['ade','fde','along','lateral','nade']].mean()
    for _ in range(3):
        groups=groups[:-1]
        v=v.groupby(level=groups).mean()
    v.reset_index().to_csv(OUT/'active_diagnostics.csv',index=False)
    # Paired active-only contrasts (same windows and same coordinate units within source).
    keys=['window_id','horizon']
    b=metrics.loc[metrics.model=='B',keys+['ade','along','lateral']].set_index(keys)
    diff=metrics[metrics.active & metrics.model.isin(['T','K','K0'])].join(b,on=keys,rsuffix='_B')
    for col in ('ade','along','lateral'):
        diff[col]=diff[col]-diff[col+'_B']
    g=['model','horizon','source','env','record','side']
    tab=diff.groupby(g)[['ade','along','lateral']].mean()
    for _ in range(3):
        g=g[:-1]
        tab=tab.groupby(level=g).mean()
    tab.to_csv(OUT/'active_paired.csv')
    d=dict(neighbor_links_checked=len(linked),blind_replays=blind_n,max_blind_difference=blind_error,
           turn_vs_quadrature_max=verify_turn(),same_source_distinct_env=True)
    (OUT/'diagnostic_checks.json').write_text(json.dumps(d,indent=2),encoding='utf8')
    print(d)


if __name__=='__main__':
    main()
