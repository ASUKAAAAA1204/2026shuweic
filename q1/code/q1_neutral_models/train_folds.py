"""Repeat nested leave-environment-out comparison on the neutral training windows."""
import argparse
from pathlib import Path
from replay import (OUT,PRIMARY_NAMES,FITTED,PRIMARY,SEEDS,load,select,split_bank,scales,
                    Curve,Network,features,predict,HORIZONS,EPS,threadpool_limits,np,pd)


def intervals(env,score):
    models = list(score.index)
    pivot = env.groupby(['env','source','model']).nade.mean().unstack(['source','model'])
    arrays = [pivot[s][models].to_numpy() for s in sorted(set(pivot.columns.get_level_values(0)))]
    rng = np.random.default_rng(PRIMARY)
    samples,remaining,rejected = [],5000,0
    while remaining:
        draws = rng.integers(len(pivot),size=(remaining,len(pivot)))
        valid = np.logical_and.reduce([np.isfinite(a[draws,0]).any(axis=1) for a in arrays])
        rejected += int((~valid).sum())
        if valid.any():
            samples.append(np.mean([np.nanmean(a[draws[valid]],axis=1) for a in arrays],axis=0))
        remaining -= int(valid.sum())
    boot = pd.DataFrame(np.vstack(samples),columns=models)
    pairs = []
    for m,ref in [(m,'B') for m in models if m!='B']+[('C','M1'),('K','M1'),('K','K0')]:
        gains = 100*(1-boot[m]/boot[ref])
        bounds = np.percentile(gains,[2.5,97.5,1.25,98.75,.625,99.375])
        pairs.append((m,ref,100*(1-score[m]/score[ref]),*bounds,rejected))
    return pd.DataFrame(pairs,columns=['candidate','reference','gain_pct','lo95','hi95',
                                      'lo975','hi975','lo9875','hi9875','rejected'])


def main(path):
    out = OUT/'train_folds'
    out.mkdir(exist_ok=True,parents=True)
    train,_,ws = load(path)
    cache = [features(w.obs) if w.reference is not None else None for w in ws]
    rows,selected = [],[]
    for env in sorted(train.env.unique()):
        tr,held = train[train.env!=env],train[train.env==env]
        options = select(tr,ws,cache,out/f'selection_{env}.csv')
        scale_map,pool = scales(tr)
        for src,block in held.groupby('source'):
            scale = scale_map.get(src,pool)
            bank = split_bank(tr,src,ws,cache)
            ids = tr[tr.source==src].sort_values(['record','side','start_s']).index
            selected.append((env,src,scale,bank.threshold,len(bank.ids),options['L'],str(options['F']),options['C']))
            assert not set(ids).intersection(held.index)
            for seed in SEEDS:
                names = PRIMARY_NAMES if seed==PRIMARY else FITTED
                for m in names:
                    model = None
                    if m in ('F','C'):
                        model = Curve(m,options[m],ws,bank.ids,bank,scale,seed)
                    elif m in ('M1','M3'):
                        model = Network(m,ws,ids if m=='M3' else bank.ids,bank,scale,seed)
                    for i in block.index:
                        w = ws[i]
                        p = (model.forecast(w.obs,w.ages) if model else predict(m,w.obs,w.ages,bank,options['L'])[0])
                        assert np.isfinite(p).all() and np.array_equal(p[0],w.obs.x[-1])
                        error = np.linalg.norm(p[1:]-w.reference[1:],axis=1)
                        for h in HORIZONS:
                            e = error[w.ages[1:]<=h+EPS]
                            rows.append((i,seed,m,h,e.mean()/scale,e.mean(),e[-1]))
        print('Completed',env,flush=True)
    scores = pd.DataFrame(rows,columns=['i','seed','model','horizon','nade','ade','fde']).join(
        train[['window_id','source','env','record']],on='i')
    assert len(scores)==1439*3*(len(PRIMARY_NAMES)+4*len(FITTED)) and not scores.duplicated(['i','horizon','seed','model']).any()
    scores.to_csv(out/'window_scores.csv.gz',index=False)
    pd.DataFrame(selected,columns=['env','source','scale','threshold','bank_size','L','F','C']).to_csv(out/'selected.csv',index=False)
    levels = ['seed','model','horizon','source','env','record']
    records = scores.groupby(levels)[['nade','ade','fde']].mean()
    by_env = records.groupby(level=levels[:-1]).mean()
    by_env.to_csv(out/'environments.csv')
    sources = by_env.groupby(level=levels[:-2]).mean()
    sources.to_csv(out/'sources.csv')
    headline = sources.nade.groupby(level=levels[:3]).mean().groupby(level=levels[:2]).mean()
    headline.to_csv(out/'seed_scores.csv')
    pairs = intervals(by_env.loc[PRIMARY].reset_index(),headline.loc[PRIMARY])
    pairs.to_csv(out/'paired.csv',index=False)
    print(pairs.to_string(index=False))


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('archive',type=Path)
    with threadpool_limits(limits=1):
        main(parser.parse_args().archive)
