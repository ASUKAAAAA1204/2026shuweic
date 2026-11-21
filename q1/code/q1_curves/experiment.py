"""Run the two approved nested environment-holdout prediction experiments."""
import sys
from pathlib import Path
from time import perf_counter
import json
import hashlib

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'code/q1_head_to_head'))
from compare import (np, pd, sklearn, platform, threadpool_limits, threadpool_info,
                     OLD, OUT as PREVIOUS, Network, Bank, SEEDS, scales, features,
                     load_windows, predict, EPS, HORIZONS, average)
from data import Obs
from models import Curve

OUT = ROOT/'results/q1_curves'
GRID = {'F': [(1, 1), (2, 2), (2, 3)], 'C': [2, 3]}
PRIMARY = SEEDS.start


def input_check():
    archive = json.loads((PREVIOUS/'run.json').read_text())
    hashes = {k: hashlib.sha256((OLD/k).read_bytes()).hexdigest() for k in archive['hashes']}
    assert hashes == archive['hashes']
    meta = pd.read_csv(PREVIOUS/'window_manifest.csv', float_precision='round_trip')
    ws = load_windows()
    assert len(ws) == len(meta) == 1523
    for key in ws[0].key:
        np.testing.assert_array_equal(meta[key], [w.key[key] for w in ws])
    offsets = np.r_[0, np.cumsum([len(w.ages) for w in ws])]
    with np.load(OLD/'predictions.npz') as z:
        np.testing.assert_array_equal(z['offsets'], offsets)
        np.testing.assert_array_equal(z['ref'], np.vstack([w.reference for w in ws]))
        np.testing.assert_array_equal(z['t'], np.concatenate([w.ages for w in ws]))
    with np.load(PREVIOUS/'predictions.npz') as z:
        np.testing.assert_array_equal(z['offsets'], offsets)
        controls = {k:z[k] for k in z.files if k == 'B' or k.startswith('M1')}
    return meta, ws, offsets, controls, hashes


def split_bank(train, src, ws, cache):
    ids = train[train.source == src].sort_values(['record','side','start_s']).index
    threshold = np.quantile([cache[i][1][0] for i in ids], .1) if len(ids) else np.inf
    return Bank(ids, ws, cache, threshold)


def split_detail(ids, ws):
    gaps = np.array([max(0., 1.-ws[i].ages[-1]) for i in ids])
    affected = gaps[gaps > EPS]
    return dict(eligible_training=len(ids), extended_targets=len(affected),
                extended_mean_s=affected.mean() if len(affected) else 0.,
                extended_max_s=affected.max() if len(affected) else 0.)


def integral_weights(w):
    weights = np.zeros_like(w.ages)
    for horizon in HORIZONS:
        h = w.ages[w.ages <= horizon+EPS]
        half = np.diff(h)/(2*h[-1]*len(HORIZONS))
        weights[:len(h)-1] += half
        weights[1:len(h)] += half
    return weights


def select(train, ws, cache, weights, outer, fits, selections):
    groups = sorted(train.env.unique())
    rows = []
    for j in range(4):
        held = train[train.env.isin(groups[j::4])]
        sub = train[~train.env.isin(groups[j::4])]
        assert set(sub.env).isdisjoint(held.env) and outer not in train.env.values
        scale_map, pool = scales(sub)
        for src, block in held.groupby('source'):
            bank = split_bank(sub, src, ws, cache)
            scale = scale_map.get(src, pool)
            for name, orders in GRID.items():
                for rank, order in enumerate(orders):
                    tic = perf_counter()
                    model = Curve(name, order, ws, bank.ids, bank, scale, PRIMARY)
                    for i in block.index:
                        w = ws[i]
                        dist = np.linalg.norm(model.forecast(w.obs, w.ages)-w.reference, axis=1)
                        rows.append((i, name, rank, weights[i] @ dist / scale))
                    fits.append(dict(outer=outer, inner=j, source=src, model=name, order=str(order),
                                     held_envs='|'.join(groups[j::4]), scale=scale, threshold=bank.threshold,
                                     fit_score_s=perf_counter()-tic, **split_detail(bank.ids, ws)))
    df = pd.DataFrame(rows, columns=['i','model','rank','nade']).join(train, on='i')
    assert len(df) == len(train)*sum(map(len, GRID.values())) and not df.duplicated(['i','model','rank']).any()
    keys = ['model','rank','source','env','record','side']
    scores = df.groupby(keys).nade.mean()
    for _ in range(4):
        keys = keys[:-1]
        scores = scores.groupby(level=keys).mean()
    chosen = {}
    for name, options in GRID.items():
        best = scores[name].idxmin()
        chosen[name] = options[best]
        for rank, order in enumerate(options):
            selections.append((outer,name,str(order),scores[name][rank],rank==best))
    return chosen


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    meta, ws, offsets, archived, hashes = input_check()
    cache = [features(w.obs) for w in ws]
    weights = [integral_weights(w) for w in ws]
    keys = ['B','F','M1','C'] + [f'{m}_{s}' for s in list(SEEDS)[1:] for m in ('C','M1')]
    saved = {k:np.full((offsets[-1],3), np.nan) for k in keys}
    timings, fits, actions, inner, choices, metrics = [], [], [], [], [], []
    rng = np.random.default_rng(761)
    max_control_error = 0.
    for env in sorted(meta.env.unique()):
        held, train = meta[meta.env==env], meta[meta.env!=env]
        chosen = select(train, ws, cache, weights, env, inner, choices)
        scale_map, pool = scales(train)
        for src, block in held.groupby('source'):
            bank = split_bank(train, src, ws, cache)
            scale = scale_map.get(src, pool)
            assert np.allclose(block.scale, scale, atol=1e-14, rtol=0)
            assert not set(held.index).intersection(bank.ids)
            for seed in SEEDS:
                names = ['B','F','M1','C'] if seed == PRIMARY else ['M1','C']
                models = {}
                for name in names:
                    start = perf_counter()
                    if name == 'M1':
                        models[name] = Network(name, ws, bank.ids, bank, scale, seed)
                    elif name != 'B':
                        models[name] = Curve(name, chosen[name], ws, bank.ids, bank, scale, seed)
                    model = models.get(name)
                    fits.append(dict(env=env,source=src,model=name,seed=seed,fit_s=perf_counter()-start,
                                     numeric_bytes=getattr(model,'nbytes',0),scale=scale,threshold=bank.threshold,
                                     network_parameters=(113 if name=='M1' and model.model is not None else getattr(model,'parameters',0)),
                                     **split_detail(bank.ids,ws)))
                def call(name, obs, h):
                    if name == 'B':
                        return predict('B',obs,h,bank,0)[0]
                    return models[name].forecast(obs,h)
                for i in block.index:
                    w = ws[i]
                    sl = slice(offsets[i],offsets[i+1])
                    v, f = cache[i][:2]
                    d = v/f[0] if f[0]>0 and f[0]>=bank.threshold else None
                    for name in names:
                        p = call(name,w.obs,w.ages)
                        key = name if seed==PRIMARY else f'{name}_{seed}'
                        saved[key][sl] = p
                        if key in archived:
                            err = np.max(np.abs(p-archived[key][sl]))
                            max_control_error = max(err,max_control_error)
                            assert err<=1e-10
                        assert np.isfinite(p).all() and np.array_equal(p[0],w.obs.x[-1])
                        if seed==PRIMARY:
                            np.testing.assert_array_equal(call(name,Obs(w.obs.t.copy(),w.obs.x.copy()),w.ages.copy()),p)
                            active = name=='B' or (bank.eligible(f) and len(bank.ids)>=5 and getattr(models.get(name),'ready',True))
                            if not active:
                                np.testing.assert_array_equal(p, saved['B'][sl])
                            actions.append((i,w.key['window_id'],env,src,name,active))
                        error = p-w.reference
                        dist = np.linalg.norm(error,axis=1)
                        for h in HORIZONS:
                            use = w.ages<=h+EPS
                            age = w.ages[use]
                            ade = average(age,dist[use])
                            along = lateral = np.nan
                            if d is not None:
                                proj = error[use] @ d
                                along = average(age,proj)
                                lateral = average(age,np.linalg.norm(error[use]-proj[:,None]*d,axis=1))
                            metrics.append((i,name,seed,h,ade/scale,ade,dist[use][-1],along,lateral))
                    if seed==PRIMARY:
                        order = list(rng.permutation(names))
                        times = {name:[] for name in names}
                        for j in range(5):
                            k = j%len(order)
                            for name in order[k:]+order[:k]:
                                tic = perf_counter()
                                call(name,w.obs,w.ages)
                                times[name].append((perf_counter()-tic)*1e6)
                        timings.extend((w.key['window_id'],name,np.median(t)) for name,t in times.items())
        print(f'{env}: {chosen}',flush=True)
    assert all(np.isfinite(p).all() for p in saved.values())
    np.savez_compressed(OUT/'predictions.npz',offsets=offsets,**saved)
    pd.DataFrame(metrics,columns=['i','model','seed','horizon','nade','ade','fde','along','lateral']).join(
        meta[['window_id','env','source','record','side']],on='i').to_csv(OUT/'metrics.csv',index=False)
    pd.DataFrame(inner).to_csv(OUT/'inner.csv',index=False)
    pd.DataFrame(fits).to_csv(OUT/'fit.csv',index=False)
    pd.DataFrame(choices,columns=['env','model','order','inner_nade','selected']).to_csv(OUT/'choices.csv',index=False)
    pd.DataFrame(actions,columns=['i','window_id','env','source','model','active']).to_csv(OUT/'actions.csv',index=False)
    pd.DataFrame(timings,columns=['window_id','model','us']).to_csv(OUT/'timing.csv',index=False)
    audit = dict(train_windows=len(ws),training_only=True,input_hashes=hashes,max_control_abs_error=max_control_error,
                 detached_replays=len(actions),python=platform.python_version(),numpy=np.__version__,
                 sklearn=sklearn.__version__,cpu=platform.processor(),threads=threadpool_info())
    (OUT/'run.json').write_text(json.dumps(audit,indent=2))


if __name__=='__main__':
    with threadpool_limits(limits=1):
        main()
