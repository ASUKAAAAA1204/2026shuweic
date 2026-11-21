"""Replay frozen position predictors on a supplied neutral window manifest."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
from time import perf_counter_ns
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code/q1_curves'))
from experiment import (np,pd,Network,Curve,GRID,SEEDS,PRIMARY,split_bank,scales,
                        threadpool_limits,threadpool_info,predict,features,HORIZONS,EPS)
from run import COEFS
from data import Window,Obs,DATA,basic

OUT = ROOT/'results/q1_neutral_models'
PRIMARY_NAMES = ('B','A','L','T','K','K0','M1','M3','F','C')
FITTED = ('M1','M3','C')


def member(bundle, name):
    hits = [p for p in bundle.namelist() if p.endswith('/'+name)]
    assert len(hits)==1
    return hits[0]


def csv(bundle,name):
    with bundle.open(member(bundle,name)) as f:
        return pd.read_csv(f,float_precision='round_trip')


def load(path):
    with ZipFile(path) as z:
        manifest = csv(z,'data/windows.csv')
        queries = csv(z,'queries.csv')
        spec = json.loads(z.read(member(z,'summary.json')))
    meta = manifest.rename(columns={'episode_uuid':'record','environment_id':'env','dataset':'source','anchor_s':'start_s'})
    assert meta.window_id.is_unique and not queries.duplicated(['window_id','frame_index']).any()
    official = pd.read_csv(next((ROOT/'C_Problem_Data').rglob('record_split.csv'))).set_index('episode_uuid')
    for k,col in [('env','environment_id'),('split','split'),('source','dataset')]:
        np.testing.assert_array_equal(meta[k],official.loc[meta.record,col])
    assert meta.groupby('env').split.nunique().max()==1
    qs = dict(tuple(queries.groupby('window_id',sort=False)))
    assert set(qs)==set(meta.window_id)
    ws = [None]*len(meta)
    meta['endpoint'] = np.nan
    for (uid,side),block in meta.groupby(['record','side']):
        with np.load(DATA/'records'/f'{uid}.npz',allow_pickle=False) as z:
            t,joints = z[side+'__t_s'],z[side+'__position']
            for row in block.itertuples():
                lo,first,end = row.history_start,row.start_index,row.end_index_exclusive
                assert row.window_id==f'{uid}_{side}_{first-1}' and abs(t[first-1]-row.start_s)<1e-12
                obs = Obs(t[lo:first],joints[lo:first].mean(axis=1))
                q = qs[row.window_id]
                np.testing.assert_array_equal(q.frame_index,np.arange(first,end))
                np.testing.assert_allclose(q.timestamp_s,t[first:end],rtol=0,atol=1e-12)
                np.testing.assert_allclose(q.age_from_anchor_s,t[first:end]-obs.t[-1],rtol=0,atol=1e-12)
                ages = np.r_[0.,q.age_from_anchor_s.to_numpy()]
                truth = None
                if row.split=='train':
                    truth = joints[first-1:end].mean(axis=1)
                    meta.loc[row.Index,'endpoint'] = np.linalg.norm(truth[-1]-obs.x[-1])
                ws[row.Index] = Window({'window_id':row.window_id},obs,ages,truth)
    train,test = meta[meta.split=='train'],meta[meta.split=='test']
    assert (len(train),train.record.nunique(),train.env.nunique())==(1439,55,16)
    assert (len(test),test.record.nunique(),test.env.nunique())==(374,7,4)
    for src,scale in scales(train)[0].items():
        np.testing.assert_allclose(scale,spec['scale_by_source_training_only'][src],rtol=1e-12)
    return train,test,ws


def select(train,ws,cache,selection_file=None):
    """Select only on held training environments, using the new per-frame ADE."""
    groups = sorted(train.env.unique())
    options = {'L':list(COEFS),**GRID}
    rows = []
    for name,grid in options.items():
        folds = np.array_split(groups,4) if name=='L' else [groups[j::4] for j in range(4)]
        for fold,envs in enumerate(folds):
            held = train[train.env.isin(envs)]
            sub = train[~train.env.isin(envs)]
            scale_map,pool = scales(sub)
            for src,block in held.groupby('source'):
                scale = scale_map.get(src,pool)
                bank = split_bank(sub,src,ws,cache)
                for rank,option in enumerate(grid):
                    model = None if name=='L' else Curve(name,option,ws,bank.ids,bank,scale,PRIMARY)
                    for i in block.index:
                        w = ws[i]
                        if name=='L':
                            p = w.obs.x[-1]+basic(cache[i][0],w.ages,'L',option)
                        else:
                            p = model.forecast(w.obs,w.ages)
                        dist = np.linalg.norm(p[1:]-w.reference[1:],axis=1)
                        score = np.mean([dist[w.ages[1:]<=h+EPS].mean() for h in HORIZONS])/scale
                        rows.append((i,name,rank,fold,score))
    vals = pd.DataFrame(rows,columns=['i','model','rank','fold','nade']).join(train,on='i')
    assert not vals.duplicated(['i','model','rank']).any() and len(vals)==len(train)*sum(map(len,options.values()))
    levels = ['model','rank','source','env','record']
    stats = vals.groupby(levels).nade.mean()
    for _ in range(3):
        levels = levels[:-1]
        stats = stats.groupby(level=levels).mean()
    chosen = {}
    for name,grid in options.items():
        pick = min(range(len(grid)),key=lambda j:(stats[name][j],abs(grid[j]) if name=='L' else j,j))
        chosen[name] = grid[pick]
    table = stats.rename('train_cv_nade').reset_index()
    table['option'] = [str(options[n][j]) for n,j in zip(table.model,table['rank'])]
    table['selected'] = [str(chosen[n])==v for n,v in zip(table.model,table.option)]
    table.to_csv(selection_file or OUT/'selection.csv',index=False)
    return chosen


def run(path):
    OUT.mkdir(parents=True,exist_ok=True)
    train,test,ws = load(path)
    cache = [features(w.obs) for w in ws]
    chosen = select(train,ws,cache)
    print('Training-only selections:',chosen,flush=True)
    scale_map,_ = scales(train)
    predictions,timing,actions,fit_rows = [],[],[],[]
    rng = np.random.default_rng(761)
    for src,block in test.groupby('source'):
        bank = split_bank(train,src,ws,cache)
        ids = train[train.source==src].sort_values(['record','side','start_s']).index
        assert set(ids).isdisjoint(test.index) and all(ws[i].reference is None for i in test.index)
        for seed in SEEDS:
            names = PRIMARY_NAMES if seed==PRIMARY else FITTED
            models = {}
            for name in names:
                tic = perf_counter_ns()
                if name in ('F','C'):
                    models[name] = Curve(name,chosen[name],ws,bank.ids,bank,scale_map[src],seed)
                elif name in ('M1','M3'):
                    models[name] = Network(name,ws,ids if name=='M3' else bank.ids,bank,scale_map[src],seed)
                if name in models:
                    fit_rows.append((src,name,seed,(perf_counter_ns()-tic)/1e6,models[name].nbytes))
            def call(name,obs,h):
                if name in models:
                    return models[name].forecast(obs,h)
                return predict(name,obs,h,bank,chosen['L'])[0]
            for row in block.itertuples():
                w = ws[row.Index]
                frames = np.arange(row.start_index,row.end_index_exclusive)
                for name in names:
                    p = call(name,w.obs,w.ages)
                    assert np.array_equal(p[0],w.obs.x[-1]) and np.isfinite(p).all()
                    if seed==PRIMARY:
                        np.testing.assert_array_equal(p,call(name,Obs(w.obs.t.copy(),w.obs.x.copy()),w.ages.copy()))
                        if name not in models:
                            _,info = predict(name,w.obs,w.ages,bank,chosen['L'])
                            status,partial = info['status'],info['partial']
                        else:
                            status = 'active' if name=='M3' or (len(bank.ids)>=5 and bank.eligible(cache[row.Index][1]) and getattr(models[name],'ready',True)) else 'fallback'
                            partial = 0
                        actions.append((row.window_id,name,status,partial))
                    predictions.append(pd.DataFrame(dict(window_id=row.window_id,frame_index=frames,
                                      model=name,seed=seed,x=p[1:,0],y=p[1:,1],z=p[1:,2])))
                if seed==PRIMARY:
                    order = list(rng.permutation(names))
                    times = {n:[] for n in names}
                    for shift in range(5):
                        for n in order[shift:]+order[:shift]:
                            tic = perf_counter_ns()
                            call(n,w.obs,w.ages)
                            times[n].append((perf_counter_ns()-tic)/1000)
                    timing.extend((row.window_id,n,np.median(t)) for n,t in times.items())
    pred = pd.concat(predictions,ignore_index=True)
    assert len(pred)==11178*(len(PRIMARY_NAMES)+4*len(FITTED))
    pred.to_csv(OUT/'predictions.csv.gz',index=False)
    pred.loc[(pred.model=='K')&(pred.seed==PRIMARY),['window_id','frame_index','x','y','z']].to_csv(OUT/'original_knn.csv',index=False)
    pd.DataFrame(timing,columns=['window_id','model','us']).to_csv(OUT/'timing.csv',index=False)
    pd.DataFrame(fit_rows,columns=['source','model','seed','fit_ms','numeric_bytes']).to_csv(OUT/'fit.csv',index=False)
    pd.DataFrame(actions,columns=['window_id','model','status','partial_points']).to_csv(OUT/'actions.csv',index=False)
    info = dict(archive_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),train=len(train),test=len(test),
                test_query_frames=11178,selection=chosen,train_scales=scale_map,threads=threadpool_info(),
                primary_seed=PRIMARY,detached_replays=len(actions),model_test_reference_available=False)
    (OUT/'run.json').write_text(json.dumps(info,indent=2))


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('archive',type=Path)
    with threadpool_limits(limits=1):
        run(parser.parse_args().archive)
