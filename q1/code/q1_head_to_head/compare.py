"""Same-process paired timing and scoring of fixed kNN and local residual MLPs."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'):
    os.environ[name]='1'
import sys
from pathlib import Path
from time import perf_counter_ns
import warnings
import json
import platform
import hashlib
import numpy as np
import pandas as pd
import sklearn
from sklearn.neural_network import MLPRegressor
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_info, threadpool_limits

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code/q1_six'))
from run import Bank, predict, scales
from data import HORIZONS, EPS, basic, features, average, load_windows

OLD=ROOT/'results/q1_six_train'
OUT=ROOT/'results/q1_knn_mlp'
SEEDS=range(20261120,20261125)
NAMES=('B','K0','K','M3','M1')


def vector(obs,kind):
    if kind=='M1':
        v,f,_,_=features(obs)
        return v,f
    v=(obs.x[-1]-obs.x[-2])/(obs.t[-1]-obs.t[-2])
    grid=obs.t[-1]+np.array([-.2,-2/15,-1/15])
    points=np.column_stack([np.interp(grid,obs.t,obs.x[:,j]) for j in range(3)])
    return v,np.r_[(points-obs.x[-1]).ravel(),v]


def design(f,h):
    return np.column_stack((np.broadcast_to(f,(len(h),len(f))),h))


class Network:
    def __init__(self,kind,ws,ids,bank,scale,seed):
        self.kind,self.threshold,self.scale=kind,bank.threshold,scale
        self.model=None
        self.nbytes=0
        self.train_windows=len(ids)
        if len(ids)==0 or (kind=='M1' and len(ids)<5):
            return
        xs,ys=[],[]
        for i in ids:
            w=ws[i]
            v,f=vector(w.obs,kind)
            ix=np.linspace(1,len(w.ages)-1,6).round().astype(int)
            h=w.ages[ix]
            res=w.reference[ix]-w.obs.x[-1]-basic(v,h)
            target=res/(scale*(-np.expm1(-h/.2))[:,None])
            if kind=='M1':
                target=target@(v/f[0])
            xs.append(design(f,h))
            ys.append(target)
        x=np.concatenate(xs)
        y=np.concatenate(ys)
        self.mu=x.mean(axis=0)
        self.sd=x.std(axis=0)
        self.sd[self.sd<1e-12]=1.
        model=MLPRegressor(hidden_layer_sizes=(16,),activation='tanh',alpha=.01,solver='adam',
                           learning_rate_init=.003,batch_size=min(256,len(x)),max_iter=60,
                           n_iter_no_change=61,random_state=seed)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',ConvergenceWarning)
            model.fit((x-self.mu)/self.sd,y)
        assert model.n_iter_==60
        self.model=model
        check=(x[:6]-self.mu)/self.sd
        manual=np.tanh(check@model.coefs_[0]+model.intercepts_[0])@model.coefs_[1]+model.intercepts_[1]
        np.testing.assert_allclose(manual,model.predict(check).reshape(manual.shape),atol=1e-12)
        self.nbytes=sum(a.nbytes for a in [*model.coefs_,*model.intercepts_,self.mu,self.sd])

    def forecast(self,obs,h):
        v,f=vector(obs,self.kind)
        base=obs.x[-1]+basic(v,h)
        if self.model is None or (self.kind=='M1' and
                not (np.isfinite(f).all() and min(f[:3])>0 and min(f[:3])>=self.threshold)):
            return base
        x=(design(f,h)-self.mu)/self.sd
        # Exact dense inference without per-call sklearn estimator input validation.
        z=np.tanh(x@self.model.coefs_[0]+self.model.intercepts_[0])
        y=z@self.model.coefs_[1]+self.model.intercepts_[1]
        residual=y*(self.scale*(-np.expm1(-h/.2))[:,None])
        if self.kind=='M1':
            residual=.5*residual*(v/f[0])
        return base+residual


def run():
    OUT.mkdir(parents=True,exist_ok=True)
    meta=pd.read_csv(OLD/'windows.csv',float_precision='round_trip')
    folds=pd.DataFrame(json.loads((OLD/'folds.json').read_text(encoding='utf8')))
    ws=load_windows()
    assert meta.window_id.tolist()==[w.key['window_id'] for w in ws]
    for field in ('env','source','record','side','history_start','start_index','end_exclusive','start_s'):
        np.testing.assert_array_equal(meta[field],[w.key[field] for w in ws])
    cache=[features(w.obs) for w in ws]
    cost,latency,metrics=[],[],[]
    offsets=np.r_[0,np.cumsum([len(w.ages) for w in ws])]
    total=int(offsets[-1])
    saved={m:np.full((total,3),np.nan) for m in NAMES}
    seed_predictions={}
    speed_rng=np.random.default_rng(761)
    with np.load(OLD/'predictions.npz') as archive:
        ref={m:archive[m] for m in ('K','K0','B')}
        np.testing.assert_array_equal(offsets,archive['offsets'])
        np.testing.assert_array_equal(np.concatenate([w.ages for w in ws]),archive['t'])
        np.testing.assert_array_equal(np.vstack([w.reference for w in ws]),archive['ref'])
    for env in sorted(meta.env.unique()):
        held=meta[meta.env==env]
        train=meta[meta.env!=env]
        scale_map,pool=scales(train)
        for src,block in held.groupby('source'):
            ids=train[train.source==src].sort_values(['record','side','start_s']).index
            params=folds[(folds.env==env)&(folds.source==src)].iloc[0]
            scale=scale_map.get(src,pool)
            assert np.isclose(scale,params.scale) and np.allclose(block.scale,scale)
            start=perf_counter_ns()
            bank=Bank(ids,ws,cache,params.speed_q10)
            assert not (set(meta.loc[bank.ids,'env']) & {env})
            cost.append((env,src,'K/K0 shared',0,len(bank.ids),(perf_counter_ns()-start)/1e6,bank.nbytes))
            primary={}
            for seed in SEEDS:
                nets={}
                for kind in ('M3','M1'):
                    start=perf_counter_ns()
                    net=Network(kind,ws,ids if kind=='M3' else bank.ids,bank,scale,seed)
                    cost.append((env,src,kind,seed,net.train_windows,(perf_counter_ns()-start)/1e6,net.nbytes))
                    nets[kind]=net
                if seed==SEEDS.start:
                    primary=nets
                for i in block.index:
                    w=ws[i]
                    sl=slice(offsets[i],offsets[i+1])
                    names=NAMES if seed==SEEDS.start else ('M3','M1')
                    for m in names:
                        p=nets[m].forecast(w.obs,w.ages) if m in nets else predict(m,w.obs,w.ages,bank,0)[0]
                        assert np.isfinite(p).all() and np.array_equal(p[0],w.obs.x[-1])
                        if seed==SEEDS.start:
                            saved[m][sl]=p
                            if m in ref:
                                np.testing.assert_array_equal(p,ref[m][sl])
                        else:
                            key=f'{m}_{seed}'
                            if key not in seed_predictions:
                                seed_predictions[key]=np.empty((total,3))
                            seed_predictions[key][sl]=p
                        dist=np.linalg.norm(p-w.reference,axis=1)
                        for h in HORIZONS:
                            ix=w.ages<=h+EPS
                            ade=average(w.ages[ix],dist[ix])
                            metrics.append((w.key['window_id'],env,src,w.key['record'],w.key['side'],h,m,seed,ade/scale,ade,dist[ix][-1]))
            for i in block.index:
                w=ws[i]
                def call(m):
                    return primary[m].forecast(w.obs,w.ages) if m in primary else predict(m,w.obs,w.ages,bank,0)[0]
                times={m:[] for m in NAMES}
                for m in NAMES:
                    np.testing.assert_array_equal(call(m),saved[m][offsets[i]:offsets[i+1]])
                order=list(speed_rng.permutation(NAMES))
                for shift in range(5):
                    for m in order[shift:]+order[:shift]:
                        start=perf_counter_ns()
                        call(m)
                        times[m].append((perf_counter_ns()-start)/1000)
                latency.extend((w.key['window_id'],m,np.median(tt)) for m,tt in times.items())
    metrics=pd.DataFrame(metrics,columns=['window_id','env','source','record','side','horizon','model','seed','nade','ade','fde'])
    metrics.to_csv(OUT/'metrics.csv',index=False)
    pd.DataFrame(latency,columns=['window_id','model','us']).to_csv(OUT/'timing.csv',index=False)
    pd.DataFrame(cost,columns=['env','source','model','seed','train_windows','fit_or_build_ms','inference_buffer_bytes']).to_csv(OUT/'fit.csv',index=False)
    np.savez_compressed(OUT/'predictions.npz',offsets=offsets,**saved,**seed_predictions)
    meta.to_csv(OUT/'window_manifest.csv',index=False)
    diagnostics=dict(window_count=len(ws),environment_count=meta.env.nunique(),record_count=meta.record.nunique(),
        guards='exact window keys, timestamps, split fields, reference coords, per-fold scale, archived B/K/K0 coordinates',
        training_only=True,python=platform.python_version(),numpy=np.__version__,sklearn=sklearn.__version__,
        cpu=platform.processor(),blas=threadpool_info(),seeds=list(SEEDS),
        hashes={f:hashlib.sha256((OLD/f).read_bytes()).hexdigest() for f in ('windows.csv','folds.json','predictions.npz')})
    (OUT/'run.json').write_text(json.dumps(diagnostics,indent=2),encoding='utf8')
    print('Completed',len(metrics),'window-level metric rows.',flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1):
        run()
