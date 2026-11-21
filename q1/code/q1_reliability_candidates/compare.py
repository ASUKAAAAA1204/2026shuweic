"""Compare four frozen quality forecasters with the two- and three-state anchors."""
import hashlib
import json
import sys

from video_inputs import ROOT, DATA, OUT, KEY, VISUAL, np, pd
sys.path.insert(0,str(ROOT/'code/q1_reliability'))
from run import three_state_rates, recovery_state, probabilities, GROUP
from assessment.ranking import ranking_metrics
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

MODELS = ['State','State3','Pool','Chain','Motion','Video']
STATE_ORDER = ['clean_history','recent_recovery','flagged']
KIN = ['median_speed','std_speed','last_speed','joint_spread']


def prepare():
    archive = ROOT/'results/q1_reliability/predictions.csv.gz'
    meta = pd.read_csv(archive,float_precision='round_trip')
    video = pd.read_csv(OUT/'video_inputs.csv.gz',float_precision='round_trip')
    df = meta.merge(video,on=KEY,validate='one_to_one')
    assert len(df)==len(meta)==len(video)==17724 and not df.duplicated(KEY).any()
    official = pd.read_csv(next((ROOT/'C_Problem_Data').rglob('record_split.csv'))).set_index('episode_uuid')
    np.testing.assert_array_equal(official.loc[df.episode_uuid,'split'],df.split)
    np.testing.assert_array_equal(official.loc[df.episode_uuid,'environment_id'],df.environment_id)
    assert set(df.split)=={'train','validation'} and df.groupby('environment_id').split.nunique().max()==1
    counts = {}
    df['state'] = recovery_state(df)
    df['steps'] = 0
    for uid,g in df.groupby('episode_uuid'):
        train = g.split.iloc[0]=='train'
        matrix = np.zeros((3,3),dtype=int)
        with np.load(DATA/'records'/f'{uid}.npz',allow_pickle=False) as z:
            for side,block in g.groupby('side'):
                t = z[side+'__t_s']
                ids = block['index'].to_numpy()
                lo = np.searchsorted(t,t[ids]-.2)
                steps = [max(1,int(np.floor(.2/np.median(np.diff(t[j:i+1]))+.5))) for j,i in zip(lo,ids)]
                df.loc[block.index,'steps'] = steps
                if not train:
                    continue
                bad = z[side+'__bad']
                known = z[side+'__quality_known']
                lower = np.searchsorted(t,t-.2)
                cumul = np.r_[0,np.cumsum(bad)]
                code = np.where(bad,2,(cumul[1:]-cumul[lower]>0).astype(int))
                np.testing.assert_array_equal(np.array(STATE_ORDER)[code[ids]],block.state)
                unavailable = np.r_[0,np.cumsum(~known)]
                gaps = np.r_[False,np.diff(t)>3*np.median(np.diff(t))]
                gap_count = np.r_[0,np.cumsum(gaps)]
                eligible = (unavailable[1:]==unavailable[lower]) & (gap_count[1:]==gap_count[lower+1])
                eligible &= t >= t[0]+.2-1e-9
                valid = eligible[:-1] & eligible[1:] & ~gaps[1:] & (np.diff(t)>0)
                np.add.at(matrix,(code[:-1][valid],code[1:][valid]),1)
        if train:
            counts[uid] = matrix
    assert df.steps.min()>=1 and set(counts)==set(df.query("split=='train'").episode_uuid)
    return df,counts,hashlib.sha256(archive.read_bytes()).hexdigest()


def fit_clean_tree(fit,cols):
    train = fit.loc[fit.state.eq('clean_history')].dropna(subset=cols)
    if train.label.nunique()<2:
        return None
    tree = HistGradientBoostingClassifier(max_depth=2,max_leaf_nodes=4,min_samples_leaf=100,
               early_stopping=False,learning_rate=.05,max_iter=60,l2_regularization=5,random_state=20261121)
    return tree.fit(train[cols],train.label)


def clean_tree(fit,x,cols,base):
    tree = fit_clean_tree(fit,cols)
    use = x.state.eq('clean_history') & x[cols].notna().all(axis=1)
    pred = base.copy()
    if tree is not None and use.any():
        pred.loc[use] = .5*(base.loc[use]+tree.predict_proba(x.loc[use,cols])[:,1])
    return pred


def forecast(fit,x,counts,held,parameters):
    p = pd.DataFrame(index=x.index)
    p['State'] = x.current_bad.map(probabilities(fit))
    rates = three_state_rates(fit).p
    p['State3'] = x.state.map(rates).fillna(p.State)
    cells = fit.groupby(['dataset','state']).label.agg(['sum','count'])
    cells['p'] = (cells['sum']+50*cells.index.get_level_values('state').map(rates))/(cells['count']+50)
    pooled = cells.p.reindex(pd.MultiIndex.from_frame(x[['dataset','state']])).to_numpy()
    p['Pool'] = np.where(np.isfinite(pooled),pooled,p.State3)
    totals = sum(counts[k] for k in fit.episode_uuid.unique())
    matrix = (totals+1)/(totals.sum(axis=1)[:,None]+3)
    for i,state in enumerate(STATE_ORDER):
        parameters.append(dict(held=held,state=state,train_pairs=int(totals[i].sum()),
                               **dict(zip(STATE_ORDER,matrix[i]))))
    survival = matrix.copy()
    survival[:,2] = 0.
    codes = pd.Categorical(x.state,categories=STATE_ORDER).codes
    for h in x.steps.unique():
        mask = (x.steps==h).to_numpy()
        prob = 1.-np.linalg.matrix_power(survival,int(h)).sum(axis=1)
        p.loc[mask,'Chain'] = prob[codes[mask]]
    p['Motion'] = clean_tree(fit,x,KIN,p.State3)
    p['Video'] = clean_tree(fit,x,KIN+VISUAL,p.State3)
    missing = x[VISUAL].isna().any(axis=1)
    p.loc[missing,'Video'] = p.loc[missing,'Motion']
    assert p.notna().all().all() and p.ge(0).all().all() and p.le(1).all().all()
    np.testing.assert_array_equal(p.loc[x.state.ne('clean_history'),'Video'],p.loc[x.state.ne('clean_history'),'State3'])
    return p


def paired(env):
    pivot = env.pivot(index='environment_id',columns='dataset',values=MODELS)
    arrays = [pivot.xs(s,axis=1,level='dataset')[MODELS].to_numpy() for s in sorted(env.dataset.unique())]
    rng = np.random.default_rng(20261120)
    batches,remaining,rejected = [],5000,0
    while remaining:
        idx = rng.integers(len(pivot),size=(remaining,len(pivot)))
        keep = np.logical_and.reduce([np.isfinite(a[idx,0]).any(axis=1) for a in arrays])
        if keep.any():
            batches.append(np.mean([np.nanmean(a[idx[keep]],axis=1) for a in arrays],axis=0))
        rejected += int((~keep).sum())
        remaining -= int(keep.sum())
    boot = pd.DataFrame(np.vstack(batches),columns=MODELS)
    point = env.groupby('dataset')[MODELS].mean().mean()
    rows = []
    for m,r in [(m,'State3') for m in MODELS[2:]]+[('State3','State'),('Video','Motion'),('Motion','State')]:
        gain = 100*(1-point[m]/point[r])
        ci = np.percentile(100*(1-boot[m]/boot[r]),[2.5,97.5,.625,99.375])
        rows.append((m,r,gain,*ci,rejected))
    return pd.DataFrame(rows,columns=['candidate','reference','gain_pct','lo95','hi95','lo9875','hi9875','rejected'])


def score(rows):
    summaries,environments,ranking,alarms = [],[],[],[]
    for split,g in rows.groupby('split'):
        for subset in ['all',*STATE_ORDER]:
            x = g if subset=='all' else g[g.state==subset]
            rec = x[GROUP].join(x[MODELS].sub(x.label,axis=0).pow(2)).groupby(GROUP).mean()
            env = rec.groupby(level=GROUP[:2]).mean().reset_index()
            environments.append(env.assign(split=split,subset=subset))
            src = env.groupby('dataset')[MODELS].mean()
            summaries.append(dict(split=split,subset=subset,n=len(x),events=x.label.sum(),**src.mean().to_dict()))
            if split=='train' and subset=='all':
                paired(env).to_csv(OUT/'paired.csv',index=False)
        for (s,e),block in g[g.state=='clean_history'].groupby(GROUP[:2]):
            for m in MODELS:
                ranking.append(dict(split=split,source=s,env=e,model=m,**ranking_metrics(block.label,block[m])))
    for s,g in rows[rows.state=='clean_history'].groupby('dataset'):
        fit = g[g.split=='train']
        val = g[g.split=='validation']
        for m in MODELS:
            cut = fit[m].nlargest(int(np.ceil(.1*len(fit)))).iloc[-1]
            frac = (.1*len(fit)-(fit[m]>cut).sum())/(fit[m]==cut).sum()
            alerts = (val[m]>cut).astype(float)+(val[m]==cut)*frac
            alarms.append(dict(dataset=s,model=m,threshold=cut,tie_fraction=frac,n=len(val),
                               positives=val.label.sum(),alerts=alerts.sum(),hits=alerts@val.label))
    summary = pd.DataFrame(summaries)
    summary.to_csv(OUT/'summary.csv',index=False)
    pd.concat(environments).to_csv(OUT/'environments.csv',index=False)
    ranks = pd.DataFrame(ranking)
    ranks.to_csv(OUT/'ranking.csv',index=False)
    ranks.dropna(subset=['ap']).groupby(['split','source','model'])[['ap','precision10','recall10','prevalence']].mean().groupby(
        level=['split','model']).mean().to_csv(OUT/'ranking_summary.csv')
    pd.DataFrame(alarms).to_csv(OUT/'alarm_transfer.csv',index=False)
    print(summary.to_string(index=False))


def main():
    data,counts,sha = prepare()
    train,val = data[data.split=='train'],data[data.split=='validation']
    inputs = ['dataset','current_bad','state','steps',*KIN,*VISUAL]
    parts,params = [],[]
    for env in [*sorted(train.environment_id.unique()),'development']:
        fit,x = (train,val) if env=='development' else (train[train.environment_id!=env],train[train.environment_id==env])
        assert set(fit.environment_id).isdisjoint(x.environment_id)
        p = forecast(fit,x[inputs],counts,env,params)
        np.testing.assert_allclose(p[['State','State3']],x[['State','State3']],atol=1e-14,rtol=0)
        parts.append(x[KEY+['dataset','environment_id','split','state','label','steps']].join(p))
    rows = pd.concat(parts)
    assert len(rows)==17724 and not rows.duplicated(KEY).any()
    rows.to_csv(OUT/'predictions.csv.gz',index=False)
    pd.DataFrame(params).to_csv(OUT/'transition_matrices.csv',index=False)
    score(rows)
    stats = data.groupby(['dataset','split']).agg(n=('index','size'),available=('gray','count'),
            max_lag_s=('lag_s','max'),min_camera_frames=('camera_frames','min'))
    stats.to_csv(OUT/'coverage.csv')
    (OUT/'audit.json').write_text(json.dumps(dict(anchor_sha256=sha,total=len(rows),
        train_envs=train.environment_id.nunique(),dev_envs=val.environment_id.nunique(),test_records_used=0),indent=2))


if __name__=='__main__':
    with threadpool_limits(limits=1):
        main()
