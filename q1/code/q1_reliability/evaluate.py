"""Matched probability accuracy, within-environment alarms, and clustered uncertainty."""
from run import OUT, MODELS, GROUP, np, pd
from assessment.ranking import ranking_metrics


def intervals(env):
    table = env.pivot(index='environment_id',columns='dataset',values=MODELS)
    arrays = [table.xs(s,axis=1,level='dataset')[MODELS].to_numpy()
              for s in sorted(env.dataset.unique())]
    rng = np.random.default_rng(20261120)
    draws,remaining,rejected = [],5000,0
    while remaining:
        idx = rng.integers(len(table),size=(remaining,len(table)))
        keep = np.logical_and.reduce([np.isfinite(a[idx,0]).any(axis=1) for a in arrays])
        rejected += int((~keep).sum())
        if keep.any():
            draws.append(np.mean([np.nanmean(a[idx[keep]],axis=1) for a in arrays],axis=0))
        remaining -= int(keep.sum())
    result = pd.DataFrame(np.vstack(draws),columns=MODELS)
    point = env.groupby('dataset')[MODELS].mean().mean()
    pairs = [(m,'State') for m in MODELS if m!='State']+[('D','D50'),('HQ','D'),('HM','HQ'),('Q','LR'),('State3','D')]
    rows = []
    for m,r in pairs:
        ci = np.percentile(100*(1-result[m]/result[r]),[2.5,97.5,1.25,98.75])
        rows.append((m,r,100*(1-point[m]/point[r]),*ci,rejected))
    return pd.DataFrame(rows,columns=['candidate','reference','gain_pct','lo95','hi95','lo975','hi975','rejected'])


def main():
    data = pd.read_csv(OUT/'predictions.csv.gz',float_precision='round_trip')
    summary,source,environment,count,rank,calibration = [],[],[],[],[],[]
    for split,frame in data.groupby('split'):
        subsets = {'all':frame,'normal':frame[~frame.current_bad],'flagged':frame[frame.current_bad],
                   'clean_history':frame[frame.bad_fraction==0],'recent_recovery':frame[(~frame.current_bad)&(frame.bad_fraction>0)]}
        for tag,x in subsets.items():
            sq = x[MODELS].sub(x.label,axis=0).pow(2)
            rec = x[GROUP].join(sq).groupby(GROUP).mean()
            env = rec.groupby(level=GROUP[:2]).mean()
            src = env.groupby(level='dataset').mean()
            if split=='train' and tag=='all':
                intervals(env.reset_index()).to_csv(OUT/'paired.csv',index=False)
            environment.append(env.reset_index().assign(split=split,subset=tag))
            source.append(src.reset_index().assign(split=split,subset=tag))
            summary.append(dict(split=split,subset=tag,**src.mean().to_dict()))
            for s,g in [('all',x),*list(x.groupby('dataset'))]:
                pos = g[g.label==1]
                count.append(dict(split=split,subset=tag,dataset=s,n=len(g),positive=len(pos),
                             distinct_positive_times=len(pos.drop_duplicates(['episode_uuid','index'])),
                             records=g.episode_uuid.nunique(),envs=g.environment_id.nunique(),
                             positive_envs=pos.environment_id.nunique()))
            if tag in ('normal','clean_history'):
                for (s,e),g in x.groupby(GROUP[:2]):
                    for m in MODELS:
                        rank.append(dict(split=split,subset=tag,dataset=s,environment_id=e,model=m,
                                         **ranking_metrics(g.label,g[m])))
        for m in MODELS:
            bins = pd.cut(frame[m],[-1,.005,.01,.025,.05,.1,.5,.9,1],include_lowest=True)
            tab = frame.assign(bin=bins).groupby(['dataset','bin'],observed=True).agg(
                n=('label','size'),positive=('label','sum'),mean_pred=(m,'mean'),observed=('label','mean'))
            calibration.append(tab.reset_index().assign(split=split,model=m))
    pd.concat(environment).to_csv(OUT/'environments.csv',index=False)
    pd.concat(source).to_csv(OUT/'sources.csv',index=False)
    pd.DataFrame(count).to_csv(OUT/'counts.csv',index=False)
    pd.concat(calibration).to_csv(OUT/'calibration.csv',index=False)
    detail = pd.DataFrame(rank)
    detail.to_csv(OUT/'ranking_environments.csv',index=False)
    valid = detail.dropna(subset=['ap'])
    rank_src = valid.groupby(['split','subset','model','dataset'])[['ap','prevalence','precision10','recall10']].mean()
    rank_src.to_csv(OUT/'ranking_sources.csv')
    rank_src.groupby(level=['split','subset','model']).mean().to_csv(OUT/'ranking_summary.csv')
    main_scores = pd.DataFrame(summary)
    main_scores.to_csv(OUT/'summary.csv',index=False)
    print(main_scores.to_string(index=False))


if __name__=='__main__':
    main()
