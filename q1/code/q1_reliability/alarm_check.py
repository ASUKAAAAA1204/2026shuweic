"""Descriptive transfer check: training OOF score threshold, no development tuning."""
from run import np,pd,OUT,MODELS
from assessment.ranking import ranking_metrics


def main():
    data = pd.read_csv(OUT/'predictions.csv.gz',float_precision='round_trip')
    out = []
    for group,mask in [('normal',~data.current_bad),('clean_history',data.bad_fraction==0)]:
        for src,rows in data[mask].groupby('dataset'):
            fit = rows[rows.split=='train']
            dev = rows[rows.split=='validation']
            for m in ('HM','HQ'):
                threshold = fit[m].nlargest(int(np.ceil(.1*len(fit)))).iloc[-1]
                ties = ((.1*len(fit))-(fit[m]>threshold).sum())/(fit[m]==threshold).sum()
                alerts = (dev[m]>threshold).astype(float)+ties*(dev[m]==threshold)
                tp = float(alerts@dev.label)
                out.append(dict(subset=group,dataset=src,model=m,threshold=threshold,tie_fraction=ties,
                                n=len(dev),positive=int(dev.label.sum()),expected_alerts=alerts.sum(),
                                alert_fraction=alerts.mean(),precision=tp/alerts.sum() if alerts.sum() else np.nan,
                                recall=tp/dev.label.sum() if dev.label.sum() else np.nan,true_positive=tp))
    result = pd.DataFrame(out)
    result.to_csv(OUT/'alarm_transfer.csv',index=False)
    print(result[result.model=='HM'].to_string(index=False))
    # Independent record/environment weights must reproduce the main Brier table.
    summary = pd.read_csv(OUT/'summary.csv').query("subset=='all'").set_index('split')
    for split,g in data.groupby('split'):
        w = 1./(g.groupby(['dataset','environment_id','episode_uuid']).label.transform('size')*
                g.groupby(['dataset','environment_id']).episode_uuid.transform('nunique')*
                g.groupby('dataset').environment_id.transform('nunique')*g.dataset.nunique())
        np.testing.assert_allclose(w.sum(),1,atol=1e-12)
        np.testing.assert_allclose(w@g[MODELS].sub(g.label,axis=0).pow(2),
                                   summary.loc[split,MODELS].to_numpy(float),atol=1e-12,rtol=0)
    check = ranking_metrics([1,0,0,0],[.25]*4)
    np.testing.assert_allclose([check['ap'],check['precision10'],check['recall10']],[.25,.25,.1])


if __name__=='__main__':
    main()
