"""Score saved forecasts only after training and neutral-test prediction finish."""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile
from replay import OUT, PRIMARY, PRIMARY_NAMES, HORIZONS, EPS, np, pd, csv, member
from audit.pairing import pairwise


def crosscheck(z, pred, scores):
    """Run only the inspected, hash-pinned bundle scorer, in an isolated directory."""
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        body = z.read(member(z,'score.py'))
        assert hashlib.sha256(body).hexdigest()=='b8378e01c045d5cb1f0d4913180c58ff214476a4ff0c95ed2eef1726c0820c60'
        (root/'score.py').write_bytes(body)
        (root/'data').mkdir()
        for name in ('data/windows.csv','local_test_ade.csv','test_reference.csv','summary.json'):
            (root/name).write_bytes(z.read(member(z,name)))
        report = {}
        for m in PRIMARY_NAMES:
            p = root/'forecast.csv'
            pred[pred.model==m][['window_id','frame_index','x','y','z']].to_csv(p,index=False)
            output = subprocess.check_output([sys.executable,'-IB',str(root/'score.py'),str(p)],cwd=root,text=True)
            official = json.JSONDecoder().raw_decode(output)[0]
            for key,name in [('original kNN',m),('MLP gate','PackGate'),('conditional residual MLP','PackResidual')]:
                report[name] = abs(official[key]-scores.at[name,'nade'])
        assert max(report.values())<1e-12
        return report


def main(path):
    with ZipFile(path) as z:
        ref = csv(z,'test_reference.csv')
        archived = csv(z,'local_test_ade.csv')
        metadata = csv(z,'data/windows.csv').query("split=='test'").rename(columns={
            'dataset':'source','environment_id':'env','episode_uuid':'record'})
        scales = json.loads(z.read(member(z,'summary.json')))['scale_by_source_training_only']
    keys = ['window_id','frame_index']
    pred = pd.read_csv(OUT/'predictions.csv.gz',float_precision='round_trip')
    assert not pred.duplicated(['seed','model',*keys]).any()
    assert not ref.duplicated(keys).any() and len(ref)==11178
    scored = pred.merge(ref,on=keys,suffixes=('','_true'),validate='many_to_one')
    assert len(scored)==len(pred) and (scored.groupby(['seed','model']).size()==len(ref)).all()
    scored['error'] = np.linalg.norm(scored[['x','y','z']].values-scored[['x_true','y_true','z_true']].values,axis=1)
    rows = []
    for h in HORIZONS:
        mask = (scored.age_from_anchor_s>0)&(scored.age_from_anchor_s<=h+EPS)
        part = scored[mask].sort_values('frame_index').groupby(['seed','model','window_id']).error.agg(ade='mean',fde='last')
        rows.append(part.reset_index().assign(horizon=h))
    for name,col in [('PackGate','mlp_gate_ade'),('PackResidual','conditional_residual_mlp_ade')]:
        rows.append(archived[['window_id','horizon_s',col]].rename(columns={'horizon_s':'horizon',col:'ade'}).assign(model=name,seed=PRIMARY))
    prep = pd.read_csv(OUT/'preprocessing/windows.csv')
    for candidate,label in [('B','GeomMedian'),('C','LowPass'),('D','OneEuro')]:
        rows.append(prep.loc[prep.model==candidate,['window_id','horizon_s','ade','fde']].rename(
            columns={'horizon_s':'horizon'}).assign(model=label,seed=PRIMARY))
    raw = pd.concat(rows,ignore_index=True).merge(metadata[['window_id','source','env','record','side']],on='window_id',validate='many_to_one')
    assert not raw.duplicated(['seed','model','window_id','horizon']).any() and (raw.groupby(['seed','model']).size()==374*3).all()
    raw['nade'] = raw.ade/raw.source.map(scales)
    raw.to_csv(OUT/'window_scores.csv',index=False)
    group = ['seed','model','horizon','source','env','record']
    record = raw.groupby(group)[['nade','ade','fde']].mean()
    env = record.groupby(level=group[:-1]).mean()
    env.to_csv(OUT/'environments.csv')
    source = env.groupby(level=group[:-2]).mean()
    source.to_csv(OUT/'sources.csv')
    horizons = source.groupby(level=group[:-3]).nade.mean()
    horizons.to_csv(OUT/'horizons.csv')
    all_scores = horizons.groupby(level=['seed','model']).mean()
    all_scores.to_csv(OUT/'seed_scores.csv')
    summary = all_scores.loc[PRIMARY].to_frame()
    summary['gain_vs_B_pct'] = 100*(1-summary.nade/summary.at['B','nade'])
    times = pd.read_csv(OUT/'timing.csv').groupby('model').us
    summary['p50_us'],summary['p95_us'] = times.median(),times.quantile(.95)
    summary.sort_values('nade').to_csv(OUT/'summary.csv')
    overall,local = pairwise(env.loc[PRIMARY].reset_index(),summary.nade)
    overall.to_csv(OUT/'paired.csv',index=False)
    local.to_csv(OUT/'paired_environments.csv',index=False)
    with ZipFile(path) as z:
        verified = crosscheck(z,pred[pred.seed==PRIMARY],summary)
    prep_scores = pd.read_csv(OUT/'preprocessing/all.csv').set_index('model').norm_ade
    for here,other in [('B','A'),('PackGate','MLP_gate'),('PackResidual','residual_MLP'),
                       ('GeomMedian','B'),('LowPass','C'),('OneEuro','D')]:
        assert abs(summary.at[here,'nade']-prep_scores[other])<1e-12
    (OUT/'score_audit.json').write_text(json.dumps(dict(frame_count=len(ref),window_count=len(metadata),
        compared_to_bundle=verified,checked_window_horizon_rows=len(raw)),indent=2))
    print(summary.sort_values('nade').to_string())


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('archive',type=Path)
    main(parser.parse_args().archive)
