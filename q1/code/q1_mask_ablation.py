"""Quality-mask ablation on all eligible overlapping one-second forecast windows."""
import numpy as np
import pandas as pd
from q1 import DATA, ROOT, TAU, aggregate

DEST = ROOT / 'results/q1_ablation'


def run():
    records = pd.read_csv(DATA / 'record_index.csv').query("split in ['train','validation']")
    rows = []
    for rec in records.itertuples():
        with np.load(DATA / 'records' / f'{rec.episode_uuid}.npz', allow_pickle=False) as data:
            for side in ('left_hand', 'right_hand'):
                def field(key):
                    return data[f'{side}__{key}']
                t = field('t_s')
                pos = field('position').mean(axis=1)
                valid = field('reference_candidate')
                bad = field('bad')
                numeric = field('finite') & field('quaternion_unit_1e3') & field('time_increasing') & field('quality_known')
                gap = field('gap_before')
                assert np.all(np.diff(t) > 0)
                assert np.array_equal(valid, numeric & ~bad)
                dt = np.median(np.diff(t))
                for i in range(6, len(t)):
                    j = np.searchsorted(t, t[i]+1+1e-9, side='right')
                    if t[j-1]-t[i] < 1-dt*.51 or not valid[i:j].all() or gap[i-5:j].any():
                        continue
                    row = dict(split=rec.split, dataset=rec.dataset, environment_id=rec.environment_id,
                               episode_uuid=rec.episode_uuid, side=side, start_index=i,
                               recent_bad=bool(bad[i-2:i].any()))
                    for name, allowed in [('Q0', numeric), ('Q1', valid)]:
                        idx = np.flatnonzero(allowed[i-6:i])+i-6
                        row[name] = np.nan
                        if len(idx) < 2:
                            continue
                        a,b = idx[-2:]
                        velocity = (pos[b]-pos[a])/(t[b]-t[a])
                        displacement = -TAU*np.expm1(-(t[i:j]-t[b])/TAU)
                        error = np.linalg.norm(pos[b] + displacement[:,None]*velocity - pos[i:j], axis=1)
                        row[name] = error.mean()
                    rows.append(row)
    frame = pd.DataFrame(rows)
    DEST.mkdir(exist_ok=True, parents=True)
    frame.to_csv(DEST/'mask_windows.csv.gz',index=False)
    parts = []
    for subset, chunk in [('all', frame), ('recent_bad', frame[frame.recent_bad])]:
        for (split, src), group in chunk.groupby(['split','dataset']):
            paired = group.dropna(subset=['Q0','Q1'])
            stats = dict(subset=subset,split=split,dataset=src,candidates=len(group),
                         Q0_available=group.Q0.count(),Q1_available=group.Q1.count(),
                         paired=len(paired),records=paired.episode_uuid.nunique(),
                         environments=paired.environment_id.nunique())
            if len(paired):
                summary,_ = aggregate(paired,['split','dataset'],['Q0','Q1'])
                stats.update(summary[['Q0','Q1']].iloc[0].to_dict())
            parts.append(stats)
    summary = pd.DataFrame(parts)
    summary.to_csv(DEST/'mask_summary.csv',index=False)
    print(summary.to_string(index=False))


if __name__ == '__main__':
    run()
