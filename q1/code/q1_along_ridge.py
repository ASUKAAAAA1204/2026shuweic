"""Evaluate an along-track ridge correction to fixed damping on Q1 training folds."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code/q1_head_to_head'))
from compare import Bank, EPS, HORIZONS, average, basic, design, features, load_windows, np, pd, scales
from data import hierarchy

ALPHA = .01  # Mean squared loss plus ALPHA times the squared slope norm.
OUT = ROOT / 'results/q1_along_ridge'
ARCHIVE = ROOT / 'results/q1_six_train'


def fit(windows, cache, ids, scale):
    if len(ids) < 5:
        return None
    inputs, labels = [], []
    for i in ids:
        window = windows[i]
        velocity, motion, _, _ = cache[i]
        points = np.linspace(1, len(window.ages) - 1, 6).round().astype(int)
        age = window.ages[points]
        residual = window.reference[points] - window.obs.x[-1] - basic(velocity, age)
        inputs.append(design(motion, age))
        labels.append((residual @ (velocity / motion[0])) / (scale * -np.expm1(-age / .2)))
    x, y = np.vstack(inputs), np.concatenate(labels)
    mean, spread = x.mean(axis=0), x.std(axis=0)
    spread[spread < 1e-12] = 1.
    z = (x - mean) / spread
    coef = np.linalg.solve(z.T @ z + len(x) * ALPHA * np.eye(z.shape[1]), z.T @ (y - y.mean()))
    return mean, spread, coef, y.mean()


def main():
    windows = load_windows()
    cache = [features(w.obs) for w in windows]
    meta = pd.read_csv(ARCHIVE / 'windows.csv', float_precision='round_trip')
    folds = pd.DataFrame(json.loads((ARCHIVE / 'folds.json').read_text(encoding='utf8')))
    scores = []
    np.testing.assert_array_equal(meta.window_id, [w.key['window_id'] for w in windows])
    with np.load(ARCHIVE / 'predictions.npz') as archive:
        offsets, times, truth, baseline = (archive[k] for k in ('offsets', 't', 'ref', 'B'))
    np.testing.assert_array_equal(times, np.concatenate([w.ages for w in windows]))
    np.testing.assert_array_equal(truth, np.vstack([w.reference for w in windows]))
    for fold in folds.itertuples():
        train = meta[meta.env != fold.env]
        scale_map, pooled = scales(train)
        scale = scale_map.get(fold.source, pooled)
        np.testing.assert_allclose(scale, fold.scale, rtol=1e-12, atol=0)
        ids = train[train.source == fold.source].sort_values(['record', 'side', 'start_s']).index
        bank = Bank(ids, windows, cache, fold.speed_q10)
        model = fit(windows, cache, bank.ids, scale)
        for row in meta[(meta.env == fold.env) & (meta.source == fold.source)].itertuples():
            i = row.Index
            window = windows[i]
            ages = window.ages
            velocity, motion, _, _ = cache[i]
            base = window.obs.x[-1] + basic(velocity, ages)
            np.testing.assert_array_equal(base, baseline[offsets[i]:offsets[i+1]])
            active = model is not None and bank.eligible(motion)
            pred = base
            if active:
                mean, spread, coef, intercept = model
                residual = (design(motion, ages) - mean) / spread @ coef + intercept
                correction = .5 * scale * -np.expm1(-ages / .2) * residual
                pred = base + correction[:, None] * (velocity / motion[0])
            assert np.isfinite(pred).all()
            np.testing.assert_array_equal(pred[0], window.obs.x[-1])
            for name, points in [('B', base), ('Ridge', pred)]:
                error = np.linalg.norm(points - window.reference, axis=1)
                for h in HORIZONS:
                    use = ages <= h + EPS
                    ade = average(ages[use], error[use])
                    scores.append((row.window_id, row.env, row.source, row.record, row.side,
                                   name, h, ade / scale, ade, error[use][-1], active))
    frame = pd.DataFrame(scores, columns=['window_id', 'env', 'source', 'record', 'side',
                                         'model', 'horizon', 'nade', 'ade', 'fde', 'active'])
    assert len(frame) == 2 * len(HORIZONS) * len(windows)
    assert not frame.duplicated(['window_id', 'model', 'horizon']).any()
    metrics = ['nade', 'ade', 'fde']
    horizons = hierarchy(frame, metrics, ['model'])
    reference = pd.read_csv(ROOT / 'results/q1_curves/horizons.csv').query('model == "B"')
    np.testing.assert_allclose(horizons.loc['B', 'nade'], reference.nade, atol=1e-12, rtol=0)
    overall = horizons.nade.groupby(level='model').mean()
    OUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT / 'window_scores.csv.gz', index=False)
    horizons.to_csv(OUT / 'horizons.csv')
    print(horizons.to_string())
    print(overall.to_string(), f'\nRidge gain: {1 - overall.Ridge / overall.B:.4%}')
    print('Active windows:', frame.loc[frame.active, 'window_id'].nunique(), '/', len(windows))


if __name__ == '__main__':
    main()
