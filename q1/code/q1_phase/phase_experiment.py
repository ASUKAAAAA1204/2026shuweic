"""Evaluate causal phase-dependent damping on the frozen first-question training windows."""

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code/q1_curves'))
from experiment import HORIZONS, EPS, input_check, integral_weights, np, pd, scales, threadpool_limits
from data import Obs, Window, average, basic
from scipy.optimize import least_squares
from hmm.model import MotionHMM

OUT = ROOT / 'results/q1_phase'
MODELS = ('B', 'P', 'I', 'H')
SEED = 20261120
TAU = .2


def motion(obs, scale):
    velocities = np.diff(obs.x, axis=0) / np.diff(obs.t)[:, None]
    speeds = np.linalg.norm(velocities, axis=1)
    norms = speeds[:-1] * speeds[1:]
    products = np.einsum('ij,ij->i', velocities[:-1], velocities[1:])
    cosines = np.divide(products, norms, out=np.ones_like(norms), where=norms > 0)
    sequence = np.column_stack((np.log1p(speeds[1:] / scale), np.clip(cosines, -1, 1)))
    return velocities[-1], sequence


def damping(ages, taus):
    return -np.expm1(-ages[:, None] / taus) * taus


def fit_taus(windows, posterior, scale, prior, penalty):
    ages = np.concatenate([w.ages for w in windows])
    weights = np.concatenate([integral_weights(w) for w in windows])
    weights = np.sqrt(weights / weights.sum())
    lengths = [len(w.ages) for w in windows]
    velocity = np.stack([motion(w.obs, scale)[0] for w in windows])
    speed = np.linalg.norm(velocity, axis=1)
    directions = np.divide(velocity, speed[:, None], out=np.zeros_like(velocity),
                           where=speed[:, None] > 0)
    targets = np.concatenate([(w.reference - w.obs.x[-1]) @ d
                              for w, d in zip(windows, directions)]) / scale
    mixed_speed = np.repeat(posterior * (speed / scale)[:, None], lengths, axis=0)
    ridge = np.sqrt(penalty / posterior.shape[1])

    def residual(taus):
        along = np.sum(mixed_speed * damping(ages, taus), axis=1)
        return np.r_[weights * (along - targets), ridge * (taus - prior)]

    def jacobian(taus):
        h_over_tau = ages[:, None] / taus
        gradient = -np.expm1(-h_over_tau) - h_over_tau * np.exp(-h_over_tau)
        return np.vstack((weights[:, None] * mixed_speed * gradient, ridge * np.eye(len(taus))))

    result = least_squares(residual, np.full(posterior.shape[1], prior), jac=jacobian,
                           bounds=(.05, .80), ftol=1e-10, xtol=1e-10, gtol=1e-10)
    assert result.success and np.isfinite(result.x).all(), result.message
    return result.x


def fit_block(windows, scale):
    sequences = [motion(w.obs, scale)[1] for w in windows]
    common = fit_taus(windows, np.ones((len(windows), 1)), scale, TAU, 0.)
    taus = dict(P=common, I=common, H=common)
    eligible = [i for i, sequence in enumerate(sequences) if len(sequence)]
    if sum(len(sequence) >= 2 for sequence in sequences) < 10:
        return None, taus
    usable = [sequences[i] for i in eligible]
    selected = [windows[i] for i in eligible]
    model = MotionHMM.fit(usable, seed=SEED)
    for name, estimate in [('I', model.independent), ('H', model.filtered)]:
        posterior = np.stack([estimate(s) for s in usable])
        np.testing.assert_allclose(posterior.sum(axis=1), 1, atol=1e-12, rtol=0)
        taus[name] = fit_taus(selected, posterior, scale, common[0], .005)
    return model, taus


def forecast(obs, ages, scale, model, taus):
    velocity, sequence = motion(obs, scale)
    anchor = obs.x[-1]
    baseline = anchor + basic(velocity, ages)
    pooled = anchor + damping(ages, taus['P']) * velocity
    predictions = dict(B=baseline, P=pooled, I=pooled, H=pooled)
    posterior = {}
    if model is not None and len(sequence):
        for name, estimate in [('I', model.independent), ('H', model.filtered)]:
            probabilities = estimate(sequence)
            posterior[name] = probabilities
            predictions[name] = anchor + (damping(ages, taus[name]) @ probabilities)[:, None] * velocity
    return predictions, posterior


def speed_change(window):
    # This statistic reads the reference only for retrospective evaluation.
    span = .1
    obs = window.obs
    assert obs.t[-1] - obs.t[0] >= span and window.ages[-1] >= span
    before = [np.interp(-span, obs.t - obs.t[-1], track) for track in obs.x.T]
    after = [np.interp(span, window.ages, track) for track in window.reference.T]
    return np.linalg.norm(np.array(after) + before - 2 * obs.x[-1]) / span


def run():
    OUT.mkdir(exist_ok=True, parents=True)
    meta, windows, offsets, archived, hashes = input_check()
    predictions = {name: np.full_like(archived['B'], np.nan) for name in MODELS}
    change = np.array([speed_change(w) for w in windows])
    parameters, metrics, assignments = [], [], []
    baseline_max, audit_max, replay_count = 0., 0., 0
    for env in sorted(meta.env.unique()):
        train, held = meta[meta.env.ne(env)], meta[meta.env.eq(env)]
        scale_by_source, _ = scales(train)
        for source, block in held.groupby('source'):
            train_ids = train.index[train.source.eq(source)]
            assert len(train_ids) and not set(train_ids) & set(held.index)
            scale = scale_by_source[source]
            np.testing.assert_allclose(block.scale, scale, atol=1e-14, rtol=0)
            model, taus = fit_block([windows[i] for i in train_ids], scale)
            threshold = np.quantile(change[train_ids], .75)
            parameters.append(dict(env=env, source=source, training_windows=len(train_ids),
                                   training_envs=sorted(train.loc[train_ids].env.unique()),
                                   scale=scale, change_threshold=threshold,
                                   tau={k: v.tolist() for k, v in taus.items()},
                                   hmm=model.parameters() if model else None))
            for i in block.index:
                w = windows[i]
                sl = slice(offsets[i], offsets[i+1])
                pred, state = forecast(w.obs, w.ages, scale, model, taus)
                altered = Window({}, Obs(w.obs.t.copy(), w.obs.x.copy()), w.ages.copy(),
                                 np.full_like(w.reference, 999.))
                replay, _ = forecast(altered.obs, altered.ages, scale, model, taus)
                high = bool(change[i] >= threshold)
                assignments.append(dict(i=i, window_id=w.key['window_id'], env=env, source=source,
                                        high_change=high, speed_change=change[i],
                                        p_I=state.get('I', np.full(3, np.nan)).tolist(),
                                        p_H=state.get('H', np.full(3, np.nan)).tolist(),
                                        fallback=not state))
                baseline_max = max(baseline_max, np.abs(pred['B'] - archived['B'][sl]).max())
                for name, estimate in pred.items():
                    np.testing.assert_array_equal(estimate, replay[name])
                    np.testing.assert_array_equal(estimate[0], w.obs.x[-1])
                    replay_count += 1
                    assert np.isfinite(estimate).all()
                    predictions[name][sl] = estimate
                    errors = np.linalg.norm(estimate - w.reference, axis=1)
                    for horizon in HORIZONS:
                        end = np.searchsorted(w.ages, horizon + EPS, side='right')
                        t, e = w.ages[:end], errors[:end]
                        ade = average(t, e)
                        audit = np.dot(np.diff(t), (e[1:] + e[:-1]) * .5) / (t[-1] - t[0])
                        audit_max = max(audit_max, abs(ade - audit))
                        metrics.append(dict(i=i, **{k: w.key[k] for k in
                                                    ('window_id', 'source', 'env', 'record', 'side')},
                                            model=name, horizon=horizon, high_change=high,
                                            ade=ade, nade=ade / scale, fde=e[-1], nfde=e[-1] / scale))
        print(f'Completed {env}', flush=True)
    frame = pd.DataFrame(metrics)
    assert baseline_max < 1e-10 and audit_max < 1e-10
    assert len(frame) == len(windows) * len(MODELS) * len(HORIZONS)
    assert not frame.duplicated(['i', 'model', 'horizon']).any()
    assert all(np.isfinite(p).all() for p in predictions.values())
    frame.to_csv(OUT / 'metrics.csv.gz', index=False)
    pd.DataFrame(assignments).to_csv(OUT / 'assignments.csv', index=False)
    (OUT / 'fitted.json').write_text(json.dumps(parameters, indent=2, allow_nan=False), encoding='utf-8')
    np.savez_compressed(OUT / 'predictions.npz', offsets=offsets, **predictions)
    protocol = ROOT / 'reports/第一问_阶段阻尼对照设计.md'
    audit = dict(windows=len(windows), records=int(meta.record.nunique()),
                 environments=int(meta.env.nunique()), splits=['train'], rows=len(frame),
                 baseline_max_abs=baseline_max, independent_integration_max_abs=audit_max,
                 reference_perturbation_replays=replay_count, archive_sha256=hashes,
                 protocol_sha256=hashlib.sha256(protocol.read_bytes()).hexdigest())
    (OUT / 'audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        run()
