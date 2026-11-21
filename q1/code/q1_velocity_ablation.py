"""Fixed-velocity ablation for the Q1 damped extrapolation model.

The four estimators share the same frozen Q1 windows: exact six contiguous
eligible historical samples inside ``history_start`` and no gaps through the
one-second target. Predictions use Q1's fixed damped coefficient
``tau * (1 - exp(-age / tau))`` with ``tau = 0.2``; nothing is fitted.

A: difference of the last two historical samples.
B: ordinary least-squares slope over the last three historical samples.
C: ordinary least-squares slope over the six historical samples.
D: median of the pairwise slopes over the same six historical samples.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from q1 import BASE, DATA, KEYS, ROOT, TAU, aggregate

DEST = ROOT / 'results/q1_ablation'
ARCHIVE = ROOT / 'results/q1/motion_windows.csv'
CATALOG = BASE / 'data/windows.csv'
HORIZONS = (0.6, 1.0)
N_HISTORY = 6
MODELS = ('A', 'B', 'C', 'D')
ARCHIVE_MODEL = {'A': 'damped', 'D': 'robust'}
METHODS = {
    'A': 'last-two-sample difference',
    'B': 'OLS slope over the last three samples',
    'C': 'OLS slope over the six historical samples',
    'D': 'median of the pairwise slopes over the six historical samples',
}
EPS = 1e-12
VERIFY_TOL = 1e-10


def _catalog():
    """Frozen Q1 window catalog, restricted to the two reported splits."""
    catalog = pd.read_csv(CATALOG)
    catalog = catalog.loc[catalog.split.isin(('train', 'validation'))].copy()
    assert not catalog.window_id.duplicated().any()
    return catalog


def _ols_velocity(t, x):
    """Slope of an ordinary least-squares line through (t, x)."""
    centered = t - t.mean()
    denominator = centered @ centered
    assert denominator > 0
    return (centered[:, None] * (x - x.mean(axis=0))).sum(axis=0) / denominator


def _median_slope_velocity(t, x):
    """Per-coordinate median of all pairwise historical slopes."""
    a, b = np.triu_indices(len(t), 1)
    slopes = (x[b] - x[a]) / (t[b] - t[a])[:, None]
    return np.median(slopes, axis=0)


def _velocities(t, x):
    """The four velocity estimates for the common six-point history."""
    return {
        'A': (x[-1] - x[-2]) / (t[-1] - t[-2]),
        'B': _ols_velocity(t[-3:], x[-3:]),
        'C': _ols_velocity(t, x),
        'D': _median_slope_velocity(t, x),
    }


def _window_rows(catalog):
    """Score all models on the common exact-six historical window set."""
    rows = []
    skipped = {'short_history': 0, 'ineligible': 0}
    for uid, group in catalog.groupby('episode_uuid', sort=True):
        with np.load(DATA / 'records' / f'{uid}.npz', allow_pickle=False) as data:
            for side, subset in group.groupby('side', sort=True):
                t = data[f'{side}__t_s']
                pos = data[f'{side}__position'].mean(axis=1)
                valid = data[f'{side}__reference_candidate']
                gap = data[f'{side}__gap_before']
                assert np.all(np.diff(t) > 0)
                for w in subset.sort_values('start_index').itertuples(index=False):
                    i = int(w.start_index)
                    j = int(w.end_index_exclusive)
                    lo = int(w.history_start)
                    if i - lo < N_HISTORY:
                        skipped['short_history'] += 1
                        continue
                    start = i - N_HISTORY
                    assert start >= lo
                    # One shared six-point history, contiguous and eligible
                    # from history_start through the end of the target.
                    if not (
                        valid[start:j].all()
                        and not gap[start + 1:j].any()
                        and np.all(np.diff(t[start:j]) > 0)
                    ):
                        skipped['ineligible'] += 1
                        continue
                    age = t[i:j] - t[i - 1]
                    rel = t[i:j] - t[i]
                    delta = pos[i:j] - pos[i - 1]
                    weight = TAU * (-np.expm1(-age / TAU))
                    meta = {key: getattr(w, key) for key in KEYS}
                    for model, velocity in _velocities(t[start:i], pos[start:i]).items():
                        error = np.linalg.norm(weight[:, None] * velocity - delta, axis=1)
                        for horizon in HORIZONS:
                            take = rel <= horizon + EPS
                            assert take.any()
                            rows.append(
                                {
                                    **meta,
                                    'horizon_s': horizon,
                                    'model': model,
                                    'ade': float(error[take].mean()),
                                    'fde': float(error[take][-1]),
                                }
                            )
    return pd.DataFrame(rows), skipped


def _counts(frame):
    """Common sample sizes by split, source and horizon."""
    base = frame.loc[frame.model == 'A']
    counts = (
        base.groupby(['split', 'dataset', 'horizon_s'])
        .agg(
            windows=('window_id', 'nunique'),
            records=('episode_uuid', 'nunique'),
            environments=('environment_id', 'nunique'),
        )
        .reset_index()
    )
    counts['models'] = len(MODELS)
    counts['paired_rows'] = counts.windows * len(MODELS)
    return counts.sort_values(['split', 'dataset', 'horizon_s']).reset_index(drop=True)


def _summary_tables(frame):
    """Q1's three-level aggregate, plus paired environment detail."""
    summary, environments = aggregate(
        frame, ['split', 'dataset', 'model', 'horizon_s'], ['ade', 'fde']
    )
    summary = summary.sort_values(
        ['split', 'dataset', 'model', 'horizon_s']
    ).reset_index(drop=True)
    environments = environments.sort_values(
        ['split', 'dataset', 'model', 'horizon_s', 'environment_id']
    ).reset_index(drop=True)
    env_keys = ['split', 'dataset', 'model', 'horizon_s', 'environment_id']
    env_counts = (
        frame.groupby(env_keys)
        .agg(
            windows=('window_id', 'nunique'),
            records=('episode_uuid', 'nunique'),
        )
        .reset_index()
    )
    environments = environments.merge(
        env_counts, on=env_keys, how='left', validate='one_to_one'
    )
    baseline_keys = ['split', 'dataset', 'horizon_s', 'environment_id']
    baseline = environments.loc[
        environments.model == 'A', baseline_keys + ['ade', 'fde']
    ].rename(columns={'ade': 'ade_A', 'fde': 'fde_A'})
    environments = environments.merge(
        baseline, on=baseline_keys, how='left', validate='many_to_one'
    )
    environments['delta_ade'] = environments.ade - environments.ade_A
    environments['ratio_ade'] = environments.ade / environments.ade_A
    environments['delta_fde'] = environments.fde - environments.fde_A
    environments['ratio_fde'] = environments.fde / environments.fde_A
    environments = environments[
        [
            'split',
            'dataset',
            'model',
            'horizon_s',
            'environment_id',
            'ade',
            'fde',
            'ade_A',
            'delta_ade',
            'ratio_ade',
            'fde_A',
            'delta_fde',
            'ratio_fde',
            'windows',
            'records',
        ]
    ]
    return summary, environments


def _ratio_tables(summary):
    """Model-to-A ratios by source and horizon, then equal-weight folds."""
    keys = ['split', 'dataset', 'horizon_s']
    ade = summary.pivot(index=keys, columns='model', values='ade').reset_index()
    fde = summary.pivot(index=keys, columns='model', values='fde').reset_index()
    columns = list(MODELS)
    assert ade[columns].notna().all().all() and fde[columns].notna().all().all()
    wide = ade.merge(
        fde, on=keys, suffixes=('_ade', '_fde'), validate='one_to_one'
    )
    parts = []
    for model in MODELS:
        part = wide[keys].copy()
        part['model'] = model
        part['ratio_ade'] = wide[f'{model}_ade'] / wide['A_ade']
        part['delta_ade'] = wide[f'{model}_ade'] - wide['A_ade']
        part['ratio_fde'] = wide[f'{model}_fde'] / wide['A_fde']
        part['delta_fde'] = wide[f'{model}_fde'] - wide['A_fde']
        parts.append(part)
    detail = pd.concat(parts, ignore_index=True)
    detail = detail.sort_values(keys + ['model']).reset_index(drop=True)
    ratios = (
        detail.groupby(['split', 'model'])
        .agg(
            ratio_ade=('ratio_ade', 'mean'),
            ratio_fde=('ratio_fde', 'mean'),
            delta_ade=('delta_ade', 'mean'),
            delta_fde=('delta_fde', 'mean'),
            source_horizon_cells=('ratio_ade', 'size'),
        )
        .reset_index()
        .sort_values(['split', 'model'])
        .reset_index(drop=True)
    )
    return detail, ratios


def _paired_rows(frame):
    """One wide row per window and horizon, with all four models paired."""
    keys = KEYS + ['horizon_s']
    assert not frame.duplicated(keys + ['model']).any()
    ade = frame.pivot(index=keys, columns='model', values='ade').reset_index()
    fde = frame.pivot(index=keys, columns='model', values='fde').reset_index()
    paired = ade.merge(
        fde, on=keys, suffixes=('_ade', '_fde'), validate='one_to_one'
    )
    for model in MODELS[1:]:
        paired[f'delta_ade_{model}_vs_A'] = paired[f'{model}_ade'] - paired['A_ade']
        paired[f'ratio_ade_{model}_vs_A'] = paired[f'{model}_ade'] / paired['A_ade']
        paired[f'delta_fde_{model}_vs_A'] = paired[f'{model}_fde'] - paired['A_fde']
        paired[f'ratio_fde_{model}_vs_A'] = paired[f'{model}_fde'] / paired['A_fde']
    return paired.sort_values(keys).reset_index(drop=True)


def _verify_archive(frame):
    """Check A and D against the archived Q1 damped and robust rows."""
    assert ARCHIVE.exists(), f'archived Q1 rows not found: {ARCHIVE}'
    archived = pd.read_csv(ARCHIVE)
    required = set(KEYS + ['model', 'horizon_s', 'ade', 'fde'])
    missing = required.difference(archived.columns)
    assert not missing, f'archive is missing columns: {sorted(missing)}'
    archived = archived.loc[
        archived.model.isin(tuple(ARCHIVE_MODEL.values()))
        & archived.horizon_s.isin(HORIZONS)
        & archived.split.isin(('train', 'validation'))
    ].rename(columns={'model': 'archive_model'})
    ours = frame.loc[frame.model.isin(tuple(ARCHIVE_MODEL))].copy()
    ours['archive_model'] = ours.model.map(ARCHIVE_MODEL)
    keys = KEYS + ['archive_model', 'horizon_s']
    joined = ours.merge(
        archived[keys + ['ade', 'fde']],
        on=keys,
        how='inner',
        validate='one_to_one',
        suffixes=('', '_archive'),
    )
    assert len(joined) == len(ours), 'archived motion_windows is missing matching rows'
    assert joined.window_id.nunique() == frame.window_id.nunique()
    joined['ade_diff'] = joined.ade - joined.ade_archive
    joined['fde_diff'] = joined.fde - joined.fde_archive
    comparisons = {}
    for model, archived_model in ARCHIVE_MODEL.items():
        block = joined.loc[joined.model == model]
        comparisons[archived_model] = {
            'ours': model,
            'matched_rows': int(len(block)),
            'matched_windows': int(block.window_id.nunique()),
            'max_abs_ade_diff': float(block.ade_diff.abs().max()),
            'max_abs_fde_diff': float(block.fde_diff.abs().max()),
        }
    max_ade = max(item['max_abs_ade_diff'] for item in comparisons.values())
    max_fde = max(item['max_abs_fde_diff'] for item in comparisons.values())
    assert max_ade <= VERIFY_TOL and max_fde <= VERIFY_TOL, (
        f'archive verification failed: ade={max_ade}, fde={max_fde}'
    )
    by_group = (
        joined.groupby(['split', 'dataset', 'horizon_s', 'model'])
        .agg(
            rows=('window_id', 'size'),
            windows=('window_id', 'nunique'),
            max_abs_ade_diff=('ade_diff', lambda values: float(values.abs().max())),
            max_abs_fde_diff=('fde_diff', lambda values: float(values.abs().max())),
        )
        .reset_index()
    )
    return {
        'archive': str(ARCHIVE.relative_to(ROOT)),
        'matched_rows': int(len(joined)),
        'matched_windows': int(joined.window_id.nunique()),
        'comparisons': comparisons,
        'max_abs_ade_diff': max_ade,
        'max_abs_fde_diff': max_fde,
        'tolerance': VERIFY_TOL,
        'verified': True,
        'by_group': json.loads(by_group.to_json(orient='records')),
    }


def _metadata(catalog, frame, skipped, verification):
    base = frame.loc[frame.model == 'A']
    return {
        'tau_s': TAU,
        'horizons_s': list(HORIZONS),
        'history_points': N_HISTORY,
        'models': METHODS,
        'fixed_no_fitting': True,
        'splits_used': sorted(frame.split.unique().tolist()),
        'catalog': str(CATALOG.relative_to(ROOT)),
        'catalog_windows': int(len(catalog)),
        'catalog_windows_by_split': {
            str(key): int(value)
            for key, value in catalog.groupby('split').window_id.nunique().items()
        },
        'eligible_windows': int(base.window_id.nunique()),
        'eligible_windows_by_split': {
            str(key): int(value)
            for key, value in base.groupby('split').window_id.nunique().items()
        },
        'eligible_windows_by_dataset': {
            str(key): int(value)
            for key, value in base.groupby('dataset').window_id.nunique().items()
        },
        'skipped_windows': skipped,
        'window_horizon_cells': int(len(base)),
        'paired_rows': int(len(frame)),
        'verification': verification,
    }


def run():
    """Run the fixed ablation and write the requested artifacts."""
    catalog = _catalog()
    frame, skipped = _window_rows(catalog)
    assert not frame.empty, 'no window has six eligible historical samples'
    keys = KEYS + ['horizon_s']
    assert not frame.duplicated(keys + ['model']).any()
    assert frame.groupby(keys).model.nunique().eq(len(MODELS)).all()
    assert set(frame.split) <= {'train', 'validation'}

    verification = _verify_archive(frame)
    summary, environments = _summary_tables(frame)
    counts = _counts(frame)
    ratios, aggregate_ratios = _ratio_tables(summary)
    paired = _paired_rows(frame)
    metadata = _metadata(catalog, frame, skipped, verification)
    metadata['aggregate_ratios'] = {
        model: {
            split: float(value)
            for split, value in group.set_index('split').ratio_ade.items()
        }
        for model, group in aggregate_ratios.groupby('model')
    }

    DEST.mkdir(parents=True, exist_ok=True)
    frame.sort_values(keys + ['model']).to_csv(
        DEST / 'velocity_windows.csv', index=False
    )
    paired.to_csv(DEST / 'velocity_paired.csv', index=False)
    summary.to_csv(DEST / 'velocity_summary.csv', index=False)
    environments.to_csv(DEST / 'velocity_environments.csv', index=False)
    counts.to_csv(DEST / 'velocity_counts.csv', index=False)
    ratios.to_csv(DEST / 'velocity_ratios.csv', index=False)
    aggregate_ratios.to_csv(DEST / 'velocity_aggregate_ratios.csv', index=False)
    (DEST / 'velocity_checks.json').write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + '\n', encoding='utf8'
    )
    print(
        f'eligible windows: {metadata["eligible_windows"]}; '
        f'paired rows: {metadata["paired_rows"]}'
    )
    print(aggregate_ratios.to_string(index=False))


if __name__ == '__main__':
    run()
