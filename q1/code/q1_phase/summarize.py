"""Recompute archived scores, aggregate environment-paired effects and draw the figure."""

import json

from phase_experiment import OUT, ROOT, MODELS, SEED, EPS, np, pd, input_check

METRICS = ['nade', 'nfde']
PAIRS = (('H', 'B'), ('H', 'I'), ('H', 'P'), ('I', 'P'), ('P', 'B'), ('I', 'B'))


def verify(metrics):
    meta, windows, offsets, _, _ = input_check()
    maxima = dict(ade=0., fde=0., nade=0., nfde=0.)
    with np.load(OUT / 'predictions.npz', allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved['offsets'], offsets)
        for name in MODELS:
            xyz = saved[name]
            for i, rows in metrics[metrics.model.eq(name)].groupby('i', sort=True):
                w = windows[i]
                e = np.linalg.norm(xyz[offsets[i]:offsets[i + 1]] - w.reference, axis=1)
                for row in rows.itertuples():
                    h = row.horizon
                    end = np.searchsorted(w.ages, h + EPS, side='right')
                    t = w.ages[:end]
                    area = (np.diff(t) * (e[1:end] + e[:end - 1])).sum() / (2 * t[-1])
                    maxima['ade'] = max(maxima['ade'], abs(row.ade - area))
                    maxima['fde'] = max(maxima['fde'], abs(row.fde - e[end - 1]))
                    maxima['nade'] = max(maxima['nade'], abs(row.nade - area / meta.at[i, 'scale']))
                    maxima['nfde'] = max(maxima['nfde'], abs(row.nfde - e[end - 1] / meta.at[i, 'scale']))
    assert max(maxima.values()) < 1e-10
    return maxima


def aggregate(rows, level):
    keys = ['model', 'horizon', 'source', 'env', 'record', 'side']
    stats = rows.groupby(keys)[METRICS].mean()
    while keys[-1] != level:
        keys = keys[:-1]
        stats = stats.groupby(level=keys).mean()
    return stats


def paired(group, env_scores, average_scores):
    tab = env_scores['nade'].groupby(['source', 'env', 'model']).mean().unstack(['source', 'model'])
    sources = sorted(set(tab.columns.get_level_values('source')))
    arrays = [tab[source][list(MODELS)].to_numpy() for source in sources]
    np.testing.assert_allclose(np.mean([np.nanmean(a, axis=0) for a in arrays], axis=0),
                               average_scores.loc[list(MODELS)], atol=1e-12, rtol=0)
    rng = np.random.default_rng(SEED)
    samples, accepted, rejected = [], 0, 0
    while accepted < 5000:
        draws = rng.integers(len(tab), size=(5000 - accepted, len(tab)))
        valid = np.logical_and.reduce([np.isfinite(a[draws, 0]).any(axis=1) for a in arrays])
        samples.append(np.mean([np.nanmean(a[draws[valid]], axis=1) for a in arrays], axis=0))
        accepted += int(valid.sum())
        rejected += int((~valid).sum())
    boot = pd.DataFrame(np.vstack(samples), columns=MODELS)
    comparisons = []
    for candidate, reference in PAIRS:
        gain = 1 - average_scores[candidate] / average_scores[reference]
        lo, hi = np.quantile(1 - boot[candidate] / boot[reference], [.025, .975])
        comparisons.append(dict(group=group, candidate=candidate, reference=reference, gain=gain,
                                lo95=lo, hi95=hi, environments=len(tab), rejected=rejected,
                                advance=bool(group == 'all' and (candidate, reference) == ('H', 'B')
                                             and gain >= .03 and lo > 0)))
    return comparisons


def plot(comparisons):
    import os

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties

    matplotlib.rcParams.update({'font.size': 9, 'pdf.fonttype': 42, 'axes.spines.top': False,
                                'axes.spines.right': False})
    font_path = ROOT / os.environ.get('WINDIR', 'C:/Windows') / 'Fonts/msyh.ttc'
    chinese = font_path.exists()
    font = FontProperties(fname=font_path, size=9) if chinese else FontProperties(size=9)
    labels = ('全部窗口', '运动变化显著窗口') if chinese else ('All windows', 'High-change windows')
    xlabel = '归一化 ADE 相对改善率（%）' if chinese else 'Relative reduction in normalized ADE (%)'
    fig, ax = plt.subplots(figsize=(6.1, 2.85), layout='constrained')
    groups = [(-.14, 'all', '#285b84', labels[0]), (.14, 'high_change', '#b05035', labels[1])]
    for column, group, color, label in groups:
        rows = comparisons[comparisons.group.eq(group)].iloc[:3]
        center = 100 * rows.gain.to_numpy()
        low, high = 100 * rows.lo95.to_numpy(), 100 * rows.hi95.to_numpy()
        y = np.arange(len(rows)) + column
        if column < 0:
            pairs = [f'{a} vs {b}' for a, b in zip(rows.candidate, rows.reference)]
        ax.hlines(y, low, high, color=color, linewidth=1.4)
        ax.scatter(center, y, color=color, s=20, label=label, zorder=3)
    ax.axvline(0, color='#6c757d', linestyle='--', linewidth=.9)
    ax.set(yticks=np.arange(3), yticklabels=pairs, ylim=(2.55, -.56))
    ax.set_xlabel(xlabel, fontproperties=font)
    ax.legend(frameon=False, ncols=2, loc='lower left', bbox_to_anchor=(0, 1), prop=font)
    fig.savefig(ROOT / 'figures/q1_phase_ablation.pdf')
    plt.close(fig)


def main():
    metrics = pd.read_csv(OUT / 'metrics.csv.gz', float_precision='round_trip')
    manifest = pd.read_csv(OUT / 'assignments.csv')
    audit = json.loads((OUT / 'audit.json').read_text(encoding='utf-8'))
    audit['recomputed_from_saved_coordinates'] = verify(metrics)
    (OUT / 'audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    groups = {'all': metrics, 'high_change': metrics[metrics.high_change],
              'other': metrics[~metrics.high_change]}
    summaries, intervals, by_source, by_environment, coverage = [], [], [], [], []
    for group, rows in groups.items():
        env = aggregate(rows, 'env')
        source = env.groupby(level=['model', 'horizon', 'source']).mean()
        horizon = source.groupby(level=['model', 'horizon']).mean()
        headline = horizon.groupby(level='model').mean()
        summaries.extend([headline.assign(group=group, horizon='overall').reset_index(),
                          horizon.assign(group=group).reset_index()])
        intervals.extend(paired(group, env, headline.nade))
        by_source.append(source.assign(group=group).reset_index())
        by_environment.append(env.assign(group=group).reset_index())
        counts = rows.drop_duplicates('i').groupby('source').i.count()
        coverage.extend({'group': group, 'source': name, 'windows': count}
                        for name, count in counts.items())
    comparisons = pd.DataFrame(intervals)
    tables = dict(summary=pd.concat(summaries), paired=comparisons, coverage=pd.DataFrame(coverage),
                  sources=pd.concat(by_source), environments=pd.concat(by_environment))
    for name, table in tables.items():
        table.to_csv(OUT / f'{name}.csv', index=False)
    plot(comparisons)
    print(tables['summary'].query("horizon == 'overall'").to_string(index=False))
    print(comparisons.to_string(index=False))
    print(tables['coverage'].to_string(index=False))
    print(f'Fallback windows: {manifest.fallback.sum()}')


if __name__ == '__main__':
    main()
