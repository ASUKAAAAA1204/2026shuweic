"""Paired evidence tables, conditional uncertainty, figures and adoption decision."""
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
RES = BASE / "results"
METHODS = ("baseline", "B_K64", "C_repaired", "E_env_s40_fb3")
METRICS = ("weighted_on_time_rate", "weighted_completion_rate", "weighted_tardiness_s", "cost",
           "observations", "elapsed_s", "observed_overrun_s", "reserved_idle_s", "censored_units", "invalid_actions")


def main():
    out = RES / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    original = pd.read_csv(RES / "full_replay" / "metrics.csv")
    reviewed = pd.read_csv(RES / "reviewed_replay" / "metrics.csv")
    d = pd.concat([original[original.method.isin(["baseline", "B_K64", "E_env_s40_fb3"])],
                   reviewed[reviewed.method == "C_repaired"]], ignore_index=True)
    d.to_csv(out / "accepted_comparison_trials.csv", index=False)
    nom = d[(d.alpha == 1.) & (d.beta == .5)]
    for name, data, by in (("nominal_common_metrics", nom, ["method"]),
                            ("five_mapping_common_metrics", d, ["method"]),
                            ("case_mapping_common_metrics", d, ["method", "case", "alpha", "beta"]),
                            ("nominal_by_case", nom, ["method", "case"]),
                            ("by_mapping", d, ["method", "alpha", "beta"])):
        data.groupby(by)[list(METRICS)].mean().to_csv(out / (name + ".csv"))
    bootstrap = np.random.default_rng(20261122).integers(0, 30, size=(4000, 30))
    rows = []
    for scope, data in (("nominal", nom), ("five_mappings", d)):
        p = data.pivot(index=["seed", "case", "alpha", "beta"], columns="method", values="weighted_on_time_rate")
        means = p.groupby(level="seed").mean()
        assert len(means) == 30
        for a, b in itertools.combinations(METHODS, 2):
            # Difference is second method minus first, so baseline comparisons
            # all have the improvement sign and candidate pairs remain explicit.
            delta = (means[b] - means[a]).to_numpy() * 100.
            boot = delta[bootstrap].mean(axis=1)
            base = means[a].to_numpy()
            rel_boot = boot / base[bootstrap].mean(axis=1)
            rows.append({"scope": scope, "first": a, "second": b,
                         "second_minus_first_pp": float(delta.mean()),
                         "conditional_ci95_low_pp": float(np.quantile(boot, .025)),
                         "conditional_ci95_high_pp": float(np.quantile(boot, .975)),
                         "relative_percent": float(delta.mean() / base.mean()),
                         "conditional_relative_ci95_low_percent": float(np.quantile(rel_boot, .025)),
                         "conditional_relative_ci95_high_percent": float(np.quantile(rel_boot, .975)),
                         "positive_seeds": int((delta > 1e-10).sum()),
                         "zero_seeds": int((abs(delta) <= 1e-10).sum()), "seed_clusters": 30})
    pair = pd.DataFrame(rows)
    pair.to_csv(out / "paired_comparisons.csv", index=False)

    key = ["case", "seed", "alpha", "beta"]
    a = original[original.method == "E_env_s40_fb3"].set_index(key)
    b = reviewed[reviewed.method == "E_documented"].set_index(key)
    e_diff = float((a[list(METRICS)] - b[list(METRICS)]).abs().max().max())
    g = pd.read_csv(BASE.parent / "parallel" / "G_统合审计" / "results" / "trials.csv")
    merge = original.merge(g, on=key + ["method"], suffixes=("_new", "_old"), validate="one_to_one")
    regression_diff = float((merge.weighted_on_time_rate_new - merge.weighted_on_time_rate_old).abs().max())
    c_orig = original[(original.method == "C_mc_t4_s256_dt10") & (original.alpha == 1.) & (original.beta == .5)]
    c_fix = nom[nom.method == "C_repaired"]
    nested = pd.read_csv(RES / "nested_selection" / "outer_delta_pp.csv").set_index("fold")
    nominal = nom.groupby("method").weighted_on_time_rate.mean() * 100
    five = d.groupby("method").weighted_on_time_rate.mean() * 100
    # Preregistered rule in nested_selection/registration.json. Semantic failures
    # are excluded first; current fixed legal B/C/E are then compared by worst fold.
    eligible = []
    mapping = {"B_K64": "B_fixed_K64", "C_repaired": "C_fixed_repaired", "E_env_s40_fb3": "E_fixed"}
    for method, column in mapping.items():
        worst = float(nested[column].min())
        if nominal[method] > nominal["baseline"] and worst >= 0:
            eligible.append((worst, float(nominal[method]), method))
    winner = max(eligible)[2]
    assert winner == "E_env_s40_fb3"
    stats = {"episodes_original": len(original), "episodes_reviewed": len(reviewed),
             "G_regression_rows": len(merge), "G_regression_max_rate_difference": regression_diff,
             "E_documentation_equivalence_max_metric_difference": e_diff,
             "nominal_percent": nominal.to_dict(), "five_mapping_percent": five.to_dict(),
             "C_original_nominal_percent": float(c_orig.weighted_on_time_rate.mean() * 100),
             "C_repaired_nominal_percent": float(c_fix.weighted_on_time_rate.mean() * 100),
             "winner_under_frozen_rule": winner,
             "candidate_acceptance": {
                 "B_K64": "engineering_pass; robustness_gate_fail; retain_as_secondary",
                 "C_mc_t4_s256_dt10": "semantic_fail; original_score_not_adopted",
                 "C_repaired": "engineering_pass; fixed_configuration_outer_means_positive; inner_selection_unstable; secondary",
                 "E_env_s40_fb3": "engineering_pass; documentation_corrected_without_behavior_change; primary"},
             "uncertainty_scope": "4000 seed-cluster bootstrap replicates conditional on existing source pools; unadjusted for historical adaptive selection, not environment-generalization confidence"}
    assert e_diff == 0 and regression_diff == 0 and len(merge) == 1800
    (out / "summary.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(pair[pair.scope == "nominal"].to_string(index=False))
    plot(nominal, five, nested, out)


def plot(nominal, five, nested, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})
    methods = ["baseline", "B_K64", "C_repaired", "E_env_s40_fb3"]
    labels = ["Baseline", "B: K=64", "C: repaired", "E: env-balanced"]
    colors = ["#B6CAED", "#2A73BA", "#87AEE6", "#1F4E89"]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.25), constrained_layout=True)
    for ax, values, title in ((axes[0], nominal, "A  Nominal weighted on-time rate"),
                               (axes[1], five, "B  Equal mean across five mappings")):
        bars = ax.bar(np.arange(4), [values[m] for m in methods], color=colors,
                      edgecolor="#1F4E89", linewidth=.6)
        for bar, method in zip(bars, methods):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1, f"{values[method]:.2f}%",
                    ha="center", fontsize=8)
        ax.set_xticks(range(4), labels, rotation=30, ha="right")
        ax.set_ylim(0, 105); ax.set_ylabel("Weighted on-time rate (%)"); ax.set_title(title, loc="left", fontsize=9)
        ax.grid(axis="y", alpha=.15); ax.set_axisbelow(True)
    ax = axes[2]
    for i, (col, label, color, marker) in enumerate(zip(
            ("B_fixed_K64", "C_fixed_repaired", "E_fixed"), labels[1:], colors[1:], ("s", "^", "o"))):
        vals = nested[col].to_numpy()
        x = np.arange(2) + (i-1)*.12
        ax.plot(x, vals, color=color, marker=marker, linestyle=["--", ":", "-"][i], label=label)
        for fold, (xx, yy) in enumerate(zip(x, vals)):
            offset = ((-5, -14), (-15, 14), (19, 5))[i] if fold == 0 else (0, 7 if i != 1 else -13)
            ax.annotate(f"{yy:+.2f}", (xx, yy), xytext=offset,
                        textcoords="offset points", ha="center", fontsize=7)
    ax.axhline(0, color="#CE4844", lw=1)
    ax.set_xticks([0, 1], ["Outer fold 0", "Outer fold 1"])
    ax.set_xlim(-.35, 1.35); ax.set_ylim(-1.2, 5.5)
    ax.set_ylabel("Difference from baseline (percentage points)")
    ax.set_title("C  Environment-group transfer", loc="left", fontsize=9)
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    ax.grid(axis="y", alpha=.15); ax.set_axisbelow(True)
    fig.savefig(out / "candidate_acceptance.pdf", bbox_inches="tight")
    fig.savefig(out / "candidate_acceptance.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
