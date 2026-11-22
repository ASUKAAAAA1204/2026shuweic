"""Aggregate role-F trials into results/trials.csv and results/summary.json.

All numbers are computed from this unit's own CSV files (trials_S*.csv from
run_roleF.py, legal_extra.csv from run_legal_extra.py, holdout_trials.csv from
holdout_roleF.py).  No number is copied from another work unit.

Roles of the methods
--------------------
* pressure_gated      legal online baseline (formal deadline_policy/baseline)
* v2_s20              legal online reference (role-A multi-step, depth=1, s=20)
* lookahead_s40       legal online reference = official confirm_lookahead config
* oracle              perfect-information exact optimum (upper bound, NOT legal)
* ub_relax_knapsack   provable relaxation upper bound (necessary conditions only)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "results"
CASES = ["S1", "S2", "S3"]
MAPPINGS = [(1.0, 0.5), (0.75, 0.5), (1.5, 0.5), (1.0, 0.0), (1.0, 1.0)]
METHODS = ["pressure_gated", "v2_s20", "lookahead_s40", "oracle", "ub_relax_knapsack"]
LEGAL = ["pressure_gated", "v2_s20", "lookahead_s40"]
COLS = ["case", "seed", "alpha", "beta", "method", "weighted_on_time_rate",
        "weighted_completion_rate", "cost", "observations"]
TARGET = 0.90198  # 6% relative over the 85.0926% pressure_gated nominal

_PRED = None


def _predecessor_ids():
    """Job ids that appear as a predecessor anywhere (i.e. sit on a chain)."""
    global _PRED
    if _PRED is None:
        import sys
        formal = HERE.parents[3] / "第三问改进正式方案" / "code"
        sys.path.insert(0, str(formal))
        from support import load, jobs_for
        _, tasks_df, _, _ = load()
        ids = set()
        for c in CASES:
            for j in jobs_for(tasks_df, c):
                ids.update(j["predecessors"])
        _PRED = ids
    return _PRED


def load_trials():
    parts = [pd.read_csv(OUT / ("trials_%s.csv" % c)) for c in CASES]
    extra = OUT / "legal_extra.csv"
    if extra.exists():
        parts.append(pd.read_csv(extra))
    d = pd.concat(parts, ignore_index=True)
    d["key"] = list(zip(d.alpha.round(6), d.beta.round(6)))
    return d


def boot_ci(values, n_boot=4000, seed=20261122):
    v = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), (n_boot, len(v)))
    means = v[idx].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def paired(d, method, base="pressure_gated", mapping=(1.0, 0.5)):
    """Per-seed avg3 paired delta (percentage points) at one mapping."""
    sub = d[d.key == mapping]
    b = sub[sub.method == base].set_index(["case", "seed"]).weighted_on_time_rate
    m = sub[sub.method == method].set_index(["case", "seed"]).weighted_on_time_rate
    return (m - b).groupby(level="seed").mean() * 100.0


def mapping_stats(d, method):
    bymap = {}
    for a, b in MAPPINGS:
        x = d[(d.method == method) & (d.alpha == a) & (d.beta == b)].weighted_on_time_rate
        bymap["%g_%g" % (a, b)] = float(x.mean() * 100.0)
    return bymap, float(np.mean(list(bymap.values())))


def main():
    d = load_trials()
    nominal = d[d.key == (1.0, 0.5)]
    present = set(d.method.unique())

    methods = {}
    for m in METHODS:
        if m not in present:
            continue
        cur = nominal[nominal.method == m]
        by_case = {c: float(cur[cur.case == c].weighted_on_time_rate.mean() * 100.0)
                   for c in CASES}
        bymap, mapavg = mapping_stats(d, m)
        methods[m] = dict(
            nominal_avg3=float(np.mean(list(by_case.values()))),
            nominal_by_case=by_case,
            mapping_avg5=mapavg,
            by_mapping=bymap,
            mean_completion_rate=None if m == "ub_relax_knapsack"
            else float(cur.weighted_completion_rate.mean()),
            mean_cost=None if m == "ub_relax_knapsack" else float(cur.cost.mean()),
        )

    # paired deltas vs pressure_gated at the nominal mapping
    paired_out = {}
    for m in ("v2_s20", "lookahead_s40", "oracle"):
        if m not in present:
            continue
        dl = paired(d, m)
        paired_out[m] = dict(
            mean_pp=float(dl.mean()),
            boot95=boot_ci(dl.values),
            positive_seeds=int((dl > 0).sum()),
            nonzero_seeds=int((dl.abs() > 1e-12).sum()),
            negative_seeds=int((dl < 0).sum()),
            n_seeds=int(len(dl)),
        )
    # five-mapping paired mean and per-mapping means (each vs baseline)
    five = {}
    for m in ("v2_s20", "lookahead_s40", "oracle"):
        if m not in present:
            continue
        per = [float(paired(d, m, mapping=mp).mean()) for mp in MAPPINGS]
        all_ge = all(x >= -1e-9 for x in per)
        five[m] = dict(per_mapping_pp=per, mean_pp=float(np.mean(per)),
                       all_mappings_ge_baseline=all_ge)

    # independent-metric audit
    diffs = {}
    for c in CASES:
        f = OUT / ("diag_%s.json" % c)
        if f.exists():
            diffs[c] = json.loads(f.read_text()).get("max_independent_diff")
    le = OUT / "legal_extra_diag.json"
    if le.exists():
        diffs["legal_extra"] = json.loads(le.read_text()).get("max_independent_diff")
    max_diff = max([v for v in diffs.values() if v is not None] or [0.0])

    # ---- gap decomposition at the nominal mapping -------------------------
    jobs = pd.concat([pd.read_csv(OUT / ("jobs_%s.csv" % c)) for c in CASES],
                     ignore_index=True)
    jl = OUT / "jobs_legal_extra.csv"
    if jl.exists():
        jobs = pd.concat([jobs, pd.read_csv(jl)], ignore_index=True)
    jobs["key"] = list(zip(jobs.alpha.round(6), jobs.beta.round(6)))
    nj = jobs[jobs.key == (1.0, 0.5)].copy()
    nj["is_chain"] = nj.job_id.isin(_predecessor_ids())
    tot = nj[nj.method == "oracle"].groupby(["case", "seed"]).weight.sum()

    def gap_table(ref):
        ref_on = nj[nj.method == ref].set_index(["case", "seed", "job_id"]).on_time
        orc = nj[nj.method == "oracle"].set_index(["case", "seed", "job_id"])
        if len(ref_on) == 0:
            return None
        orc = orc.join(ref_on.rename("ref_on"), how="left")
        ref_on = orc["ref_on"].fillna(False).astype(bool)
        missing = orc[(orc.on_time) & (~ref_on)].copy()
        extra = orc[(~orc.on_time) & (ref_on)]      # ref-only on-time
        out = {"oracle_only_weight": float(missing.weight.sum()),
               "ref_only_weight": float(extra.weight.sum()),
               "total_weight": float(tot.sum())}
        miss_total = float(missing.weight.sum())
        for dim in ("family", "zone", "is_chain"):
            g = missing.groupby(dim).weight.sum()
            out[dim] = {str(k): float(v) for k, v in g.items()}
            out[dim + "_share"] = {str(k): (float(v) / miss_total if miss_total else 0.0)
                                   for k, v in g.items()}
        missing["lat_bucket"] = pd.cut(missing.latitude, [-1e-9, 20, 40, 60, 80, 1e9],
                                       labels=["<=20", "20-40", "40-60", "60-80", ">80"])
        g = missing.groupby("lat_bucket", observed=False).weight.sum()
        out["latitude"] = {str(k): float(v) for k, v in g.items()}
        out["latitude_share"] = {str(k): (float(v) / miss_total if miss_total else 0.0)
                                 for k, v in g.items()}
        # avg3-consistent gap: mean over (case, seed) of the rate difference
        mws = missing.groupby(["case", "seed"]).weight.sum().reindex(tot.index, fill_value=0.0)
        ews = extra.groupby(["case", "seed"]).weight.sum().reindex(tot.index, fill_value=0.0)
        out["net_missing_avg3_pp"] = float(((mws - ews) / tot).mean() * 100.0)
        return out

    gap = {}
    for ref in ("pressure_gated", "v2_s20", "lookahead_s40"):
        if ref in present:
            gap["oracle_vs_%s" % ref] = gap_table(ref)

    # per-case headroom
    headroom = {}
    for c in CASES:
        row = dict(target=TARGET * 100.0)
        for m in LEGAL + ["oracle", "ub_relax_knapsack"]:
            if m in methods:
                row[m] = methods[m]["nominal_by_case"][c]
        headroom[c] = row

    # ---- feasibility verdict ---------------------------------------------
    legal_best = max((m for m in LEGAL if m in methods),
                     key=lambda m: methods[m]["nominal_avg3"])
    feas = dict(
        target_nominal_pct=TARGET * 100.0,
        oracle_nominal_avg3=methods["oracle"]["nominal_avg3"],
        oracle_minus_target_pp=methods["oracle"]["nominal_avg3"] - TARGET * 100.0,
        relaxation_ub_nominal_avg3=methods["ub_relax_knapsack"]["nominal_avg3"],
        ub_minus_oracle_pp=methods["ub_relax_knapsack"]["nominal_avg3"]
                           - methods["oracle"]["nominal_avg3"],
        best_legal_method=legal_best,
        best_legal_nominal_avg3=methods[legal_best]["nominal_avg3"],
        best_legal_reaches_target=methods[legal_best]["nominal_avg3"] >= TARGET * 100.0,
        best_legal_paired_ci95=paired_out[legal_best]["boot95"],
        best_legal_five_mapping_avg_pp=five[legal_best]["mean_pp"],
        best_legal_all_mappings_ge_baseline=five[legal_best]["all_mappings_ge_baseline"],
        online_headroom_pp=methods["oracle"]["nominal_avg3"]
                          - methods[legal_best]["nominal_avg3"],
        oracle_reaches_target=methods["oracle"]["nominal_avg3"] >= TARGET * 100.0,
    )

    holdout = None
    hf = OUT / "holdout_summary.json"
    if hf.exists():
        holdout = json.loads(hf.read_text())
    ablation = None
    af = OUT / "ablation.json"
    if af.exists():
        ablation = json.loads(af.read_text())

    summary = dict(
        protocol=dict(cases=CASES, seeds=int(nominal.seed.nunique()),
                      mappings=[list(m) for m in MAPPINGS],
                      target_nominal_pct=TARGET * 100.0,
                      sampling="confirm_lookahead.hidden_for (validation & eligible)"),
        methods=methods,
        paired_vs_pressure_gated_nominal=paired_out,
        paired_five_mapping=five,
        independent_metric_max_diff=max_diff,
        independent_metric_by_source=diffs,
        gap_decomposition=gap,
        per_case_headroom=headroom,
        feasibility=feas,
        ablation=ablation,
        holdout=holdout,
        selection_bias=dict(
            parameters_selected_by_this_unit=False,
            note=("No parameter was tuned in this unit. lookahead_s40="
                  "(strength=40,feedback_strength=3,quantiles=5) and v2_s20="
                  "(depth=1,strength=20,feedback_strength=3,nq_root=5,nq=2,beam=3) "
                  "were fixed by other work units and tuned on the same validation "
                  "split, so their point estimates inherit that selection bias. "
                  "oracle and ub_relax_knapsack read hidden durations / only "
                  "necessary conditions and are not affected by policy selection; "
                  "ub_relax_knapsack rows in trials.csv have blank completion/cost/"
                  "observations because they are not simulated."),
        ),
    )
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                                      encoding="utf8")

    full = d[COLS]
    full.to_csv(OUT / "trials.csv", index=False)
    print(json.dumps(feas, indent=2, ensure_ascii=False))
    print("nominal_avg3:", {m: round(methods[m]["nominal_avg3"], 4) for m in methods})
    print("wrote", OUT / "summary.json", "and", OUT / "trials.csv")


if __name__ == "__main__":
    main()
