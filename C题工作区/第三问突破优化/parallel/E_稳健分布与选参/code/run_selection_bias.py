"""Quantify validation selection bias *post hoc* (never used for selection).

All candidate configurations are replayed on the official validation grid
(nominal mapping, 30 seeds) after the selection was frozen.  The winner's-curse
estimate is the split-half optimism of validation-based selection:

    pick i* = argmax_i S_i(half A);  optimism = S_i*(A) - S_i*(B)

with a seed-cluster bootstrap that repeats the pick inside every resample.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import common  # noqa: E402
import robust_chain_policy as rcp  # noqa: E402
from configs import CONFIGS  # noqa: E402

OUT = HERE.parent / "results" / "selection_bias"
NOMINAL_PP = 100.0


def main():
    common.add_v2_path()
    cond, tasks, zones, train, seg = common.load_all()
    OUT.mkdir(parents=True, exist_ok=True)
    pools = common.pools_from(common.train_rows(seg))
    med = {f: float(np.median(v)) for f, v in pools.items()}
    records = common.records_from(common.train_rows(seg))
    priors = {n: rcp.build_priors(records, strength=c["strength"], balance=c["balance"],
                                  winsor_q=c.get("winsor_q"))
              for n, c in CONFIGS.items()}
    vpool = seg[(seg.split == "validation") & (seg.eligible == 1)]
    seeds = cond["validation_seeds"]
    jobs_by_case = {c["case_id"]: common.jobs_for(tasks, c["case_id"]) for c in cond["cases"]}

    rows = []
    max_diff = 0.0
    for i, (name, cfg) in enumerate(CONFIGS.items()):
        def ctx_for(_method, prior=priors[name], cfg=cfg):
            return dict(pools=pools, med=med, prior=prior, params=cfg)

        sub, diff = common.run_grid(cond["cases"], jobs_by_case, seeds, [common.NOMINAL],
                                    zones, vpool, ["role_e"], ctx_for)
        for r in sub:
            r["config"] = name
        rows.extend(sub)
        max_diff = max(max_diff, diff)
        print(i + 1, "/", len(CONFIGS), name, flush=True)

    d = pd.DataFrame(rows)
    d.to_csv(OUT / "trials.csv", index=False)
    per_seed = d.pivot_table(index=["config", "seed"], columns="case",
                             values="weighted_on_time_rate", aggfunc="mean")
    score = per_seed.mean(axis=1).unstack("config")  # seeds x configs
    names = list(score.columns)
    S = {n: float(score[n].mean()) for n in names}
    selected = json.loads((HERE.parent / "results" / "selection_decision.json")
                          .read_text(encoding="utf8"))["selected_config"]

    seed_ids = list(score.index)
    half_a = [s for i, s in enumerate(seed_ids) if i % 2 == 0]
    half_b = [s for i, s in enumerate(seed_ids) if i % 2 == 1]

    def pick_optimism(a, b):
        sa = score.loc[a].mean()
        sb = score.loc[b].mean()
        i_star = sa.idxmax()
        return i_star, float(sa[i_star] - sb[i_star]) * NOMINAL_PP

    star_a, opt_a = pick_optimism(half_a, half_b)
    star_b, opt_b = pick_optimism(half_b, half_a)
    rng = np.random.default_rng(20261122)
    boots = []
    for _ in range(2000):
        draw = score.loc[rng.choice(seed_ids, len(seed_ids), replace=True)].reset_index(drop=True)
        cut = len(draw) // 2
        sa = draw.iloc[:cut].mean()
        sb = draw.iloc[cut:].mean()
        i_star = sa.idxmax()
        boots.append(float(sa[i_star] - sb[i_star]) * NOMINAL_PP)

    ranked = sorted(S.items(), key=lambda kv: -kv[1])
    summary = {
        "note": "post-hoc only; selection was frozen in results/selection_decision.json "
                "before this run",
        "max_independent_metric_diff": max_diff,
        "validation_score_by_config": {n: float(v) for n, v in ranked},
        "cv_selected_config": selected,
        "cv_selected_validation_rank": [n for n, _ in ranked].index(selected) + 1,
        "validation_best_config": ranked[0][0],
        "validation_best_minus_cv_selected_pp": float(ranked[0][1] - S[selected]) * NOMINAL_PP,
        "split_half": {
            "pick_on_half_A": star_a, "optimism_A_pp": opt_a,
            "pick_on_half_B": star_b, "optimism_B_pp": opt_b,
            "mean_optimism_pp": (opt_a + opt_b) / 2,
            "bootstrap_ci95_pp": [float(np.percentile(boots, 2.5)),
                                  float(np.percentile(boots, 97.5))],
            "bootstrap_mean_pp": float(np.mean(boots)),
            "half_sizes": [len(half_a), len(half_b)],
        },
        "spread_all_candidates_pp": float((ranked[0][1] - ranked[-1][1]) * NOMINAL_PP),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                                      encoding="utf8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
