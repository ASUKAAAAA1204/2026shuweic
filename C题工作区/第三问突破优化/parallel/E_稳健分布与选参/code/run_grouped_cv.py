"""Environment-grouped CV for role E parameter selection.

Only the two training-environment groups from
`results/grouped_training/registration.json` are used.  For each fold the
duration belief is fitted on the *other* group; the held group supplies the
durations.  Validation split is never touched here.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import common  # noqa: E402
import robust_chain_policy as rcp  # noqa: E402
from configs import CONFIGS  # noqa: E402

OUT = HERE.parent / "results" / "grouped_cv"
GROUPS = json.loads((HERE.parents[2] / "results" / "grouped_training" /
                     "registration.json").read_text())["groups"]
SEED0 = 2026112300


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--configs", default="full")
    args = ap.parse_args()
    names = list(CONFIGS) if args.configs == "full" else args.configs.split(",")

    common.add_v2_path()
    cond, tasks, zones, train, seg = common.load_all()
    OUT.mkdir(parents=True, exist_ok=True)
    jobs_by_case = {c["case_id"]: common.jobs_for(tasks, c["case_id"]) for c in cond["cases"]}
    rows = []
    max_diff = 0.0
    t0 = time.time()
    for fold, held in enumerate(GROUPS):
        fit = common.train_rows(seg, environments=set(seg.environment_id.unique()) - set(held))
        evp = common.train_rows(seg, environments=set(held))
        records = common.records_from(fit)
        pools = common.pools_from(fit)
        med = {f: float(np.median(v)) for f, v in pools.items()}
        priors = {n: rcp.build_priors(records, strength=c["strength"], balance=c["balance"],
                                     winsor_q=c.get("winsor_q"))
                  for n, c in CONFIGS.items() if n in names}
        for seed in range(SEED0, SEED0 + args.seeds):
            for case in cond["cases"]:
                jobs = jobs_by_case[case["case_id"]]
                hidden = common.wrap_hidden(common.hidden_for(jobs, seed, evp))
                for method in ["baseline"] + names:
                    ctx = dict(pools=pools, med=med,
                               prior=priors.get(method),
                               params=CONFIGS.get(method, {}))
                    decide = common.make_decider("baseline" if method == "baseline" else "role_e", ctx)
                    m, diff, steps = common.replay(jobs, hidden, zones, case, 1.0, 0.5, decide)
                    assert not m["invalid_actions"], (fold, seed, method)
                    assert diff < 1e-9, (fold, seed, method, diff)
                    max_diff = max(max_diff, diff)
                    rows.append(dict(fold=fold, case=case["case_id"], seed=seed, method=method,
                                     weighted_on_time_rate=float(m["weighted_on_time_rate"]),
                                     weighted_completion_rate=float(m["weighted_completion_rate"]),
                                     cost=float(m["cost"]), observations=float(m["observations"]),
                                     actions=steps))
        print("fold", fold, "done", round(time.time() - t0, 1), "s", flush=True)

    d = pd.DataFrame(rows)
    d.to_csv(OUT / "trials.csv", index=False)
    base = d[d.method == "baseline"].set_index(["fold", "case", "seed"]).weighted_on_time_rate
    summary = {"seeds": [SEED0, SEED0 + args.seeds - 1], "folds": 2,
               "max_independent_metric_diff": max_diff, "configs": {}, "selection_rule":
               "maximise pooled paired delta; tie-break by worst-fold delta"}
    rng = np.random.default_rng(20261122)
    for n in names:
        x = d[d.method == n].set_index(["fold", "case", "seed"]).weighted_on_time_rate
        delta = (x - base).dropna()
        by_fold = {int(k): float(v.mean() * 100) for k, v in delta.groupby(level=0)}
        by_cluster = delta.groupby(level=[0, 2]).mean().to_numpy()
        boot = [float(by_cluster[rng.integers(0, len(by_cluster), len(by_cluster))].mean() * 100)
                for _ in range(2000)]
        summary["configs"][n] = dict(
            params={k: v for k, v in CONFIGS[n].items()},
            otr_pooled=float(x.mean()), delta_pooled_pp=float(delta.mean() * 100),
            delta_ci95_pp=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            delta_fold0_pp=by_fold.get(0), delta_fold1_pp=by_fold.get(1),
            worst_fold_pp=min(by_fold.values()), n=int(len(delta)))
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                                      encoding="utf8")
    table = pd.DataFrame(summary["configs"]).T.sort_values("delta_pooled_pp", ascending=False)
    print(table[["delta_pooled_pp", "delta_fold0_pp", "delta_fold1_pp", "worst_fold_pp"]]
          .round(3).to_string())
    print("max diff", max_diff, "elapsed", round(time.time() - t0, 1))


if __name__ == "__main__":
    main()
