"""Training-only B/C configuration selection under environment exclusions.

There is only one L environment in each outer fit partition. The inner held
group contains that L environment and the inner fit has no L. Its L prior must
therefore fall back to its own training-global pool. Complementary inner folds
cannot evaluate all three families and are not fabricated.
"""
import json
from pathlib import Path

import pandas as pd
import runtime as rt

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "results" / "nested_selection"
B_GRID = [dict(K=k) for k in (16, 32, 64)]
C_GRID = [dict(samples=n, dt=dt) for n in (64, 256) for dt in (.5, 1.)]


def cfg_name(method, cfg):
    return method + "_" + "_".join(str(k) + str(v).replace(".", "p") for k, v in cfg.items())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cond, tasks, zones, train, seg = rt.acc.ar.load_all()
    tr = seg[(seg.split == "train") & (seg.eligible == 1)]
    groups = json.loads((rt.acc.BREAK / "results" / "grouped_training" / "registration.json")
                        .read_text(encoding="utf8"))["groups"]
    registration = {"purpose": "retrospective training-only configuration-selection audit; not pristine prospective validation",
                    "outer_held_groups": groups, "B_grid": B_GRID, "C_repaired_grid": C_GRID,
                    "inner_seeds": list(range(2026112400, 2026112408)),
                    "outer_seeds": list(range(2026120100, 2026120120)),
                    "mappings": [[1., .5]],
                    "selection": "max inner equal-case mean on-time rate; exact ties choose smaller K/samples then coarser dt",
                    "missing_train_family": "copy inner-training global source-duration pool; never use held data",
                    "inner_folds": "one all-family held group per outer fold; L scarcity prevents a complementary all-family group",
                    "adoption_order": "legal semantics, positive nominal improvement, no negative outer-fold mean versus baseline, then highest worst-outer-fold mean; if tied, nominal rate then runtime",
                    "source_hashes": {p.name: rt.acc.sha256(p) for p in sorted((HERE / "frozen").glob("*.py"))}}
    regpath = OUT / "registration.json"
    if regpath.exists():
        if json.loads(regpath.read_text(encoding="utf8")) != registration:
            raise RuntimeError("Existing frozen registration differs")
    else:
        regpath.write_text(json.dumps(registration, ensure_ascii=False, indent=2), encoding="utf8")
    inner_rows, outer_rows, selections, partitions = [], [], [], []
    for fold, held in enumerate(groups):
        outer_fit = tr[~tr.environment_id.isin(held)]
        outer_eval = tr[tr.environment_id.isin(held)]
        fit_envs = sorted(outer_fit.environment_id.unique())
        l_envs = sorted(outer_fit[outer_fit.family_id == "L"].environment_id.unique())
        assert len(l_envs) == 1
        inner_held = l_envs + [x for x in fit_envs if x not in l_envs][:3]
        inner_fit = outer_fit[~outer_fit.environment_id.isin(inner_held)]
        inner_eval = outer_fit[outer_fit.environment_id.isin(inner_held)]
        assert set(inner_fit.environment_id).isdisjoint(inner_eval.environment_id)
        assert set(outer_fit.environment_id).isdisjoint(outer_eval.environment_id)
        assert set(inner_eval.family_id) == {"A", "L", "P"}
        state = rt.context(inner_fit, inner_eval)
        partitions.append({"fold": fold, "outer_fit": fit_envs, "outer_held": held,
                           "inner_fit": sorted(inner_fit.environment_id.unique()), "inner_held": inner_held,
                           "inner_missing_train_families": state["missing_train_families"],
                           "inner_fit_units": list(inner_fit.unit_id), "inner_eval_units": list(inner_eval.unit_id),
                           "outer_fit_units": list(outer_fit.unit_id), "outer_eval_units": list(outer_eval.unit_id)})
        means = {}
        for method, grid in (("B", B_GRID), ("C", C_GRID)):
            for cfg in grid:
                name = cfg_name(method, cfg)
                block = []
                for case in cond["cases"]:
                    jobs = rt.acc.ar.jobs_of(tasks, case["case_id"])
                    for seed in registration["inner_seeds"]:
                        row, _, vio = rt.run(case, jobs, zones, state, name, method, seed, **cfg)
                        assert not vio and not row["invalid_actions"]
                        row.update(fold=fold, stage="inner", config=name)
                        block.append(row); inner_rows.append(row)
                means[name] = sum(r["weighted_on_time_rate"] for r in block) / len(block)
                print("inner", fold, name, round(means[name], 6), flush=True)
        chosen = {}
        for method, grid in (("B", B_GRID), ("C", C_GRID)):
            if method == "B":
                winner = min(grid, key=lambda c: (-means[cfg_name(method, c)], c["K"]))
            else:
                winner = min(grid, key=lambda c: (-means[cfg_name(method, c)], c["samples"], -c["dt"]))
            chosen[method] = winner
        selections.append({"fold": fold, "selected": chosen, "inner_means": means})
        (OUT / "selections.partial.json").write_text(json.dumps(selections, indent=2), encoding="utf8")
        pd.DataFrame(inner_rows).to_csv(OUT / "inner_trials.csv", index=False)

        outer_state = rt.context(outer_fit, outer_eval)
        policies = [("baseline", "baseline", {}), ("B_fixed_K64", "B", {"K": 64}),
                    ("C_fixed_repaired", "C", {"samples": 256, "dt": 1.}), ("E_fixed", "E", {}),
                    ("B_inner_selected", "B", chosen["B"]), ("C_inner_selected", "C", chosen["C"])]
        for case in cond["cases"]:
            jobs = rt.acc.ar.jobs_of(tasks, case["case_id"])
            for seed in registration["outer_seeds"]:
                cache = {}
                for name, method, cfg in policies:
                    cachekey = (method, tuple(sorted(cfg.items())))
                    if cachekey not in cache:
                        cache[cachekey] = rt.run(case, jobs, zones, outer_state, name, method, seed, **cfg)
                    row, _, vio = cache[cachekey]
                    assert not vio and not row["invalid_actions"]
                    row = dict(row, method=name, fold=fold, stage="outer")
                    outer_rows.append(row)
        pd.DataFrame(outer_rows).to_csv(OUT / "outer_trials.csv", index=False)
        (OUT / "partitions.json").write_text(json.dumps(partitions, ensure_ascii=False, indent=2), encoding="utf8")
        print("outer fold completed", fold, flush=True)
    (OUT / "selections.json").write_text(json.dumps(selections, indent=2), encoding="utf8")
    d = pd.DataFrame(outer_rows)
    p = d.pivot(index=["fold", "case", "seed"], columns="method", values="weighted_on_time_rate")
    delta = p.subtract(p.baseline, axis=0) * 100
    print(delta.groupby(level="fold").mean().to_string())
    delta.groupby(level="fold").mean().to_csv(OUT / "outer_delta_pp.csv")


if __name__ == "__main__":
    main()
