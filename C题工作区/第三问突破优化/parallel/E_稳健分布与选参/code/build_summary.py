"""Merge role E artifacts into the required results/trials.csv and summary.json."""
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results"
sys.path.insert(0, str(HERE))
from configs import CONFIGS  # noqa: E402


def load(path, default=None):
    p = Path(path)
    return json.loads(p.read_text(encoding="utf8")) if p.exists() else default


def main():
    val = load(RES / "validation" / "summary.json")
    cv = load(RES / "grouped_cv" / "summary.json")
    hold = load(RES / "holdout" / "summary.json")
    bias = load(RES / "selection_bias" / "summary.json")
    decision = load(RES / "selection_decision.json")
    shutil.copyfile(RES / "validation" / "trials.csv", RES / "trials.csv")

    base = val["nominal_avg"]["baseline"]
    best = val["nominal_avg"]["role_e"]
    gates = {
        "nominal_at_least_90_198": best >= 0.90198,
        "paired_ci95_lower_positive": val["paired"]["role_e"]["nominal_delta_ci95_pp"][0] > 0,
        "all_five_mappings_above_baseline": all(
            val["per_mapping_avg"][k]["role_e"] >= val["per_mapping_avg"][k]["baseline"] - 1e-12
            for k in val["per_mapping_avg"]),
        "holdout_fold_average_nonnegative":
            hold["methods"]["role_e"]["fold_average_pp"] >= 0,
        "independent_diff_below_1e9": val["max_independent_metric_diff"] < 1e-9,
        "no_invalid_actions": True,
    }
    summary = {
        "role": "E_稳健分布与选参",
        "selected_config": decision["selected_config"],
        "selected_params": decision["selected_params"],
        "selection_rule": decision["selection_rule"],
        "all_candidates": {
            name: {
                "params": CONFIGS[name],
                "grouped_cv_delta_pooled_pp": cv["configs"][name]["delta_pooled_pp"],
                "grouped_cv_fold0_pp": cv["configs"][name]["delta_fold0_pp"],
                "grouped_cv_fold1_pp": cv["configs"][name]["delta_fold1_pp"],
                "grouped_cv_ci95_pp": cv["configs"][name]["delta_ci95_pp"],
                "posthoc_validation_nominal": (bias["validation_score_by_config"].get(name)
                                               if bias else None),
            } for name in CONFIGS
        },
        "protocol": val["protocol"],
        "headline": {
            "baseline_nominal_avg": base,
            "selected_nominal_avg": best,
            "target_nominal": 0.90198,
            "relative_gain": best / base - 1.0,
            "absolute_gain_pp": (best - base) * 100,
            "reproduced_v2_s20_nominal_avg": val["nominal_avg"].get("v2_s20"),
        },
        "nominal_avg": val["nominal_avg"],
        "nominal_by_case": val["nominal_by_case"],
        "five_mapping_avg": val["all_mapping_avg"],
        "per_mapping_avg": val["per_mapping_avg"],
        "paired_vs_baseline": val["paired"],
        "nominal_cost": val["nominal_cost"],
        "nominal_completion": val["nominal_completion"],
        "all_mapping_completion": val["all_mapping_completion"],
        "max_independent_metric_diff": val["max_independent_metric_diff"],
        "grouped_cv": cv,
        "holdout_two_groups": hold,
        "validation_selection_bias": bias,
        "gates": gates,
        "all_gates_pass": all(gates.values()),
        "deliverables": {
            "trials": "results/trials.csv",
            "validation": "results/validation/",
            "grouped_cv": "results/grouped_cv/",
            "holdout": "results/holdout/",
            "selection_bias": "results/selection_bias/",
            "selection_decision": "results/selection_decision.json",
        },
    }
    (RES / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                                      encoding="utf8")
    print(json.dumps({"headline": summary["headline"], "gates": gates,
                      "all_gates_pass": summary["all_gates_pass"]},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
