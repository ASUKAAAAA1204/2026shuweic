"""Freeze the grouped-CV selection *before* any full validation run.

Rule (declared before the grid was run): maximise the pooled paired delta over
the two environment folds; tie-break by the worst fold.  This script writes the
decision, the full candidate table reference, code hashes and the disclosure of
the pre-selection compatibility checks.
"""
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results"
sys.path.insert(0, str(HERE))
from configs import CONFIGS  # noqa: E402


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    cv = json.loads((RES / "grouped_cv" / "summary.json").read_text(encoding="utf8"))
    cfg = cv["configs"]
    ranked = sorted(cfg.items(), key=lambda kv: (-kv[1]["delta_pooled_pp"],
                                                 -kv[1]["worst_fold_pp"]))
    name, best = ranked[0]
    decision = {
        "selected_config": name,
        "selected_params": CONFIGS[name],
        "selection_rule": cv["selection_rule"],
        "selected_cv": best,
        "candidate_count": len(cfg),
        "ranking": [{"config": k, "delta_pooled_pp": v["delta_pooled_pp"],
                     "worst_fold_pp": v["worst_fold_pp"],
                     "delta_ci95_pp": v["delta_ci95_pp"]} for k, v in ranked],
        "grouped_cv_source": str((RES / "grouped_cv" / "summary.json").relative_to(HERE.parent)),
        "grouped_cv_seeds": cv["seeds"],
        "code_sha256": {p.name: sha(p) for p in
                        (HERE / "robust_chain_policy.py", HERE / "configs.py",
                         HERE / "run_grouped_cv.py", HERE / "common.py")},
        "pre_validation_checks_disclosed": [
            "v2_s20 nominal validation replay (3 cases x 30 seeds) to verify the "
            "shared reference number; reproduced 90.2431% (S1 99.1667 / S2 87.5000 / "
            "S3 84.0625), 0.08pp above the quoted 90.16% shared number.",
            "pooled_s20_fb3 nominal validation replay (3 cases x 30 seeds) as a code "
            "compatibility check; 90.6134%. It did NOT enter the selection rule.",
        ],
        "note": "Validation is used once for confirmation only; the selected config "
                "is not changed after this file is written.",
    }
    (RES / "selection_decision.json").write_text(
        json.dumps(decision, indent=2, ensure_ascii=False), encoding="utf8")
    print(name, best["delta_pooled_pp"], best["delta_ci95_pp"])


if __name__ == "__main__":
    main()
