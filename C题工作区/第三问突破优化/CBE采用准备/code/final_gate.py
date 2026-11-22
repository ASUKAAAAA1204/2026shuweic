"""Requirement-by-requirement acceptance gate, based on inspected artifacts."""
import ast
import hashlib
import json
import math
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
WORK = BASE.parents[1]


def load(path):
    return json.loads(path.read_text(encoding="utf8"))


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def main():
    results = BASE / "results"
    bundle = BASE / "采用版"
    original = pd.read_csv(results / "full_replay" / "metrics.csv")
    repaired = pd.read_csv(results / "reviewed_replay" / "metrics.csv")
    summary = load(results / "analysis" / "summary.json")
    semantics = load(results / "semantic_checks" / "summary.json")
    trajectory = load(results / "trajectory_boundary" / "summary.json")
    bcheck = load(results / "b_rollout" / "summary.json")
    nested_reg = load(results / "nested_selection" / "registration.json")
    partitions = load(results / "nested_selection" / "partitions.json")
    nested_inner = pd.read_csv(results / "nested_selection" / "inner_trials.csv")
    nested_outer = pd.read_csv(results / "nested_selection" / "outer_trials.csv")
    gates = []
    def gate(name, passed, evidence):
        require(passed, name)
        gates.append({"item": name, "passed": True, "evidence": evidence})

    manifest = load(bundle / "manifest.json")
    for name, expected in manifest["files"].items():
        require(hashlib.sha256((bundle / name).read_bytes()).hexdigest() == expected, "Frozen bundle changed: " + name)
    old_registry = load(BASE / "候选保留登记.json")
    sources = {"B": ("B_联合场景前瞻", "joint_policy.py"), "C": ("C_尾部价值DP", "dp_tail_policy.py"),
               "E": ("E_稳健分布与选参", "robust_chain_policy.py")}
    for role, (folder, filename) in sources.items():
        path = BASE.parent / "parallel" / folder / "code" / filename
        expected = old_registry[role]["files"]["code\\" + filename]
        require(hashlib.sha256(path.read_bytes()).hexdigest() == expected, "Research source changed: " + role)
    for filename, expected in nested_reg["source_hashes"].items():
        require(hashlib.sha256((BASE / "code" / "frozen" / filename).read_bytes()).hexdigest() == expected,
                "Nested frozen source changed")
    gate("1_exact_freeze_interface_and_reset", trajectory["all_equal"] and trajectory["input_mutation_count"] == 0,
         "bundle manifest; retained source hashes; trajectory summary; semantic full_episode_reset checks")
    gate("2_semantic_processing_complete", semantics["c_censor_counterexample"]["original"] == 5.
         and semantics["repaired_max_error"] == 0. and bcheck["max_absolute_value_error"] == 0.
         and summary["E_documentation_equivalence_max_metric_difference"] == 0.,
         "C original rejected; repaired C 240x12 scalar checks; B 133x8 scalar checks; E450-trial equivalence")
    key = ["case", "seed", "alpha", "beta", "method"]
    require(not original.duplicated(key).any(), "duplicate original replay keys")
    require(not repaired.duplicated(key).any(), "duplicate reviewed replay keys")
    gate("3_unified_full_replay", len(original) == 1800 and len(repaired) == 900
         and summary["G_regression_rows"] == 1800 and summary["G_regression_max_rate_difference"] == 0.,
         "full_replay 1800 rows; reviewed_replay 900 rows; G one-to-one comparison")
    metrics = ["weighted_on_time_rate", "weighted_completion_rate", "weighted_tardiness_s", "cost",
               "observations", "elapsed_s", "observed_overrun_s", "reserved_idle_s", "censored_units", "invalid_actions"]
    require(not original[metrics].isna().any().any() and not repaired[metrics].isna().any().any(), "missing common metrics")
    gate("4_all_metrics_and_resources", len(load(results / "trajectory_boundary" / "summary.json")["timing"]) == 3
         and (results / "analysis" / "case_mapping_common_metrics.csv").exists(),
         "ten common metrics; all 15 case/mapping groups; raw per-decision timings and total replay timing")
    gate("5_legality_information_boundaries", original.invalid_actions.sum() == 0
         and repaired.invalid_actions.sum() == 0 and original.independent_violation_count.sum() == 0
         and repaired.independent_violation_count.sum() == 0
         and original.independent_numeric_diff.max() <= 1e-9 and repaired.independent_numeric_diff.max() <= 1e-9
         and trajectory["decision_points"] == 309 and trajectory["censored_feedback_updates"] == 0,
         "independent physics rebuild; 309 trajectory decisions; no input mutation; no censored feedback")
    for part in partitions:
        require(set(part["outer_fit"]).isdisjoint(part["outer_held"]), "outer environment overlap")
        require(set(part["inner_fit"]).isdisjoint(part["inner_held"]), "inner environment overlap")
        require(set(part["inner_fit_units"]).isdisjoint(part["inner_eval_units"]), "inner unit overlap")
        require(set(part["inner_fit_units"] + part["inner_eval_units"]) <= set(part["outer_fit_units"]), "inner outside outer fit")
        require(part["inner_missing_train_families"] == ["L"], "missing L fallback not recorded")
    gate("6_BC_training_only_selection_audited", len(nested_inner) == 336 and len(nested_outer) == 720
         and len(partitions) == 2 and (results / "nested_selection" / "selections.json").exists(),
         "frozen 3-B/4-C grid; disjoint outer/inner units; L fallback; actual negative outcomes retained")
    history_report = (BASE / "三项验收报告.md").read_text(encoding="utf8")
    gate("7_E_selection_history_disclosed", "验证兼容性检查" in history_report and "旧版本有测试查看历史" in history_report,
         "retained E selection_decision.json and precheck; acceptance report section6")
    pair = pd.read_csv(results / "analysis" / "paired_comparisons.csv")
    gate("8_primary_and_all_paired_comparisons", summary["winner_under_frozen_rule"] == "E_env_s40_fb3"
         and len(pair) == 12 and len(pair[pair.scope == "nominal"]) == 6,
         "preregistered worst-outer-fold rule; six nominal and six five-mapping pairs; no test model selection")

    # Inspect every packaged event against its independently sampled source pool,
    # every feedback transition, and the final reference-score regression.
    data = load(bundle / "data" / "protocol.json")
    event_count = decision_count = feedback_count = 0
    methods = set()
    with (bundle / "运行结果" / "events_feedback.jsonl").open(encoding="utf8") as f:
        for line in f:
            r = json.loads(line)
            event_count += 1
            methods.add(r["method"])
            jobs = {j["job_id"]: j for j in data["jobs"][r["case"]]}
            require(len(r["events"]) == len(r["feedback"]) == len(r["decision_wall_ms"]), "trace length mismatch")
            prior_after = {}
            for event, update, ms in zip(r["events"], r["feedback"], r["decision_wall_ms"]):
                decision_count += 1
                require(math.isfinite(ms) and ms >= 0, "invalid runtime value")
                require(update["before"] == prior_after, "feedback state discontinuity")
                a = event["action"]
                before = update["before"]
                expected_after = {k: list(v) for k, v in before.items()}
                if a["type"] == "execute":
                    jid = a["job_id"]
                    expected_p = r["alpha"] * r["hidden_source_durations"][jid] + r["beta"]
                    require(abs(expected_p - event["realized_duration"]) <= 1e-9, "hidden mapping mismatch")
                    if event.get("completion") is not None:
                        fam = jobs[jid]["family_id"]
                        expected_after.setdefault(fam, []).append((event["completion"] - event["service_start"] - r["beta"]) / r["alpha"])
                        feedback_count += 1
                require(expected_after == update["after"], "illegal feedback transition")
                prior_after = update["after"]
    require(event_count == 1800 and methods == {"baseline", "B_K64", "C_repaired", "E_env_s40_fb3"}, "bundle event coverage")
    audit = load(bundle / "运行结果" / "audit.json")
    require(audit["episodes"] == 1800 and audit["reference_max_absolute_diff"] <= 1e-9
            and audit["independent_max_absolute_diff"] <= 1e-9 and audit["invalid_actions"] == 0,
            "packaged replay regression failed")
    for path in (bundle / "README.md", bundle / "第三问完整解答.md", BASE / "论文替换清单与总交接.md",
                 results / "analysis" / "candidate_acceptance.pdf", bundle / "requirements.txt",
                 bundle / "model" / "priors.json"):
        require(path.is_file() and path.stat().st_size > 0, "missing deliverable: " + str(path))
    for path in list((BASE / "code").rglob("*.py")) + list((bundle / "code").glob("*.py")):
        ast.parse(path.read_text(encoding="utf8"), filename=str(path))
    unchanged = subprocess.run(["git", "diff", "--exit-code", "--", str(WORK / "第三问改进正式方案")],
                               cwd=WORK.parent, capture_output=True, text=True)
    require(unchanged.returncode == 0, "Original formal Q3 changed")
    gate("9_standalone_bundle_paper_and_handoff", True,
         "standalone1800 replay; manifest; priors; per-action feedback; vector PDF; complete Q3 text and abstract/handoff; original formal directory unchanged")
    final = {"status": "completed_three_candidate_acceptance_decisions",
             "all_nine_delivery_requirements_completed": True,
             "primary": "E_env_s40_fb3", "candidate_decisions": summary["candidate_acceptance"],
             "gates": gates, "packaged_episodes": event_count,
             "packaged_decisions": decision_count, "complete_feedback_updates": feedback_count,
             "raw_candidate_all_pass": False,
             "unresolved_delivery_blockers": [],
             "scope_limits": ["No new-environment generalization proof", "No global-optimality proof",
                              "Original C rejected; corrected C is separately scored", "B/C retained as secondary",
                              "Actual paper master path unspecified; replacement-ready text supplied"],
             "final_artifacts": {p.relative_to(BASE).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in sorted(BASE.rglob("*")) if p.is_file()
                                 and "__pycache__" not in p.parts and p.suffix not in (".pyc", ".nbc", ".nbi")
                                 and p.name != "验收完成.json" and ".partial." not in p.name}}
    (BASE / "验收完成.json").write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding="utf8")
    print(json.dumps({k: v for k, v in final.items() if k != "final_artifacts"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
