"""Replay C's reviewed correction and E's behavior-preserving documentation copy."""
import json
from pathlib import Path

import pandas as pd
import runtime as rt


def main():
    root = Path(__file__).resolve().parents[1] / "results" / "reviewed_replay"
    root.mkdir(parents=True, exist_ok=True)
    cond, tasks, zones, train, seg = rt.acc.ar.load_all()
    fit = seg[(seg.split == "train") & (seg.eligible == 1)]
    shared = rt.context(fit, seg[(seg.split == "validation") & (seg.eligible == 1)])
    rows = []
    with (root / "events.jsonl").open("w", encoding="utf8") as event_file:
        for case in cond["cases"]:
            jobs = rt.acc.ar.jobs_of(tasks, case["case_id"])
            for seed in cond["validation_seeds"][:30]:
                for alpha, beta in rt.acc.MAPPINGS:
                    for method in ("C", "E"):
                        name = "C_repaired" if method == "C" else "E_documented"
                        row, events, violations = rt.run(case, jobs, zones, shared, name, method,
                                                         seed, alpha, beta)
                        assert not violations
                        rows.append(row)
                        event_file.write(json.dumps({"case": case["case_id"], "seed": seed,
                                                     "alpha": alpha, "beta": beta, "method": name,
                                                     "events": events}, ensure_ascii=False) + "\n")
                        if len(rows) % 50 == 0:
                            print(f"{len(rows)}/900", flush=True)
                            pd.DataFrame(rows).to_csv(root / "metrics.partial.csv", index=False)
    d = pd.DataFrame(rows)
    d.to_csv(root / "metrics.csv", index=False)
    print(d.groupby("method")[["weighted_on_time_rate", "weighted_completion_rate",
                                "cost", "invalid_actions", "independent_numeric_diff"]].agg(["mean", "max"]))


if __name__ == "__main__":
    main()
