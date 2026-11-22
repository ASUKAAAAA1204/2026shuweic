"""Finalise summary.json: add paired information effect vs v2_s20 and attach the
holdout, oracle-bound and prefix-quality artefacts.  Reads only result files."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from protocol import bootstrap_ci, MAPPINGS  # noqa: E402

OUT = HERE.parent / "results"
REF = "v2_s20"


def paired_vs(d, ref):
    out = {}
    for m in sorted(d.method.unique()):
        if m == ref:
            continue
        piv = d[d.method.isin([ref, m])].pivot_table(
            index=["case", "seed", "alpha", "beta"], columns="method", values="weighted_on_time_rate")
        delta = piv[m] - piv[ref]
        per_seed = delta.groupby(level="seed").mean()
        idx = delta.index
        nom = delta[np.isclose(idx.get_level_values("alpha"), 1.0)
                    & np.isclose(idx.get_level_values("beta"), .5)]
        by_map = delta.groupby(level=["alpha", "beta"]).mean()
        out[m] = dict(
            nominal_delta_pp=float(100 * nom.mean()),
            overall_delta_pp=float(100 * delta.mean()),
            by_case_delta_pp={k: float(100 * v) for k, v in delta.groupby(level="case").mean().items()},
            by_mapping_delta_pp={f"{a}|{b}": float(100 * v) for (a, b), v in by_map.items()},
            bootstrap95_pp=[float(100 * x) for x in bootstrap_ci(per_seed.to_numpy())],
            positive_seeds=int((per_seed > 0).sum()),
            seeds=int(len(per_seed)),
        )
    return out


def main():
    d = pd.read_csv(OUT / "trials.csv")
    s = json.loads((OUT / "summary.json").read_text(encoding="utf8"))
    s["_paired_vs_v2_s20"] = paired_vs(d, REF)
    for name, f in (("_holdout", "holdout_summary.json"),
                    ("_oracle_bound_analysis_only", "oracle_bound.json"),
                    ("_prefix_quality", "prefix_quality.json")):
        p = OUT / f
        if p.exists():
            s[name] = json.loads(p.read_text(encoding="utf8"))
    (OUT / "summary.json").write_text(json.dumps(s, indent=2), encoding="utf8")
    print(json.dumps(s["_paired_vs_v2_s20"], indent=2))


if __name__ == "__main__":
    main()
