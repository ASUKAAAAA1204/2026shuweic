"""Assemble a self-contained frozen review package, preserving formal baseline."""
import ast
import json
import shutil
from pathlib import Path

import pandas as pd
import runtime as rt

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "采用版"

POLICY = '''"""Public-only scheduling interface. Construct a fresh Policy per episode."""
import json
from pathlib import Path
import numpy as np
import b_policy
import c_policy
import e_policy
from baseline import action
from deadline_policy import choose, pressure

class Policy:
    def __init__(self, candidate="E_env_s40_fb3"):
        self.candidate = candidate
        model = json.loads((Path(__file__).resolve().parents[1] / "model" / "priors.json").read_text(encoding="utf8"))
        self.pools = {f: np.asarray(v, float) for f, v in model["training_family_samples"].items()}
        self.medians = {f: float(np.median(v)) for f, v in self.pools.items()}
        self.e_prior = {f: e_policy.Belief(v["values"], v["weights"]) for f, v in model["E_prior"].items()}
        self.history = {}
        self.b = b_policy.JointScenarioPolicy(K=64, strength=20, feedback_strength=3, seed=20261122,
                                               base="survival", stratified=True)
        if candidate not in ("E_env_s40_fb3", "B_K64", "C_repaired", "baseline"):
            raise ValueError(candidate)

    def act(self, observation):
        o = dict(observation, jobs=sorted(observation["jobs"], key=lambda j: j["job_id"]))
        if self.candidate == "E_env_s40_fb3":
            return e_policy.select(o, self.e_prior, self.history, feedback_strength=3., nq_root=5, grid=64)
        if self.candidate == "B_K64":
            return self.b.select(o, self.pools, self.history)
        if self.candidate == "C_repaired":
            return c_policy.select(o, self.pools, self.history, mode="mc", strength=20,
                                   feedback_strength=3, topk=4, mc_samples=256, mc_seed=20261122,
                                   timeline="quantile", scenarios=(.2,.5,.8), tail_quantiles=3,
                                   dt=1., orders=("edf","wdd","slack"), use_dp=True,
                                   use_greedy=True, force_first=False)
        return (choose(o, self.pools, self.history) if pressure(o, self.pools, self.history)
                else action(o, self.medians, "density", "minimum"))

    def feedback(self, event, family, alpha, beta):
        if event["action"]["type"] == "execute" and event.get("completion") is not None:
            source_duration = (event["completion"] - event["service_start"] - beta) / alpha
            self.history.setdefault(family, []).append(source_duration)
'''

RUNNER = '''"""Reproduce the four frozen acceptance policies and all common metrics."""
import argparse
import hashlib
import json
import random
import time
from pathlib import Path
import pandas as pd
from engine import Engine
from independent import independent_rebuild
from policy import Policy

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", default="baseline,B_K64,C_repaired,E_env_s40_fb3")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf8"))
    for name, expected_hash in manifest["files"].items():
        path = ROOT / name
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected_hash:
            raise RuntimeError("Frozen file changed: " + name)
    data = json.loads((ROOT / "data" / "protocol.json").read_text(encoding="utf8"))
    rows = []
    out = ROOT / "运行结果"
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with (out / "events_feedback.jsonl").open("w", encoding="utf8") as log:
        for case in data["cases"]:
            jobs = data["jobs"][case["case_id"]]
            family = {j["job_id"]:j["family_id"] for j in jobs}
            for seed in data["seeds"]:
                rng = random.Random(seed)
                dur = {}
                for j in jobs:
                    pool = data["validation_pool"][j["family_id"]]
                    dur[j["job_id"]] = pool[rng.randrange(len(pool))]
                hidden = {k:{"duration":v,"features":{"0.2":[],"0.6":[],"1.0":[]}} for k,v in dur.items()}
                for alpha,beta in data["mappings"]:
                    for method in args.methods.split(","):
                        policy = Policy(method)
                        e = Engine(jobs,hidden,data["zones"],case["horizon_s"],case["cost_budget"],
                                   case["observation_limit"],alpha,beta)
                        snapshots, timings = [], []
                        while not e.terminal:
                            before = {f:list(v) for f,v in policy.history.items()}
                            t0 = time.perf_counter()
                            a = policy.act(e.observe())
                            timings.append((time.perf_counter()-t0)*1000)
                            event = e.step(a)
                            policy.feedback(event, family.get(a.get("job_id")), alpha, beta)
                            snapshots.append({"before":before,"after":{f:list(v) for f,v in policy.history.items()}})
                        m = e.metrics()
                        check = independent_rebuild(jobs,e.events,data["zones"],e.H,e.B,e.K)
                        diff = max(abs(float(m[k])-float(check[k])) for k in check if k!="violations")
                        if check["violations"] or diff>1e-9 or m["invalid_actions"]:
                            raise RuntimeError("Independent legality or scoring failure")
                        rows.append(dict(case=case["case_id"],seed=seed,alpha=alpha,beta=beta,
                                         method=method,**m,independent_numeric_diff=diff))
                        log.write(json.dumps(dict(case=case["case_id"],seed=seed,alpha=alpha,beta=beta,
                                                  method=method,hidden_source_durations=dur,events=e.events,
                                                  feedback=snapshots,decision_wall_ms=timings),ensure_ascii=False)+"\\n")
                        if len(rows)%100==0:print("completed",len(rows),flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(out/"metrics.csv",index=False)
    expected = pd.read_csv(ROOT/"data"/"expected_metrics.csv")
    key=["case","seed","alpha","beta","method"]
    joined = d.merge(expected,on=key,suffixes=("_now","_expected"),validate="one_to_one")
    if len(joined)!=len(d):raise RuntimeError("Missing expected reference rows")
    metrics=["weighted_on_time_rate","weighted_completion_rate","weighted_tardiness_s","cost",
             "observations","elapsed_s","observed_overrun_s","reserved_idle_s","censored_units","invalid_actions"]
    maxdiff=max(float((joined[k+"_now"]-joined[k+"_expected"]).abs().max()) for k in metrics)
    if maxdiff>1e-9:raise RuntimeError("Reference regression mismatch: "+str(maxdiff))
    summary=dict(episodes=len(d),reference_max_absolute_diff=maxdiff,invalid_actions=int(d.invalid_actions.sum()),
                 independent_max_absolute_diff=float(d.independent_numeric_diff.max()),seconds=time.perf_counter()-started,
                 methods=args.methods.split(","))
    (out/"audit.json").write_text(json.dumps(summary,indent=2),encoding="utf8")
    print(json.dumps(summary,indent=2))

if __name__=="__main__":main()
'''


def main():
    for name in ("code", "model", "data"):
        (OUT / name).mkdir(parents=True, exist_ok=True)
    cond, tasks, zones, train, seg = rt.acc.ar.load_all()
    fit = seg[(seg.split == "train") & (seg.eligible == 1)]
    validation = seg[(seg.split == "validation") & (seg.eligible == 1)]
    shared = rt.context(fit, validation)
    # Match the frozen formal training samples byte-for-value, preserving order.
    pools = {f: g.duration_s.to_numpy(float).tolist() for f, g in train.groupby("family_id")}
    priors = {"training_family_samples": pools,
              "E_prior": {f: {"values": b.values.tolist(), "weights": b.weights.tolist()}
                          for f, b in shared["e"]["e_prior"]["env_s40_fb3"].items()},
              "training_units": list(fit.unit_id), "training_environments": sorted(fit.environment_id.unique()),
              "E_config": {"balance": "env", "strength": 40., "feedback_strength": 3., "grid": 64, "nq_root": 5},
              "B_config": {"K": 64, "strength": 20, "feedback_strength": 3, "seed": 20261122, "base": "survival"},
              "C_config": {"samples": 256, "dt": 1., "topk": 4, "version": "legal-MC-repaired"}}
    (OUT / "model" / "priors.json").write_text(json.dumps(priors, ensure_ascii=False, indent=2), encoding="utf8")
    data = {"cases": cond["cases"], "zones": zones, "seeds": cond["validation_seeds"][:30],
            "mappings": rt.acc.MAPPINGS,
            "jobs": {c["case_id"]: rt.acc.ar.jobs_of(tasks, c["case_id"]) for c in cond["cases"]},
            "validation_pool": {f: g.duration_s.to_numpy(float).tolist() for f, g in validation.groupby("family_id")},
            "validation_source_units": {f: list(g.unit_id) for f, g in validation.groupby("family_id")}}
    (OUT / "data" / "protocol.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf8")
    for name in ("b_policy.py", "c_policy.py", "e_policy.py"):
        shutil.copyfile(HERE / "frozen" / name, OUT / "code" / name)
    for name in ("engine.py", "baseline.py", "deadline_policy.py"):
        shutil.copyfile(rt.acc.ROOT / "第三问改进正式方案" / "code" / name, OUT / "code" / name)
    (OUT / "code" / "policy.py").write_text(POLICY, encoding="utf8")
    (OUT / "code" / "run_all.py").write_text(RUNNER, encoding="utf8")
    source = (HERE / "acceptance_run.py").read_text(encoding="utf8")
    tree = ast.parse(source)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "independent_rebuild")
    (OUT / "code" / "independent.py").write_text("import math\n\n" + ast.get_source_segment(source, node) + "\n", encoding="utf8")
    old = pd.read_csv(HERE.parent / "results" / "full_replay" / "metrics.csv")
    new = pd.read_csv(HERE.parent / "results" / "reviewed_replay" / "metrics.csv")
    expected = pd.concat([old[old.method.isin(["baseline", "B_K64", "E_env_s40_fb3"])],
                          new[new.method == "C_repaired"]], ignore_index=True)
    expected.to_csv(OUT / "data" / "expected_metrics.csv", index=False)
    versions = {"python": sys_version(), "numpy": __import__("numpy").__version__, "pandas": pd.__version__}
    (OUT / "environment.json").write_text(json.dumps(versions, indent=2), encoding="utf8")
    (OUT / "requirements.txt").write_text("numpy\npandas\n", encoding="utf8")
    files = {}
    for directory in (OUT / "code", OUT / "model", OUT / "data"):
        for path in sorted(directory.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                files[path.relative_to(OUT).as_posix()] = rt.acc.sha256(path)
    manifest = {"primary": "E_env_s40_fb3", "secondary": ["B_K64", "C_repaired"],
                "original_C": "rejected: duration-dependent free skipping before observation",
                "formal_baseline_preserved": True, "files": files}
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf8")
    print("Bundle built", OUT)


def sys_version():
    import sys
    return sys.version


if __name__ == "__main__":
    main()
