"""Reproduce the four frozen acceptance policies and all common metrics."""
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
                                                  feedback=snapshots,decision_wall_ms=timings),ensure_ascii=False)+"\n")
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
