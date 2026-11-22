"""Public-only scheduling interface. Construct a fresh Policy per episode."""
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
