"""Candidate parameter grid for role E (all candidates must be reported)."""

# Each config changes only the duration belief; the decision core is fixed.
CONFIGS = {
    # sample-count shrinkage to the pooled global prior (feedback on)
    "pooled_s5_fb3": dict(balance="pooled", strength=5.0, feedback_strength=3.0),
    "pooled_s10_fb3": dict(balance="pooled", strength=10.0, feedback_strength=3.0),
    "pooled_s20_fb3": dict(balance="pooled", strength=20.0, feedback_strength=3.0),
    "pooled_s40_fb3": dict(balance="pooled", strength=40.0, feedback_strength=3.0),
    # feedback sensitivity at the reference strength
    "pooled_s20_fb0": dict(balance="pooled", strength=20.0, feedback_strength=0.0),
    "pooled_s20_fb1": dict(balance="pooled", strength=20.0, feedback_strength=1.0),
    "pooled_s20_fb6": dict(balance="pooled", strength=20.0, feedback_strength=6.0),
    # winsorised upper tail
    "pooled_s20_fb3_w90": dict(balance="pooled", strength=20.0, feedback_strength=3.0, winsor_q=0.90),
    "pooled_s20_fb3_w95": dict(balance="pooled", strength=20.0, feedback_strength=3.0, winsor_q=0.95),
    # environment-balanced aggregation
    "env_s20_fb3": dict(balance="env", strength=20.0, feedback_strength=3.0),
    "env_s20_fb3_w95": dict(balance="env", strength=20.0, feedback_strength=3.0, winsor_q=0.95),
    "env_s40_fb3_w95": dict(balance="env", strength=40.0, feedback_strength=3.0, winsor_q=0.95),
    # region-balanced aggregation
    "region_s20_fb3": dict(balance="region", strength=20.0, feedback_strength=3.0),
    "region_s20_fb3_w95": dict(balance="region", strength=20.0, feedback_strength=3.0, winsor_q=0.95),
    # sensitivity sweep around the grouped-CV winner (added before validation)
    "env_s5_fb3_w95": dict(balance="env", strength=5.0, feedback_strength=3.0, winsor_q=0.95),
    "env_s10_fb3_w95": dict(balance="env", strength=10.0, feedback_strength=3.0, winsor_q=0.95),
    "env_s80_fb3_w95": dict(balance="env", strength=80.0, feedback_strength=3.0, winsor_q=0.95),
    "env_s40_fb3": dict(balance="env", strength=40.0, feedback_strength=3.0),
    "env_s40_fb3_w90": dict(balance="env", strength=40.0, feedback_strength=3.0, winsor_q=0.90),
    "env_s40_fb0_w95": dict(balance="env", strength=40.0, feedback_strength=0.0, winsor_q=0.95),
    "env_s40_fb1_w95": dict(balance="env", strength=40.0, feedback_strength=1.0, winsor_q=0.95),
    "env_s40_fb6_w95": dict(balance="env", strength=40.0, feedback_strength=6.0, winsor_q=0.95),
    "region_s40_fb3_w95": dict(balance="region", strength=40.0, feedback_strength=3.0, winsor_q=0.95),
}

QUICK = ["pooled_s20_fb3", "env_s20_fb3_w95"]
