"""T.1 natural-recovery comparison, scored before each valid observation arrives."""
import numpy as np


def recovery_events(t, x, valid, bad, scale):
    # The registered 0.2-s eligibility rule includes a one-sample margin.
    median_dt = np.median(np.diff(t))
    edges = np.diff(np.r_[False, bad, False].astype(int))
    eligibility, scores = [], []
    for start, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1), strict=True):
        reason = "eligible"
        if start < 2 or not valid[start-2:start].all():
            reason = "no_continuous_qualified_initial_pair"
        elif end == len(t):
            reason = "right_censored"
        else:
            stop = np.searchsorted(t, t[end] + .2, side="right")
            if stop == len(t) or t[stop-1] - t[end] < .2 - 1.1 * median_dt:
                reason = "short_reference_span"
            elif not valid[end:stop].all():
                reason = "nonqualified_recovery_reference"
            elif (np.diff(t[start-2:stop]) > 3 * median_dt).any():
                reason = "timestamp_gap"
        eligibility.append((start, reason))
        if reason != "eligible":
            continue
        for method in ("stale", "reset_decay", "stream"):
            last = start - 1
            velocity = (x[last] - x[last-1]) / (t[last] - t[last-1])
            errors = []
            for k in range(end, stop):
                elapsed = t[k] - t[last]
                prediction = x[last] - .2 * np.expm1(-elapsed / .2) * velocity
                errors.append(np.linalg.norm(prediction - x[k]) / scale)
                if method == "stale":
                    continue
                if method == "stream" and k > end:
                    velocity = (x[k] - x[last]) / elapsed
                else:
                    velocity = velocity * np.exp(-elapsed / .2)
                last = k
            scores.append((start, method, np.mean(errors), np.mean(np.asarray(errors) <= .05), len(errors)))
    return eligibility, scores
