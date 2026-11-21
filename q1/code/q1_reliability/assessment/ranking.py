"""Ranking metrics for binary targets and probability scores."""

import numpy as np
from sklearn.metrics import average_precision_score


def ranking_metrics(y, p):
    """Return ranking metrics with a fractional top-10% quota."""
    y = np.asarray(y, dtype=np.int8).ravel()
    p = np.asarray(p, dtype=float).ravel()
    n = int(y.size)
    positives = int(y.sum())

    if n == 0:
        return {
            "n": 0,
            "positives": 0,
            "prevalence": float("nan"),
            "ap": float("nan"),
            "precision10": float("nan"),
            "recall10": float("nan"),
        }

    prevalence = positives / n
    ap = (
        float(average_precision_score(y, p))
        if 0 < positives < n
        else float("nan")
    )

    quota = n / 10.0
    order = np.argsort(-p, kind="stable")
    cutoff = p[order[(n - 1) // 10]]
    above = p > cutoff
    tied = p == cutoff
    fraction = (quota - int(above.sum())) / int(tied.sum())
    true_positives = float(y[above].sum() + fraction * y[tied].sum())

    precision10 = true_positives / quota
    recall10 = true_positives / positives if positives > 0 else float("nan")

    return {
        "n": n,
        "positives": positives,
        "prevalence": float(prevalence),
        "ap": ap,
        "precision10": float(precision10),
        "recall10": float(recall10),
    }
