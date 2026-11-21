"""Standalone One Euro velocity filters for short position histories.

The candidate grid varies the minimum cutoff (``fmin``) and the adaptive
gain (``k``).  A single cutoff is shared by all three coordinates and is set
from the normalized magnitude of the low-pass-filtered derivative:

    cutoff = fmin + k * ||v_lowpass|| / speed_scale

The derivative is always low-pass filtered at 1 Hz.  Positions are then
filtered with the candidate cutoff, starting from the first input position.
The returned velocity is the difference of the final two filtered positions
divided by the actual time step between them.
"""
from __future__ import annotations

import numpy as np

__all__ = ["filtered_velocities"]

FMINS = (0.5, 1.0, 2.0, 4.0)
KS = (0.0, 0.5, 1.0, 2.0)
DERIVATIVE_CUTOFF_HZ = 1.0


def _lowpass_alpha(cutoff_hz, dt):
    """First-order One Euro smoothing factor for a cutoff and time step."""
    tau = 1.0 / (2.0 * np.pi * cutoff_hz)
    return dt / (dt + tau)


def filtered_velocities(t, x, speed_scale):
    """Return the One Euro velocity grid for a position history.

    Parameters
    ----------
    t : array_like, shape (N,)
        Increasing sample times in seconds.
    x : array_like, shape (N, 3)
        Position samples.
    speed_scale : float
        Positive speed normalization, normally the source median speed.

    Returns
    -------
    dict
        ``{(fmin, k): velocity}`` with 16 entries, where ``velocity`` is a
        float array of shape ``(3,)``.
    """
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)

    if t.ndim != 1:
        raise ValueError("t must be one-dimensional")
    if x.ndim != 2 or x.shape[1] != 3:
        raise ValueError("x must have shape (N, 3)")
    if t.shape[0] != x.shape[0]:
        raise ValueError("t and x must contain the same number of samples")
    if t.size == 0:
        raise ValueError("at least one sample is required")

    scale = float(speed_scale)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("speed_scale must be finite and positive")

    keys = [(fmin, k) for fmin in FMINS for k in KS]
    dt = np.diff(t)
    if t.size > 1 and not np.all(dt > 0.0):
        raise ValueError("t must be strictly increasing")
    if t.size < 2:
        return {key: np.zeros(3, dtype=float) for key in keys}

    # Low-pass the raw derivative once at the fixed 1 Hz derivative cutoff.
    derivative = np.zeros_like(x)
    derivative[0] = 0.0
    for i in range(1, t.size):
        raw_derivative = (x[i] - x[i - 1]) / dt[i - 1]
        alpha = _lowpass_alpha(DERIVATIVE_CUTOFF_HZ, dt[i - 1])
        derivative[i] = derivative[i - 1] + alpha * (
            raw_derivative - derivative[i - 1]
        )

    normalized_speed = np.linalg.norm(derivative, axis=1) / scale

    # One shared adaptive cutoff per candidate at each sample.
    fmin_grid = np.asarray(FMINS, dtype=float)[:, None]
    k_grid = np.asarray(KS, dtype=float)[None, :]
    cutoffs = fmin_grid + k_grid * normalized_speed[1:, None, None]
    cutoffs = cutoffs.reshape(t.size - 1, len(keys))

    # Filter all candidates together, advancing one sample at a time.
    alphas = _lowpass_alpha(cutoffs, dt[:, None])
    positions = np.empty((t.size, len(keys), 3), dtype=float)
    positions[0] = x[0]
    for i in range(1, t.size):
        positions[i] = positions[i - 1] + alphas[i - 1, :, None] * (
            x[i] - positions[i - 1]
        )

    velocities = (positions[-1] - positions[-2]) / dt[-1]
    return {key: velocities[i].copy() for i, key in enumerate(keys)}
