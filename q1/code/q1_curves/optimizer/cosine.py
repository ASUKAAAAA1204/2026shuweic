"""Numpy trainer for a cosine-basis MLP with a scalar regression objective."""

from __future__ import annotations

import numpy as np

__all__ = ["fit"]

_HIDDEN_UNITS = 16
_EPOCHS = 60
_ALPHA = 0.01
_MAX_BATCH_SIZE = 256
_LEARNING_RATE = 0.003
_EPSILON = 1e-8
_BETA1 = 0.9
_BETA2 = 0.999


def _glorot_uniform(rng, fan_in, fan_out):
    """Match the baseline's uniform initialization for both weights and biases."""
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return rng.uniform(-limit, limit, size=(fan_in, fan_out)), rng.uniform(-limit, limit, fan_out)


def _adam_update(
    parameter: np.ndarray,
    gradient: np.ndarray,
    first_moment: np.ndarray,
    second_moment: np.ndarray,
    step: int,
) -> None:
    """Apply one Adam update to ``parameter`` in place."""
    first_moment *= _BETA1
    first_moment += (1.0 - _BETA1) * gradient
    second_moment *= _BETA2
    second_moment += (1.0 - _BETA2) * gradient * gradient

    rate = _LEARNING_RATE * np.sqrt(1.0 - _BETA2**step) / (1.0 - _BETA1**step)
    parameter -= rate * first_moment / (np.sqrt(second_moment) + _EPSILON)


def _validate_inputs(
    x: np.ndarray, basis: np.ndarray, y: np.ndarray
) -> tuple[int, int]:
    """Validate array shapes and return ``(n_samples, n_basis)``."""
    if x.ndim != 2 or x.shape[1] != 4:
        raise ValueError("x must have shape (n, 4)")
    if basis.ndim != 2:
        raise ValueError("basis must have shape (n, p)")
    if y.ndim != 1:
        raise ValueError("y must have shape (n,)")

    n_samples = x.shape[0]
    if n_samples == 0:
        raise ValueError("x, basis, and y must contain at least one row")
    if basis.shape[0] != n_samples:
        raise ValueError("basis must have the same number of rows as x")
    if y.shape[0] != n_samples:
        raise ValueError("y must have the same length as the number of rows in x")
    if basis.shape[1] == 0:
        raise ValueError("basis must have at least one column")

    return n_samples, basis.shape[1]


def fit(x, basis, y, seed):
    """Train a one-hidden-layer MLP on a cosine basis.

    The network maps each four-feature input row to a vector of ``p`` cosine
    coefficients.  A prediction is the dot product of that coefficient vector
    with the corresponding row of ``basis``, so the regression target is a
    single scalar per row.

    Parameters
    ----------
    x : array-like of shape (n, 4)
        Standardized input features.
    basis : array-like of shape (n, p)
        Cosine features for each row.
    y : array-like of shape (n,)
        Scalar targets.
    seed : int
        Seed for the :class:`numpy.random.RandomState` used for initialization
        and epoch shuffling.

    Returns
    -------
    W1, b1, W2, b2 : tuple of numpy.ndarray
        Hidden weights/biases with shapes ``(4, 16)`` and ``(16,)``, and
        output weights/biases with shapes ``(16, p)`` and ``(p,)``.

    Notes
    -----
    The optimized objective for a minibatch of ``m`` rows is

    ``(0.5 * sum(residual**2) + 0.5 * alpha * sum(weight**2)) / m``,

    where the L2 penalty is applied to the weight matrices but not the biases.
    Training uses exactly 60 epochs of Adam with a constant step size.
    """
    x = np.asarray(x, dtype=np.float64)
    basis = np.asarray(basis, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)

    n_samples, n_basis = _validate_inputs(x, basis, y)

    rng = np.random.RandomState(seed)

    W1, b1 = _glorot_uniform(rng, x.shape[1], _HIDDEN_UNITS)
    W2, b2 = _glorot_uniform(rng, _HIDDEN_UNITS, n_basis)

    W1_moment = np.zeros_like(W1)
    b1_moment = np.zeros_like(b1)
    W2_moment = np.zeros_like(W2)
    b2_moment = np.zeros_like(b2)

    W1_velocity = np.zeros_like(W1)
    b1_velocity = np.zeros_like(b1)
    W2_velocity = np.zeros_like(W2)
    b2_velocity = np.zeros_like(b2)

    batch_size = min(_MAX_BATCH_SIZE, n_samples)
    step = 0

    order = np.arange(n_samples)
    for _ in range(_EPOCHS):
        order = rng.permutation(order)

        for start in range(0, n_samples, batch_size):
            rows = order[start : start + batch_size]
            n_batch = rows.size

            x_batch = x[rows]
            basis_batch = basis[rows]
            y_batch = y[rows]

            hidden = np.tanh(x_batch @ W1 + b1)
            coefficients = hidden @ W2 + b2
            prediction = np.einsum("ij,ij->i", coefficients, basis_batch)
            residual = prediction - y_batch

            # d(objective)/d(coefficients) for the scalar-per-row loss.
            output_gradient = (residual / n_batch)[:, None] * basis_batch
            hidden_gradient = (output_gradient @ W2.T) * (1.0 - hidden * hidden)

            W1_gradient = x_batch.T @ hidden_gradient + (_ALPHA / n_batch) * W1
            b1_gradient = np.sum(hidden_gradient, axis=0)
            W2_gradient = hidden.T @ output_gradient + (_ALPHA / n_batch) * W2
            b2_gradient = np.sum(output_gradient, axis=0)

            step += 1
            _adam_update(W1, W1_gradient, W1_moment, W1_velocity, step)
            _adam_update(b1, b1_gradient, b1_moment, b1_velocity, step)
            _adam_update(W2, W2_gradient, W2_moment, W2_velocity, step)
            _adam_update(b2, b2_gradient, b2_moment, b2_velocity, step)

    return W1, b1, W2, b2
