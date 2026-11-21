"""Deterministic three-state diagonal-Gaussian HMM for motion sequences.

The implementation targets a collection of independent sequences.  It learns
per-feature standardization from those sequences only, fits a three-state
first-order Gaussian HMM by batched Baum-Welch, and provides filtered and
final-observation-only state posteriors.

Fallbacks
---------
The required deterministic initialization is a diagonal scikit-learn
``GaussianMixture`` with ``random_state=seed`` and ``n_init=1``.  If that
initializer cannot be used (for example, there are fewer observations than
states, or it produces non-finite parameters), emissions are initialized
deterministically from an SVD projection with an equal-count three-way split.
A constant feature receives unit standardization scale because it carries no
emission information after centering.
"""

from __future__ import annotations

import json
import math
from typing import Any, Optional, Sequence

import numpy as np
from sklearn.mixture import GaussianMixture

__all__ = ["MotionHMM"]

_LOG_2PI = math.log(2.0 * math.pi)


class MotionHMM:
    """Three-state first-order HMM with diagonal Gaussian observations."""

    N_STATES = 3
    VARIANCE_FLOOR = 0.05
    MAX_ITER = 50
    TOL = 1e-5
    INITIAL_PSEUDOCOUNT = 1e-3
    TRANSITION_PSEUDOCOUNT = 1e-3

    def __init__(
        self,
        mean: np.ndarray,
        scale: np.ndarray,
        initial: np.ndarray,
        transition: np.ndarray,
        emission_means: np.ndarray,
        emission_vars: np.ndarray,
        iterations: int,
        seed: int,
    ) -> None:
        self.mean = np.asarray(mean, dtype=np.float64)
        self.scale = np.asarray(scale, dtype=np.float64)
        self.initial = np.asarray(initial, dtype=np.float64)
        self.transition = np.asarray(transition, dtype=np.float64)
        self.emission_means = np.asarray(emission_means, dtype=np.float64)
        self.emission_vars = np.asarray(emission_vars, dtype=np.float64)
        self.iterations = int(iterations)
        self.seed = int(seed)
        self.n_features = int(self.mean.size)
        self.n_states = int(self.N_STATES)

        # Conventional aliases for downstream callers.
        self.priors = self.initial
        self.means = self.emission_means
        self.variances = self.emission_vars
        self.em_iterations = self.iterations

    # ------------------------------------------------------------------
    # Validation and fitting
    # ------------------------------------------------------------------
    @staticmethod
    def _as_sequence_array(sequence: Any) -> np.ndarray:
        """Validate one sequence and return a float64 ``(time, features)`` array."""
        arr = np.asarray(sequence)
        if arr.ndim != 2:
            raise ValueError(
                "each sequence must be a two-dimensional (time, features) array"
            )
        if arr.shape[0] == 0 or arr.shape[1] == 0:
            raise ValueError("sequences must be non-empty in time and features")
        if np.iscomplexobj(arr):
            raise ValueError("complex-valued sequences are not supported")
        try:
            arr = arr.astype(np.float64, copy=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("sequences must contain numeric values") from exc
        if not np.all(np.isfinite(arr)):
            raise ValueError("sequences must contain only finite values")
        return arr

    @classmethod
    def _validate_sequences(cls, sequences: Any) -> list[np.ndarray]:
        """Materialize and validate a non-empty list of 2-D sequences."""
        if isinstance(sequences, np.ndarray):
            # A single (time, features) array is accepted as one sequence;
            # a 3-D array is treated as a stack of sequences.
            if sequences.ndim == 2:
                raw = [sequences]
            elif sequences.ndim == 3:
                raw = [sequences[i] for i in range(sequences.shape[0])]
            else:
                raise ValueError(
                    "sequences must be a list of (time, features) arrays"
                )
        else:
            try:
                raw = list(sequences)
            except TypeError as exc:
                raise ValueError("sequences must be an iterable of arrays") from exc

        if not raw:
            raise ValueError("at least one sequence is required")

        arrays = [cls._as_sequence_array(item) for item in raw]
        n_features = int(arrays[0].shape[1])
        for arr in arrays[1:]:
            if arr.shape[1] != n_features:
                raise ValueError("all sequences must have the same feature count")
        return arrays

    @classmethod
    def fit(
        cls, sequences: Sequence[np.ndarray], seed: int = 20261120
    ) -> "MotionHMM":
        """Fit the HMM by batched Baum-Welch.

        Equal-length sequences are stacked directly.  Unequal lengths share
        one padded time axis; invalid slots are masked and contribute zero
        responsibilities, so each EM iteration still uses whole-batch NumPy
        operations rather than looping over individual sequences.
        """
        arrays = cls._validate_sequences(sequences)
        n_sequences = len(arrays)
        n_features = int(arrays[0].shape[1])
        lengths = np.asarray([arr.shape[0] for arr in arrays], dtype=np.int64)
        total_observations = int(lengths.sum())

        # Standardization is learned solely from the supplied sequences.
        stacked = np.concatenate(arrays, axis=0)
        mean = np.asarray(stacked.mean(axis=0), dtype=np.float64)
        scale = np.asarray(stacked.std(axis=0), dtype=np.float64)
        # Constant features cannot be standardized; a unit scale leaves them
        # centered at zero.  The emission model then treats them as noise.
        scale = np.where(np.isfinite(scale) & (scale > 0.0), scale, 1.0)
        standardized = [(arr - mean) / scale for arr in arrays]

        max_length = int(lengths.max())
        if np.all(lengths == lengths[0]):
            batch = np.stack(standardized, axis=0)
            valid: Optional[np.ndarray] = None
        else:
            batch = np.zeros(
                (n_sequences, max_length, n_features), dtype=np.float64
            )
            valid = np.zeros((n_sequences, max_length), dtype=bool)
            for i, arr in enumerate(standardized):
                batch[i, : arr.shape[0], :] = arr
                valid[i, : arr.shape[0]] = True

        init_means, init_vars = cls._initial_emissions(batch, valid, seed)
        emission_means = np.asarray(init_means, dtype=np.float64)
        emission_vars = np.maximum(
            np.asarray(init_vars, dtype=np.float64), cls.VARIANCE_FLOOR
        )

        initial = np.full(cls.N_STATES, 1.0 / cls.N_STATES, dtype=np.float64)
        transition = np.full(
            (cls.N_STATES, cls.N_STATES),
            1.0 / cls.N_STATES,
            dtype=np.float64,
        )

        previous_log_likelihood: Optional[float] = None
        iterations = 0

        for iteration in range(1, cls.MAX_ITER + 1):
            gamma, xi_sum, log_likelihood = cls._e_step(
                batch,
                lengths,
                valid,
                initial,
                transition,
                emission_means,
                emission_vars,
            )

            # M-step with the requested small smoothing pseudocounts.
            initial = gamma[:, 0, :].sum(axis=0) + cls.INITIAL_PSEUDOCOUNT
            initial /= initial.sum()

            transition = xi_sum + cls.TRANSITION_PSEUDOCOUNT
            transition /= transition.sum(axis=1, keepdims=True)

            state_mass = gamma.sum(axis=(0, 1))
            safe_mass = np.where(state_mass > 1e-12, state_mass, 1.0)
            new_means = np.einsum("nts,ntf->sf", gamma, batch)
            new_means /= safe_mass[:, None]

            # E[x x^T] - mean mean^T is memory-friendlier than materializing
            # an (N,T,S,F) difference tensor on every iteration.
            second_moment = np.einsum("nts,ntf->sf", gamma, batch * batch)
            new_vars = second_moment / safe_mass[:, None]
            new_vars -= new_means * new_means
            new_vars = np.maximum(new_vars, cls.VARIANCE_FLOOR)

            empty = state_mass <= 1e-12
            if np.any(empty):
                new_means[empty] = emission_means[empty]
                new_vars[empty] = np.maximum(
                    emission_vars[empty], cls.VARIANCE_FLOOR
                )

            emission_means = new_means
            emission_vars = new_vars
            iterations = iteration

            if previous_log_likelihood is not None:
                delta = abs(log_likelihood - previous_log_likelihood) / max(
                    total_observations, 1
                )
                if np.isfinite(delta) and delta <= cls.TOL:
                    break
            previous_log_likelihood = log_likelihood

        return cls(
            mean=mean,
            scale=scale,
            initial=initial,
            transition=transition,
            emission_means=emission_means,
            emission_vars=emission_vars,
            iterations=iterations,
            seed=seed,
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def _prepare(self, sequence: Any) -> np.ndarray:
        arr = self._as_sequence_array(sequence)
        if arr.shape[1] != self.n_features:
            raise ValueError(
                f"expected {self.n_features} features, got {arr.shape[1]}"
            )
        return (arr - self.mean) / self.scale

    def filtered(self, sequence: np.ndarray) -> np.ndarray:
        """Return the final forward-filtered state posterior, shape ``(3,)``.

        The recursion consumes observations in order and never uses a future
        observation.  The result is the posterior over the state emitting the
        final observed time point.
        """
        x = self._prepare(sequence)
        emission_log_prob = self._emission_log_prob_batch(
            x[None, :, :], self.emission_means, self.emission_vars
        )[0]

        log_initial = np.log(np.maximum(self.initial, 1e-300))
        log_transition = np.log(np.maximum(self.transition, 1e-300))
        log_alpha = log_initial + emission_log_prob[0]
        for t in range(1, x.shape[0]):
            log_alpha = self._logsumexp(
                log_alpha[:, None] + log_transition, axis=0
            )
            log_alpha += emission_log_prob[t]

        log_alpha -= np.max(log_alpha)
        posterior = np.exp(log_alpha)
        posterior /= posterior.sum()
        return np.asarray(posterior, dtype=np.float64)

    def independent(self, sequence: np.ndarray) -> np.ndarray:
        """Return a final-observation-only posterior, shape ``(3,)``.

        This baseline uses the shared emission parameters and the learned HMM
        start prior, but conditions only on the last observation.  It ignores
        all earlier observations and the transition matrix.
        """
        x = self._prepare(sequence)
        log_joint = np.log(np.maximum(self.initial, 1e-300))
        log_joint += self._emission_log_prob_single(
            x[-1], self.emission_means, self.emission_vars
        )
        log_joint -= np.max(log_joint)
        posterior = np.exp(log_joint)
        posterior /= posterior.sum()
        return np.asarray(posterior, dtype=np.float64)

    def parameters(self) -> dict[str, Any]:
        """Return JSON-serializable parameters and fit metadata."""
        transition = self.transition.tolist()
        means = self.emission_means.tolist()
        variances = self.emission_vars.tolist()
        priors = self.initial.tolist()
        return {
            "states": int(self.N_STATES),
            "n_states": int(self.N_STATES),
            "seed": int(self.seed),
            "standardization": {
                "mean": self.mean.tolist(),
                "scale": self.scale.tolist(),
                "std": self.scale.tolist(),
            },
            # Primary field names.
            "priors": priors,
            "transition": transition,
            "emission_means": means,
            "emission_variances": variances,
            "em_iterations": int(self.iterations),
            # Common aliases, kept in sync for downstream consumers.
            "start_priors": list(priors),
            "initial_priors": list(priors),
            "initial": list(priors),
            "transitions": [list(row) for row in transition],
            "transition_rows": [list(row) for row in transition],
            "means": [list(row) for row in means],
            "emission_vars": [list(row) for row in variances],
            "variances": [list(row) for row in variances],
            "iterations": int(self.iterations),
            "n_iter": int(self.iterations),
            "emission": {
                "means": [list(row) for row in means],
                "variances": [list(row) for row in variances],
                "covariance": "diagonal",
                "variance_floor": float(self.VARIANCE_FLOOR),
            },
            "variance_floor": float(self.VARIANCE_FLOOR),
            "max_em_iterations": int(self.MAX_ITER),
            "tolerance": float(self.TOL),
        }

    # ------------------------------------------------------------------
    # Numerical helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _logsumexp(values: np.ndarray, axis: int) -> np.ndarray:
        maximum = np.max(values, axis=axis, keepdims=True)
        finite = np.isfinite(maximum)
        safe_maximum = np.where(finite, maximum, 0.0)
        out = safe_maximum + np.log(
            np.sum(np.exp(values - safe_maximum), axis=axis, keepdims=True)
        )
        out = np.where(finite, out, -np.inf)
        return np.squeeze(out, axis=axis)

    @classmethod
    def _emission_log_prob_batch(
        cls,
        batch: np.ndarray,
        means: np.ndarray,
        variances: np.ndarray,
    ) -> np.ndarray:
        """Log diagonal-Gaussian densities with shape ``(N, T, S)``."""
        diff = batch[:, :, None, :] - means[None, None, :, :]
        log_norm = -0.5 * (
            batch.shape[2] * _LOG_2PI + np.log(variances).sum(axis=1)
        )
        log_exp = -0.5 * np.einsum(
            "ntsf,sf->nts", diff * diff, 1.0 / variances
        )
        return log_norm[None, None, :] + log_exp

    @staticmethod
    def _emission_log_prob_single(
        x: np.ndarray, means: np.ndarray, variances: np.ndarray
    ) -> np.ndarray:
        """Log diagonal-Gaussian density for one observation, shape ``(S,)``."""
        diff = x[None, :] - means
        log_norm = -0.5 * (
            means.shape[1] * _LOG_2PI + np.log(variances).sum(axis=1)
        )
        return log_norm - 0.5 * np.sum(diff * diff / variances, axis=1)

    @classmethod
    def _e_step(
        cls,
        batch: np.ndarray,
        lengths: np.ndarray,
        valid: Optional[np.ndarray],
        initial: np.ndarray,
        transition: np.ndarray,
        means: np.ndarray,
        variances: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """One scaled forward-backward E-step over the whole batch."""
        n_sequences, max_length, _ = batch.shape
        log_emission = cls._emission_log_prob_batch(batch, means, variances)

        # Shift each time slice by its maximum before exponentiating.  The
        # shift is restored in the forward log-likelihood below; posterior
        # quantities are invariant to it.
        emission_shift = np.max(log_emission, axis=2)
        emission = np.exp(log_emission - emission_shift[:, :, None])

        if valid is None:
            valid = np.ones((n_sequences, max_length), dtype=bool)

        alpha = np.zeros(
            (n_sequences, max_length, cls.N_STATES), dtype=np.float64
        )
        scales = np.ones((n_sequences, max_length), dtype=np.float64)
        log_likelihood = 0.0

        first = initial[None, :] * emission[:, 0, :]
        first_scale = first.sum(axis=1)
        first_safe = np.where(first_scale > 0.0, first_scale, 1.0)
        alpha[:, 0, :] = first / first_safe[:, None]
        scales[:, 0] = first_safe
        log_likelihood += float(
            np.sum(np.log(first_safe) + emission_shift[:, 0])
        )

        for t in range(1, max_length):
            previous = alpha[:, t - 1, :] @ transition
            current = previous * emission[:, t, :]
            current *= valid[:, t, None]
            scale = current.sum(axis=1)
            safe_scale = np.where(scale > 0.0, scale, 1.0)
            alpha[:, t, :] = current / safe_scale[:, None]
            scales[:, t] = safe_scale
            log_likelihood += float(
                np.sum(
                    np.where(
                        valid[:, t],
                        np.log(safe_scale) + emission_shift[:, t],
                        0.0,
                    )
                )
            )

        # Backward pass.  Beta is normalized at each time because any
        # time-specific multiplicative factor cancels in gamma and xi.
        beta = np.zeros_like(alpha)
        beta[np.arange(n_sequences), lengths - 1, :] = 1.0
        for t in range(max_length - 2, -1, -1):
            future = beta[:, t + 1, :] * emission[:, t + 1, :]
            current = future @ transition.T
            current /= scales[:, t + 1, None]
            current = np.where(
                (lengths == t + 1)[:, None], 1.0, current
            )
            current = np.where((lengths > t)[:, None], current, 0.0)
            row_max = np.max(current, axis=1, keepdims=True)
            beta[:, t, :] = current / np.where(row_max > 0.0, row_max, 1.0)

        gamma = alpha * beta
        gamma_sum = gamma.sum(axis=2, keepdims=True)
        gamma = gamma / np.where(gamma_sum > 0.0, gamma_sum, 1.0)
        gamma *= valid[:, :, None]

        xi_sum = np.zeros(
            (cls.N_STATES, cls.N_STATES), dtype=np.float64
        )
        if max_length > 1:
            left = alpha[:, :-1, :, None]
            right = beta[:, 1:, None, :] * emission[:, 1:, None, :]
            numerator = left * transition[None, None, :, :] * right
            transition_valid = (
                np.arange(1, max_length)[None, :] < lengths[:, None]
            )
            numerator *= transition_valid[:, :, None, None]
            denominator = numerator.sum(axis=(2, 3), keepdims=True)
            numerator /= np.where(denominator > 0.0, denominator, 1.0)
            xi_sum = numerator.sum(axis=(0, 1))

        return gamma, xi_sum, float(log_likelihood)

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    @classmethod
    def _initial_emissions(
        cls,
        batch: np.ndarray,
        valid: Optional[np.ndarray],
        seed: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Initialize in standardized space, preferring the required GMM."""
        if valid is None:
            observations = batch.reshape(-1, batch.shape[2])
        else:
            observations = batch[valid]

        if observations.shape[0] >= cls.N_STATES:
            try:
                gmm = GaussianMixture(
                    n_components=cls.N_STATES,
                    covariance_type="diag",
                    random_state=int(seed),
                    n_init=1,
                    reg_covar=1e-6,
                )
                gmm.fit(observations)
                means = np.asarray(gmm.means_, dtype=np.float64)
                variances = np.asarray(gmm.covariances_, dtype=np.float64)
                if (
                    means.shape == (cls.N_STATES, batch.shape[2])
                    and variances.shape == (cls.N_STATES, batch.shape[2])
                    and np.all(np.isfinite(means))
                    and np.all(np.isfinite(variances))
                ):
                    return means, np.maximum(
                        variances, cls.VARIANCE_FLOOR
                    )
            except Exception:
                # The deterministic empirical fallback below is documented
                # in the module docstring.
                pass

        return cls._fallback_emissions(observations)

    @classmethod
    def _fallback_emissions(
        cls, observations: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Deterministic SVD/quantile split used when GMM initialization fails."""
        n_observations, n_features = observations.shape
        if n_observations == 0:
            raise ValueError("cannot initialize emissions without observations")

        global_mean = observations.mean(axis=0)
        global_var = observations.var(axis=0)
        global_var = np.where(global_var > 0.0, global_var, 1.0)

        centered = observations - global_mean
        direction: Optional[np.ndarray] = None
        if n_observations > 1 and np.any(centered != 0.0):
            try:
                _, _, vt = np.linalg.svd(centered, full_matrices=False)
                if vt.shape[0] > 0:
                    direction = np.asarray(vt[0], dtype=np.float64)
                    # Canonicalize the arbitrary SVD sign for determinism.
                    pivot = int(np.argmax(np.abs(direction)))
                    if direction[pivot] < 0.0:
                        direction = -direction
            except np.linalg.LinAlgError:
                direction = None

        if direction is None or not np.all(np.isfinite(direction)):
            scores = np.zeros(n_observations, dtype=np.float64)
        else:
            scores = centered @ direction

        order = np.argsort(scores, kind="stable")
        groups = np.array_split(order, cls.N_STATES)
        means = np.empty((cls.N_STATES, n_features), dtype=np.float64)
        variances = np.empty((cls.N_STATES, n_features), dtype=np.float64)

        for state, group in enumerate(groups):
            if group.size == 0:
                means[state] = global_mean
                variances[state] = global_var
            else:
                group_values = observations[group]
                means[state] = group_values.mean(axis=0)
                group_var = group_values.var(axis=0)
                variances[state] = np.where(
                    group_var > 0.0, group_var, global_var
                )

        return means, np.maximum(variances, cls.VARIANCE_FLOOR)


def _self_check() -> None:
    """Minimal synthetic checks required by the task."""
    sequences = [
        np.asarray(
            [
                [0.0, 0.2],
                [0.4, 0.1],
                [0.9, 0.5],
                [1.3, 0.6],
                [0.8, 0.0],
                [0.2, -0.3],
                [0.5, 0.2],
                [1.0, 0.4],
                [1.5, 0.8],
            ],
            dtype=np.float64,
        ),
        np.asarray(
            [
                [-1.0, -0.2],
                [-0.6, 0.1],
                [-0.2, 0.0],
                [0.2, 0.3],
                [0.7, 0.1],
                [1.1, 0.5],
                [0.6, 0.2],
            ],
            dtype=np.float64,
        ),
    ]

    model = MotionHMM.fit(sequences, seed=20261120)
    params = model.parameters()
    json.dumps(params)  # Must be JSON serializable.

    priors = np.asarray(params["priors"], dtype=np.float64)
    transition = np.asarray(params["transition"], dtype=np.float64)
    means = np.asarray(params["emission_means"], dtype=np.float64)
    variances = np.asarray(params["emission_variances"], dtype=np.float64)

    assert priors.shape == (3,)
    assert transition.shape == (3, 3)
    assert means.shape == (3, sequences[0].shape[1])
    assert variances.shape == (3, sequences[0].shape[1])
    assert np.allclose(priors.sum(), 1.0, atol=1e-12)
    assert np.allclose(transition.sum(axis=1), 1.0, atol=1e-12)
    assert np.all(variances >= MotionHMM.VARIANCE_FLOOR - 1e-15)
    assert int(params["em_iterations"]) <= MotionHMM.MAX_ITER

    filtered = model.filtered(sequences[0])
    independent = model.independent(sequences[0])
    assert filtered.shape == (3,) and filtered.dtype == np.float64
    assert independent.shape == (3,) and independent.dtype == np.float64
    assert np.allclose(filtered.sum(), 1.0, atol=1e-12)
    assert np.allclose(independent.sum(), 1.0, atol=1e-12)

    # Compare the forward filter with explicit enumeration of all 3^3 state
    # paths for a short sequence, using this fitted model's parameters.
    short = sequences[0][:3]
    standardized = (short - model.mean) / model.scale
    emission = np.stack(
        [
            model._emission_log_prob_single(
                standardized[t], model.emission_means, model.emission_vars
            )
            for t in range(3)
        ],
        axis=0,
    )
    log_prior = np.log(np.maximum(model.initial, 1e-300))
    log_transition = np.log(np.maximum(model.transition, 1e-300))
    log_paths = np.empty((3, 3, 3), dtype=np.float64)
    for z0 in range(3):
        for z1 in range(3):
            for z2 in range(3):
                log_paths[z0, z1, z2] = (
                    log_prior[z0]
                    + emission[0, z0]
                    + log_transition[z0, z1]
                    + emission[1, z1]
                    + log_transition[z1, z2]
                    + emission[2, z2]
                )
    enumerated = np.exp(log_paths - np.max(log_paths))
    enumerated = enumerated.sum(axis=(0, 1))
    enumerated /= enumerated.sum()
    short_filtered = model.filtered(short)
    assert np.allclose(short_filtered, enumerated, atol=1e-10)

    print("MotionHMM self-check passed")


if __name__ == "__main__":
    _self_check()
