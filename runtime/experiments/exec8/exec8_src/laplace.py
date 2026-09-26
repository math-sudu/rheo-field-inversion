"""Laplace posterior + credible-interval coverage check (stage 4,
post-gate rung of the frozen ladder).

Laplace approximation at the MAP (weighted least squares with the
frozen per-channel noise): over the FREE (machine-estimable)
coordinates in raw units,

    Sigma_post = (J_w^T J_w)^{-1},   J_w[i, a] = (1/sigma_i) d m_i /
    d theta_a  evaluated at theta_hat (through-system AD Jacobian).

No prior term: the box constraints act as bounds, not Gaussian priors
(the fixed coordinates ARE the strong-prior limit, declared).  The
frozen coverage gates (kickoff freeze section 5): over the 32
replicates of each truth, the 95% interval (z = 1.960) must cover the
truth in >= 28/32 replicates and the 68% interval (z = 1.000) in
17..27/32 (binomial 2-sigma bands).
"""

from __future__ import annotations

import numpy as np

from .identify import jacobian
from .observe import ObservationOperator

Z68 = 1.0
Z95 = 1.959963984540054


def laplace_sigma(op: ObservationOperator, free: list[str],
                  theta_hat: dict[str, float]) -> np.ndarray:
    """Posterior covariance over the free coordinates (raw units)."""
    J = jacobian(op, free, theta_hat)
    sig = op.sigma_vector().numpy()
    Jw = J / sig[:, None]
    return np.linalg.inv(Jw.T @ Jw)


def interval_covers(theta_hat: float, sd: float, truth: float,
                    z: float) -> bool:
    return abs(theta_hat - truth) <= z * sd


def coverage_counts(hats: np.ndarray, sds: np.ndarray,
                    truth: float) -> tuple[int, int]:
    """(n_cover_68, n_cover_95) over replicates."""
    c68 = int(np.sum(np.abs(hats - truth) <= Z68 * sds))
    c95 = int(np.sum(np.abs(hats - truth) <= Z95 * sds))
    return c68, c95


def coverage_gates(n68: int, n95: int, m: int = 32) -> tuple[str, str]:
    """Frozen gates: 95%: >= 28/32; 68%: 17..27/32."""
    g95 = "pass" if n95 >= 28 else "fail"
    g68 = "pass" if 17 <= n68 <= 27 else "fail"
    return g68, g95
