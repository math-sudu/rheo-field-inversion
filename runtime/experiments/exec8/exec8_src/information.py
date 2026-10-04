"""Coordinate uncertainty with an explicit structural-nullspace check."""
from __future__ import annotations

import numpy as np


def coordinate_crb(jacobian, rtol=1e-10):
    """Return standard deviations; unidentifiable coordinates have no bound.

    A Moore-Penrose inverse describes the identified subspace only. Its
    diagonal is not a finite bound for a coordinate with a null component.
    The relative rank tolerance is recorded with the migrated protocol.
    """
    J = np.asarray(jacobian, dtype=float)
    _, singular, vt = np.linalg.svd(J, full_matrices=True)
    cutoff = rtol * (singular[0] if len(singular) else 0.0)
    rank = int(np.count_nonzero(singular > cutoff))
    identified = vt[:rank].T
    variance = np.sum((identified / singular[:rank]) ** 2, axis=1)
    null_weight = np.sum(vt[rank:] ** 2, axis=0)
    answer = np.sqrt(variance)
    answer[null_weight > rtol ** 2] = np.inf
    return answer


def fisher_condition(jacobian, rtol=1e-10):
    """Condition number of J.T @ J, infinite when numerically rank deficient."""
    J = np.asarray(jacobian, dtype=float)
    singular = np.linalg.svd(J, compute_uv=False)
    if (len(singular) < J.shape[1] or not len(singular)
            or singular[-1] <= rtol * singular[0]):
        return float("inf")
    return float((singular[0] / singular[-1]) ** 2)
