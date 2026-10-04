"""Numerical primitives shared by OG-PINN training and certification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping

import numpy as np


LOG_COORDS = frozenset(
    {"s_star", "r_K", "tau_K", "tau_M", "tau_vp",
     "r_c", "chi_c", "gamma_c", "gamma_phi"}
)


@dataclass(frozen=True)
class CoordinateMap:
    """Map physical box coordinates to a common unit interval.

    Logarithmic coordinates are mapped affinely after taking a logarithm;
    the remaining coordinates are mapped affinely in physical space.  The
    map is the bounded counterpart of the transforms used by the original
    Levenberg--Marquardt implementation.
    """

    box: Mapping[str, tuple[float, float]]
    log_coords: frozenset[str] = LOG_COORDS

    def _bounds(self, name: str) -> tuple[float, float]:
        lo, hi = (float(v) for v in self.box[name])
        if not np.isfinite(lo) or not np.isfinite(hi) or not lo < hi:
            raise ValueError(f"invalid bounds for {name}: {(lo, hi)}")
        if name in self.log_coords and lo <= 0.0:
            raise ValueError(f"log coordinate {name} must be positive")
        return lo, hi

    def to_unit(self, name: str, value: float) -> float:
        lo, hi = self._bounds(name)
        value = float(value)
        if name in self.log_coords:
            unit = (np.log(value) - np.log(lo)) / (np.log(hi) - np.log(lo))
        else:
            unit = (value - lo) / (hi - lo)
        return float(unit)

    def from_unit(self, name: str, unit: float, *, clip: bool = True) -> float:
        lo, hi = self._bounds(name)
        unit = float(np.clip(unit, 0.0, 1.0) if clip else unit)
        if name in self.log_coords:
            return float(np.exp(np.log(lo) + unit * (np.log(hi) - np.log(lo))))
        return float(lo + unit * (hi - lo))

    def dphysical_dunit(self, name: str, value: float) -> float:
        """Derivative of physical coordinate with respect to unit coordinate."""
        lo, hi = self._bounds(name)
        if name in self.log_coords:
            return float(value) * float(np.log(hi) - np.log(lo))
        return float(hi - lo)


@dataclass(frozen=True)
class FloorCandidate:
    """One constrained solve on a constitutive substep floor."""

    floor: float
    sse: float
    converged: bool
    payload: Any = None


@dataclass(frozen=True)
class TerminalKKT:
    """Scale-normalised first-order diagnostics on a unit box.

    The least-squares objective is written as ``0.5 * r.T @ r`` so that
    ``gradient = J.T @ r`` and the Gauss--Newton curvature is ``J.T @ J``.
    The projected mapping uses the Frobenius norm of that curvature, floored
    at one, as its dimensionless step scale.
    """

    feasibility_residual: float
    gradient: np.ndarray
    curvature: np.ndarray
    curvature_scale: float
    projected_mapping: np.ndarray
    projected_residual: float
    active_lower: np.ndarray
    active_upper: np.ndarray


def solve_floor_ladder(
    floors: Iterable[float],
    solve: Callable[[float], FloorCandidate],
) -> tuple[FloorCandidate, list[FloorCandidate]]:
    """Run every requested floor and retain the lowest finite objective.

    A shallow-floor result being locally converged is deliberately not a
    stopping condition: convergence certifies only that local solve, while a
    deeper substep floor can expose another basin.  This is the correction to
    the early-break behaviour that produced the former softening needles.
    """

    attempted: list[FloorCandidate] = []
    for floor in floors:
        candidate = solve(float(floor))
        if candidate.floor != float(floor):
            raise ValueError("solver returned a candidate for a different floor")
        attempted.append(candidate)
    finite = [row for row in attempted if np.isfinite(row.sse)]
    if not finite:
        raise RuntimeError("no substep floor returned a finite objective")
    return min(finite, key=lambda row: row.sse), attempted


def anchored_delta_sse(
    sses: Iterable[float],
    *,
    anchor_sse: float,
    atol: float = 1.0e-8,
) -> np.ndarray:
    """Return objective increments relative to a verified physical anchor.

    The anchor is included in the lower-envelope reference.  A computed value
    materially below a known noiseless zero anchor indicates an inconsistent
    objective and is rejected instead of silently re-basing the curve.
    """

    values = np.asarray(list(sses), dtype=float)
    if values.ndim != 1 or not len(values):
        raise ValueError("at least one objective value is required")
    if not np.isfinite(anchor_sse):
        raise ValueError("anchor_sse must be finite")
    if np.nanmin(values) < float(anchor_sse) - float(atol):
        raise ValueError("profile objective lies below its verified anchor")
    return values - float(anchor_sse)


def pin_profile_coordinate(
    starts: Iterable[Mapping[str, float]],
    *,
    coordinate: str,
    value: float,
) -> list[dict[str, float]]:
    """Copy proposals and impose the target profile coordinate.

    Continuation proposals may originate at a neighbouring grid node.  Their
    nuisance coordinates remain valid starts, but the stored profile value
    must not leak into a direct-objective diagnostic at the target node.
    """

    pinned = []
    for start in starts:
        target = {name: float(parameter) for name, parameter in start.items()}
        target[coordinate] = float(value)
        pinned.append(target)
    return pinned


def projected_gradient_mapping(
    unit: np.ndarray,
    gradient: np.ndarray,
    *,
    scale: float = 1.0,
) -> np.ndarray:
    """Bound-aware first-order residual for a unit-box problem.

    The mapping vanishes at an interior stationary point and at a bound that
    satisfies the corresponding KKT sign.  Scaling only fixes the trial step
    used by the mapping and must be positive.
    """

    unit = np.asarray(unit, dtype=float)
    gradient = np.asarray(gradient, dtype=float)
    if unit.shape != gradient.shape:
        raise ValueError("unit and gradient must have the same shape")
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("scale must be finite and positive")
    return unit - np.clip(unit - gradient / float(scale), 0.0, 1.0)


def terminal_box_kkt(
    unit: np.ndarray,
    weighted_residual: np.ndarray,
    weighted_jacobian_unit: np.ndarray,
    *,
    active_bound_tolerance: float = 1.0e-8,
) -> TerminalKKT:
    """Evaluate a curvature-scaled projected-KKT residual.

    ``weighted_jacobian_unit`` is the exact discrete residual Jacobian with
    respect to the free coordinates after mapping them to ``[0, 1]``.  The
    returned residual is the Euclidean norm of

    ``u - projection(u - (J.T @ r) / max(||J.T @ J||_F, 1))``.

    Feasibility is the largest coordinate violation of the unit box.  Active
    coordinates are classified geometrically and independently of whether
    their gradient has the KKT-consistent sign.
    """

    unit = np.asarray(unit, dtype=float)
    residual = np.asarray(weighted_residual, dtype=float)
    jacobian = np.asarray(weighted_jacobian_unit, dtype=float)
    if unit.ndim != 1 or residual.ndim != 1:
        raise ValueError("unit and weighted_residual must be one-dimensional")
    if jacobian.shape != (len(residual), len(unit)):
        raise ValueError("weighted Jacobian has incompatible shape")
    if not np.all(np.isfinite(unit)) or not np.all(np.isfinite(residual)) \
            or not np.all(np.isfinite(jacobian)):
        raise ValueError("terminal KKT inputs must be finite")
    if not np.isfinite(active_bound_tolerance) \
            or active_bound_tolerance < 0.0:
        raise ValueError("active-bound tolerance must be finite and nonnegative")

    lower_violation = np.maximum(-unit, 0.0)
    upper_violation = np.maximum(unit - 1.0, 0.0)
    feasibility = float(max(
        np.max(lower_violation, initial=0.0),
        np.max(upper_violation, initial=0.0),
    ))
    gradient = jacobian.T @ residual
    curvature = jacobian.T @ jacobian
    curvature_scale = max(float(np.linalg.norm(curvature, ord="fro")), 1.0)
    mapping = projected_gradient_mapping(
        unit,
        gradient,
        scale=curvature_scale,
    )
    return TerminalKKT(
        feasibility_residual=feasibility,
        gradient=gradient,
        curvature=curvature,
        curvature_scale=curvature_scale,
        projected_mapping=mapping,
        projected_residual=float(np.linalg.norm(mapping, ord=2)),
        active_lower=unit <= float(active_bound_tolerance),
        active_upper=unit >= 1.0 - float(active_bound_tolerance),
    )


def fixed_active_set_tangent(
    hessian: np.ndarray,
    cross: np.ndarray,
    unit: np.ndarray,
    gradient: np.ndarray,
    *,
    ridge_relative: float = 1.0e-8,
    bound_tolerance: float = 1.0e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve an implicit Gauss--Newton tangent on the KKT-free subspace."""

    hessian = np.asarray(hessian, dtype=float)
    cross = np.asarray(cross, dtype=float)
    unit = np.asarray(unit, dtype=float)
    gradient = np.asarray(gradient, dtype=float)
    n = len(unit)
    if hessian.shape != (n, n) or cross.shape != (n,) \
            or gradient.shape != (n,):
        raise ValueError("incompatible active-set tangent shapes")
    active_lower = (unit <= bound_tolerance) & (gradient >= 0.0)
    active_upper = (unit >= 1.0 - bound_tolerance) & (gradient <= 0.0)
    active = active_lower | active_upper
    free = ~active
    tangent = np.zeros(n, dtype=float)
    if np.any(free):
        reduced = hessian[np.ix_(free, free)]
        scale = max(float(np.trace(reduced)) / int(np.sum(free)), 1.0)
        regularized = (
            reduced + float(ridge_relative) * scale
            * np.eye(int(np.sum(free)))
        )
        tangent[free] = -np.linalg.solve(regularized, cross[free])
    return tangent, active
