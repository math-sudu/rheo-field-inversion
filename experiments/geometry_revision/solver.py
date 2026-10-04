from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import time

import numpy as np
from scipy.optimize import lsq_linear

from experiments.fe_pinc.core import CoordinateMap, terminal_box_kkt


class InvalidEvaluation(RuntimeError):
    """A returned FE history or derivative cannot support an objective."""


@dataclass
class _State:
    theta: dict
    unit: np.ndarray
    values: np.ndarray
    history: object
    residual: np.ndarray
    jacobian: np.ndarray | None = None
    physical_jacobian: np.ndarray | None = None
    replay_error: float | None = None
    kkt: object | None = None

    @property
    def sse(self):
        return float(self.residual @ self.residual)


class _Problem:
    def __init__(self, ev, y, box, free, *, coordinate_map, replay_tolerance):
        self.ev = ev
        self.y = np.asarray(y, dtype=float)
        self.sigma = np.asarray(ev.sigma, dtype=float)
        self.box, self.free = dict(box), tuple(free)
        self.cmap = coordinate_map or CoordinateMap(box)
        self.replay_tolerance = float(replay_tolerance)
        self.n_forward = self.n_jacobian = 0
        if (self.y.ndim != 1 or not len(self.y) or self.sigma.shape != self.y.shape
                or not np.isfinite(self.y).all() or not np.isfinite(self.sigma).all()
                or np.any(self.sigma <= 0)):
            raise ValueError("Observations and positive noise scales must be finite aligned vectors")
        if len(set(self.free)) != len(self.free) or any(n not in box for n in self.free):
            raise ValueError("Free coordinates must be unique members of the supplied box")
        if not np.isfinite(self.replay_tolerance) or self.replay_tolerance <= 0:
            raise ValueError("Replay tolerance must be finite and positive")

    def unit_of(self, theta):
        if not all(np.isfinite(float(v)) for v in theta.values()):
            raise ValueError("All physical coordinates must be finite")
        for name in self.box.keys() & theta.keys():
            lo, hi = self.cmap._bounds(name)
            if name in self.cmap.log_coords and theta[name] <= 0:
                raise ValueError(f"Nonpositive logarithmic coordinate: {name}")
            unit = self.cmap.to_unit(name, theta[name])
            if not np.isfinite(unit) or unit < -1e-12 or unit > 1 + 1e-12:
                raise ValueError(f"Coordinate outside the supplied box: {name}={theta[name]}")
        return np.array([self.cmap.to_unit(n, theta[n]) for n in self.free])

    def theta_of(self, unit, fixed):
        theta = dict(fixed)
        theta.update({n: self.cmap.from_unit(n, float(v), clip=False)
                      for n, v in zip(self.free, unit)})
        return theta

    def forward(self, theta):
        unit = self.unit_of(theta)
        self.n_forward += 1
        values, history = self.ev.model(dict(theta))
        values = np.asarray(values, dtype=float)
        if values.shape != self.y.shape or not np.isfinite(values).all():
            raise InvalidEvaluation("FE model returned nonfinite or misaligned observations")
        executed = getattr(history, "executed", None)
        if executed is None or not len(executed):
            raise InvalidEvaluation("FE model returned no executed increment history")
        if not all(bool(ex.trial.converged) for ex in executed):
            raise InvalidEvaluation("FE model returned an uncommitted or unconverged increment")
        return _State(dict(theta), unit, values, history, (values - self.y) / self.sigma)

    def differentiate(self, state):
        if state.jacobian is not None:
            return state
        self.n_jacobian += 1
        replay, physical = self.ev.jacobian(state.history, dict(state.theta), self.free)
        replay, physical = np.asarray(replay, dtype=float), np.asarray(physical, dtype=float)
        if (replay.shape != self.y.shape or physical.shape != (len(self.y), len(self.free))
                or not np.isfinite(replay).all() or not np.isfinite(physical).all()):
            raise InvalidEvaluation("FE replay returned nonfinite or misaligned derivatives")
        error = float(np.max(np.abs(replay - state.values), initial=0.0))
        if error > self.replay_tolerance:
            raise InvalidEvaluation(f"FE replay differs from its forward by {error:.9g} mm")
        chain = np.array([self.cmap.dphysical_dunit(n, state.theta[n]) for n in self.free])
        # This is the full physical derivative, including inward derivatives at
        # a box face. There is no derivative of a clamp or logistic transform.
        state.physical_jacobian = physical
        state.jacobian = physical / self.sigma[:, None] * chain[None, :]
        state.replay_error = error
        state.kkt = terminal_box_kkt(state.unit, state.residual, state.jacobian)
        return state

    def terminal(self, theta, *, kkt_tolerance, feasibility_tolerance):
        # A fresh forward and replay independently check the proposed endpoint;
        # optimizer cached histories/Jacobians cannot serve as the certificate.
        state = self.differentiate(self.forward(theta))
        kkt = state.kkt
        units = {n: self.cmap.to_unit(n, theta[n]) for n in self.box if n in theta}
        feasibility = max(kkt.feasibility_residual,
                          max((max(-v, v - 1, 0.) for v in units.values()), default=0.))
        active = []
        for i, name in enumerate(self.free):
            if not (kkt.active_lower[i] or kkt.active_upper[i]):
                continue
            lower = bool(kkt.active_lower[i])
            gradient = float(kkt.gradient[i])
            active.append({"coordinate": name, "side": "lower" if lower else "upper",
                           "unit_value": float(state.unit[i]),
                           "half_objective_gradient": gradient,
                           "kkt_sign_satisfied": gradient >= 0 if lower else gradient <= 0})
        eligible = feasibility <= feasibility_tolerance and kkt.projected_residual <= kkt_tolerance
        return {"terminal_theta": dict(state.theta), "free_nuisance_coordinates": list(self.free),
                "unit_nuisance_coordinates": dict(zip(self.free, state.unit.tolist())),
                "unit_box_coordinates": units, "exact_objective": state.sse,
                "wrmse": float(np.sqrt(state.sse / len(self.y))),
                "feasibility_residual": float(feasibility),
                "projected_kkt_residual": kkt.projected_residual,
                "projected_mapping": kkt.projected_mapping.tolist(),
                "curvature_scale": kkt.curvature_scale,
                "half_objective_gradient": kkt.gradient.tolist(),
                "active_coordinates": active, "replay_deviation": state.replay_error,
                "all_increments_converged": True, "executed_increments": len(state.history.executed),
                "predictions_mm": state.values.tolist(),
                "physical_jacobian": state.physical_jacobian.tolist(),
                "eligible": bool(eligible), "kkt_tolerance": float(kkt_tolerance),
                "feasibility_tolerance": float(feasibility_tolerance),
                "verification": "fresh FE forward and physical-coordinate replay; local first-order condition"}


def evaluate_terminal(ev, y, box, free, theta, *, coordinate_map=None,
                      kkt_tolerance=1e-6, feasibility_tolerance=1e-10,
                      replay_tolerance=1e-8):
    """Independently check a proposed physical endpoint; never clip its inputs."""
    problem = _Problem(ev, y, box, free, coordinate_map=coordinate_map,
                       replay_tolerance=replay_tolerance)
    return problem.terminal(theta, kkt_tolerance=kkt_tolerance,
                            feasibility_tolerance=feasibility_tolerance)


def _bounded_step(state, radius, damping):
    """Solve the small damped GN subproblem with its actual step bounds."""
    n = len(state.unit)
    if not n:
        return np.empty(0)
    diagonal = np.maximum(np.sum(state.jacobian ** 2, axis=0), state.kkt.curvature_scale * 1e-10)
    matrix = np.vstack((state.jacobian, np.diag(np.sqrt(damping * diagonal))))
    rhs = np.concatenate((-state.residual, np.zeros(n)))
    lower = np.maximum(-state.unit, -radius)
    upper = np.minimum(1 - state.unit, radius)
    # BVLS handles a changing active set; a coordinate at a face may move back
    # into the box as soon as its coupled descent direction requires that.
    return lsq_linear(matrix, rhs, bounds=(lower, upper), method="bvls",
                      tol=1e-12, max_iter=100).x


def solve_multistart(ev, y, box, free, fixed, starts, *, failure_type=(),
                     coordinate_map=None, max_nfev=160, max_iterations=100,
                     solve_tolerance=1e-8, kkt_tolerance=1e-6,
                     feasibility_tolerance=1e-10, replay_tolerance=1e-8,
                     initial_radius=.25, checkpoint=None):
    """Run an explicit finite local pool and independently verify every endpoint.

    ``starts`` contains physical coordinate dictionaries. ``fixed`` overrides
    nonfree values in those proposals, as required at a constrained profile
    node. All free coordinates must be present. Expected FE exceptions are
    specified by ``failure_type``; unrelated programming errors propagate.

    The result is JSON-serializable. ``selected_index`` is set only for an
    eligible endpoint with no lower unresolved candidate in the declared pool.
    It is a local first-order result, never a global minimum certificate.
    ``max_nfev`` includes failed optimization trials; one additional fresh
    forward per valid start is reserved for endpoint verification.
    """
    free, fixed, starts = tuple(free), dict(fixed), list(starts)
    if set(free) & fixed.keys():
        raise ValueError("A coordinate cannot be both free and fixed")
    if not starts:
        raise ValueError("At least one explicit start is required")
    if max_nfev < 1 or max_iterations < 0 or not 0 < initial_radius <= 1:
        raise ValueError("Invalid evaluation budget, iteration budget, or initial radius")
    if not 0 < solve_tolerance <= kkt_tolerance or feasibility_tolerance < 0:
        raise ValueError("Invalid stationarity or feasibility tolerances")
    failures = (failure_type,) if isinstance(failure_type, type) else tuple(failure_type)
    if any(not isinstance(cls, type) or not issubclass(cls, Exception) for cls in failures):
        raise TypeError("failure_type must contain exception classes")
    failures = failures + (InvalidEvaluation,)
    problem = _Problem(ev, y, box, free, coordinate_map=coordinate_map,
                       replay_tolerance=replay_tolerance)
    report = {"schema_version": 1, "method": "bounded GN trust region with exact FE trial rejection",
              "free": list(free), "fixed": fixed, "box": {n: list(v) for n, v in box.items()},
              "scope": "explicit finite local multistart pool; independent first-order endpoint checks",
              "candidates": [], "selected_index": None}

    def emit(event, row=None):
        if checkpoint is not None:
            checkpoint(deepcopy({"event": event, "candidate": row, "report": report}))

    for index, start in enumerate(starts):
        began = time.perf_counter()
        nf0, nj0 = problem.n_forward, problem.n_jacobian
        theta = dict(fixed, **{n: float(start[n]) for n in free})
        problem.unit_of(theta)  # Invalid inputs are errors, never clipped starts.
        row = {"start_index": index, "start_theta": theta, "iterations": [], "failed_trials": [],
               "status": "running", "terminal": None}
        report["candidates"].append(row)
        emit("start", row)
        current = None
        reason = "iteration_budget"
        radius, damping = float(initial_radius), 1e-6

        def failed(candidate, exc, phase):
            row["failed_trials"].append({"theta": dict(candidate), "phase": phase,
                                         "exception": type(exc).__name__, "error": str(exc)})
            emit("rejected_trial", row)

        def trial(unit):
            candidate = problem.theta_of(unit, fixed)
            try:
                return problem.forward(candidate)
            except failures as exc:
                failed(candidate, exc, "optimization")
                return None

        try:
            current = problem.forward(theta)
        except failures as exc:
            failed(theta, exc, "start")
            reason = "failed_initial_FE"
        if current is not None:
            for iteration in range(max_iterations):
                try:
                    problem.differentiate(current)
                except failures as exc:
                    failed(current.theta, exc, "derivative")
                    reason = "invalid_derivative"
                    break
                row["iterations"].append({"iteration": iteration,
                    "function_evaluations": problem.n_forward - nf0,
                    "theta": dict(current.theta), "sse": current.sse,
                    "projected_kkt_residual": current.kkt.projected_residual,
                    "trust_radius": radius, "damping": damping})
                emit("iteration", row)
                if current.kkt.projected_residual <= solve_tolerance:
                    reason = "projected_KKT"
                    break
                if problem.n_forward - nf0 >= max_nfev:
                    reason = "function_evaluation_budget"
                    break
                accepted = None
                for local_damping in (damping, max(damping, .01), max(damping, 1.)):
                    step = _bounded_step(current, radius, local_damping)
                    for half in range(12):
                        if problem.n_forward - nf0 >= max_nfev:
                            break
                        # Project the proposal only. Derivatives are evaluated
                        # later at the resulting physical coordinates.
                        proposed = np.clip(current.unit + step * (0.5 ** half), 0., 1.)
                        delta = proposed - current.unit
                        if np.max(np.abs(delta), initial=0.) < 2e-14:
                            break
                        candidate = trial(proposed)
                        if candidate is None or candidate.sse >= current.sse:
                            continue
                        predicted = -float(current.kkt.gradient @ delta) - .5 * float(
                            (current.jacobian @ delta) @ (current.jacobian @ delta))
                        actual = .5 * (current.sse - candidate.sse)
                        if predicted > 0 and actual < 1e-4 * predicted:
                            continue
                        ratio = actual / predicted if predicted > 0 else 0.
                        if ratio > .75 and half == 0:
                            radius, damping = min(.5, radius * 1.5), max(1e-12, local_damping * .2)
                        else:
                            damping = max(1e-12, local_damping)
                        accepted = candidate
                        break
                    if accepted is not None:
                        break
                if accepted is None:
                    # Changing adaptive FE histories can defeat a local linear
                    # model. A small true-objective poll may leave that path;
                    # it never changes the final exact derivative requirement.
                    for magnitude in (.001, .01):
                        for coordinate in np.argsort(-np.abs(current.kkt.projected_mapping)):
                            for sign in (-1., 1.):
                                if problem.n_forward - nf0 >= max_nfev:
                                    break
                                proposed = current.unit.copy()
                                proposed[coordinate] = np.clip(proposed[coordinate] + sign * magnitude, 0., 1.)
                                if np.array_equal(proposed, current.unit):
                                    continue
                                candidate = trial(proposed)
                                if candidate is not None and candidate.sse < current.sse:
                                    accepted = candidate
                                    break
                            if accepted is not None:
                                break
                        if accepted is not None:
                            break
                if accepted is None:
                    reason = ("function_evaluation_budget" if problem.n_forward - nf0 >= max_nfev
                              else "no_decreasing_exact_FE_trial")
                    break
                current = accepted

        if current is not None:
            row["candidate_theta"] = dict(current.theta)
            row["candidate_sse"] = current.sse
            try:
                row["terminal"] = problem.terminal(current.theta, kkt_tolerance=kkt_tolerance,
                                                    feasibility_tolerance=feasibility_tolerance)
                repeated_error = float(np.max(np.abs(
                    np.asarray(row["terminal"]["predictions_mm"]) - current.values), initial=0.))
                row["terminal"]["independent_forward_deviation_mm"] = repeated_error
                if repeated_error > replay_tolerance:
                    row["terminal"]["eligible"] = False
                    reason = "independent_FE_repeat_mismatch"
                row["status"] = "eligible_local" if row["terminal"]["eligible"] else "nonstationary_upper_bound"
            except failures as exc:
                failed(current.theta, exc, "terminal_verification")
                row["status"] = "failed_terminal_verification"
        else:
            row["status"] = "failed_initial_FE"
        row.update(stop_reason=reason, function_evaluations=problem.n_forward - nf0,
                   jacobian_evaluations=problem.n_jacobian - nj0,
                   wall_seconds=time.perf_counter() - began)
        emit("terminal", row)

    eligible = [r for r in report["candidates"] if r["status"] == "eligible_local"]
    report["best_eligible_index"] = None
    report["lower_unresolved_indices"] = []
    if eligible:
        best = min(eligible, key=lambda r: r["terminal"]["exact_objective"])
        report["best_eligible_index"] = best["start_index"]
        scale = max(abs(best["terminal"]["exact_objective"]), 1.)
        lower = [r["start_index"] for r in report["candidates"]
                 if r["status"] != "eligible_local" and "candidate_sse" in r
                 and r["candidate_sse"] < best["terminal"]["exact_objective"] - 1e-10 * scale]
        report["lower_unresolved_indices"] = lower
        if not lower:
            report["selected_index"] = best["start_index"]
    report["status"] = ("completed_local_pool" if report["selected_index"] is not None
                        else "unresolved_local_pool")
    report["function_evaluations"] = problem.n_forward
    report["jacobian_evaluations"] = problem.n_jacobian
    emit("complete")
    return report
