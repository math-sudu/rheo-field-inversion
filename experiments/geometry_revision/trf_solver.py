"""Independent bounded TRF solves with the common fresh-FE endpoint check.

An invalid FE trial aborts that TRF attempt. It never becomes a penalty
residual or a zero Jacobian. Both the returned point and any lower real
point visited by the optimizer remain in the reported candidate pool.
"""
from __future__ import annotations

import time

import numpy as np
from scipy.optimize import least_squares

from .solver import InvalidEvaluation, _Problem


def solve_trf(ev, y, box, free, fixed, start, *, failure_type=(),
              coordinate_map=None, max_nfev=15, solve_tolerance=1e-10,
              kkt_tolerance=1e-6, feasibility_tolerance=1e-10,
              replay_tolerance=1e-8):
    """Execute one independent start; all candidate endpoints are re-evaluated."""
    free, fixed = tuple(free), dict(fixed)
    if not free or set(free) & fixed.keys():
        raise ValueError("TRF requires disjoint nonempty free and fixed coordinates")
    if max_nfev < 1 or not 0 < solve_tolerance <= kkt_tolerance:
        raise ValueError("Invalid TRF budget or tolerance")
    failures = (failure_type,) if isinstance(failure_type, type) else tuple(failure_type)
    failures += (InvalidEvaluation,)
    problem = _Problem(ev, y, box, free, coordinate_map=coordinate_map,
                       replay_tolerance=replay_tolerance)
    initial = {**start, **fixed}
    unit = problem.unit_of(initial)  # An invalid proposal is never clipped.
    cache, best, refused = None, None, []

    def state_at(point):
        nonlocal cache, best
        if cache is not None and np.array_equal(cache.unit, point):
            return cache
        theta = problem.theta_of(point, fixed)
        try:
            state = problem.forward(theta)
        except failures as error:
            refused.append({"theta": theta, "phase": "forward", "error": repr(error)})
            cache = None
            raise
        cache = state
        if best is None or state.sse < best.sse:
            best = state
        return state

    def residual(point):
        return state_at(point).residual.copy()

    def jacobian(point):
        state = state_at(point)
        try:
            return problem.differentiate(state).jacobian.copy()
        except failures as error:
            refused.append({"theta": state.theta, "phase": "derivative", "error": repr(error)})
            raise

    started, result, aborted = time.perf_counter(), None, None
    try:
        result = least_squares(residual, unit, jac=jacobian, bounds=(0., 1.),
            method="trf", max_nfev=max_nfev, ftol=solve_tolerance,
            xtol=solve_tolerance, gtol=solve_tolerance)
    except failures as error:
        aborted = repr(error)
    proposed = []
    if result is not None:
        proposed.append(("returned", problem.theta_of(result.x, fixed)))
    if best is not None:
        if not proposed or proposed[0][1] != best.theta:
            proposed.append(("best_seen", dict(best.theta)))
    candidates = []
    for origin, theta in proposed:
        candidate = {"origin": origin, "candidate_theta": theta,
                     "candidate_sse": None, "terminal": None}
        if best is not None and theta == best.theta:
            candidate["candidate_sse"] = best.sse
        try:
            terminal = problem.terminal(theta, kkt_tolerance=kkt_tolerance,
                                         feasibility_tolerance=feasibility_tolerance)
            value = terminal["exact_objective"]
            previous = candidate["candidate_sse"]
            candidate.update(terminal=terminal,
                candidate_sse=value if previous is None else min(value, previous),
                status="eligible_local" if terminal["eligible"] else "nonstationary_upper_bound")
        except failures as error:
            candidate.update(status="failed_terminal_verification", error=repr(error))
        candidates.append(candidate)
    if not candidates:
        candidates.append({"origin": "initial", "candidate_theta": initial,
            "candidate_sse": None, "terminal": None, "status": "failed_initial_FE"})
    return {"solver": "scipy.optimize.least_squares/trf/unit-box/physical-replay",
        "start": initial, "max_nfev": max_nfev, "free": list(free), "fixed": fixed,
        "aborted": aborted, "optimizer": None if result is None else {
            "status": int(result.status), "message": str(result.message),
            "nfev": int(result.nfev), "njev": int(result.njev),
            "success": bool(result.success), "optimality": float(result.optimality)},
        "failed_trials": refused, "candidates": candidates,
        "actual_forward_calls": problem.n_forward, "actual_jacobian_calls": problem.n_jacobian,
        "wall_seconds": time.perf_counter()-started,
        "selection_rule": "The caller must assess every candidate and all rescue attempts; optimizer success is not an endpoint certificate."}
