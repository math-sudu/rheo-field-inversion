"""Residual-decreasing Newton trials for the verified local FE runtime.

Only the nonlinear equation solver changes. Constitutive integration, loads,
boundary equations, convergence tolerance, committed state, and the final
consistent factorization retain the frozen FE contract. Install explicitly
after activating the local runtime, and include the returned policy in the
calculation identity. Callers control the Newton iteration budget as before.
"""
from __future__ import annotations

from functools import partial
import hashlib
import inspect
from pathlib import Path

import numpy as np
import scipy.sparse.linalg as spla


def trial_increment(system, committed, lam_end, bc, newton_tol=1e-11,
                    max_iter=40, dt=None, *, max_backtracks=24, armijo=1e-4,
                    residual_gradient=True):
    """Solve from one immutable commit; never accept an increasing residual."""
    from fedev_src import assembly, solver as fe

    if max_iter < 0 or max_backtracks < 1 or not 0 < armijo < 1:
        raise ValueError("Invalid Newton or line-search controls")
    if not np.isfinite(newton_tol) or newton_tol <= 0:
        raise ValueError("Newton tolerance must be finite and positive")
    if system.viscous:
        if dt is None:
            raise ValueError(f"viscous system (kind {system.kind!r}) requires a physical-time increment dt")
        dt = float(dt)
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError(f"dt must be positive finite, got {dt!r}")
        if committed.eps_K is None or committed.t is None:
            raise ValueError("viscous trial needs committed Kelvin strain and physical time")
    elif dt is not None:
        raise ValueError(f"rate-independent system (kind {system.kind!r}) rejects dt")

    mesh = system.mesh
    mode = system._mode_setup(bc)
    u = committed.u.copy()
    if mode["kind"] == "dir":
        u[mode["fixed"]] = mode["g"]
    n_border = 3 if mode["kind"] == "neu" else 2 if mode["kind"] == "cond" else 0
    omega = np.zeros(n_border)
    reference = system.load_reference(mode, lam_end)
    tolerance = newton_tol * reference
    res_hist, searches = [], []
    converged, fail_reason = False, None

    def evaluate(displacement, multiplier):
        deps = mesh.qp_strains(displacement - committed.u)
        if system.viscous:
            sig, dep, dlam, eps_kelvin, dkappa = system.integrate_visc_np(
                committed.sig, committed.eps_K, committed.kappa, deps, dt)
            region = (dlam > 0).astype(np.int64)
        else:
            sig, dep, dlam, region, dkappa = system.integrate_np(
                committed.sig, committed.kappa, deps)
            eps_kelvin = None
        bulk = (assembly.f_int_qp(mesh, sig[:, [0, 1, 3]] - system._sig0_in)
                - lam_end * system.f1)
        residual = system.mode_residual(mode, bulk, displacement, multiplier)
        return (deps, sig, dep, dkappa, region, eps_kelvin, residual,
                float(np.linalg.norm(residual)))

    def tangent(deps):
        if system.viscous:
            return system.tangent_visc_np(committed.sig, committed.eps_K,
                                          committed.kappa, deps, dt)
        return system.tangent_np(committed.sig, committed.kappa, deps)

    state = evaluate(u, omega)
    res_hist.append(state[-1])
    for iteration in range(max_iter):
        deps, _, _, _, _, _, residual, norm = state
        if not np.isfinite(norm):
            fail_reason = "non-finite residual"
            break
        if norm <= tolerance:
            converged = True
            break
        jacobian = system.mode_jacobian(mode, assembly.assemble_tangent_K(mesh, tangent(deps)))
        try:
            lu = spla.splu(jacobian)
            newton_direction = lu.solve(residual)
        except RuntimeError as exc:
            fail_reason = f"factorization failed: {exc}"
            break
        if not np.isfinite(newton_direction).all():
            fail_reason = "non-finite Newton step"
            break
        directions = [("newton", newton_direction)]
        if residual_gradient:
            gradient = np.asarray(jacobian.T @ residual).ravel()
            diagonal = np.asarray(jacobian.power(2).sum(axis=0)).ravel()
            floor = max(float(np.max(diagonal, initial=0.)), 1.) * 1e-14
            direction = gradient / np.maximum(diagonal, floor)
            directions.append(("scaled_residual_gradient", direction))

        accepted = None
        for label, direction in directions:
            directional_decrease = float(residual @ (jacobian @ direction))
            if not np.isfinite(directional_decrease) or directional_decrease <= 0:
                continue
            step = 1.
            rejected = []
            for _ in range(max_backtracks):
                u_trial, omega_trial = u.copy(), omega.copy()
                if mode["kind"] == "dir":
                    u_trial[mode["free"]] -= step * direction
                else:
                    u_trial -= step * direction[:mesh.n_dof]
                    if n_border:
                        omega_trial -= step * direction[mesh.n_dof:]
                candidate = evaluate(u_trial, omega_trial)
                candidate_norm = candidate[-1]
                merit_bound = .5 * norm ** 2 - armijo * step * directional_decrease
                if (np.isfinite(candidate_norm) and candidate_norm < norm
                        and (.5 * candidate_norm ** 2 <= merit_bound or candidate_norm <= tolerance)):
                    accepted = (u_trial, omega_trial, candidate)
                    searches.append({"iteration": iteration, "direction": label,
                        "accepted_step": step, "residual_before": norm,
                        "residual_after": candidate_norm, "rejected": rejected})
                    break
                rejected.append({"step": step,
                                 "residual": candidate_norm if np.isfinite(candidate_norm) else None})
                step *= .5
            if accepted is not None:
                break
            searches.append({"iteration": iteration, "direction": label,
                "accepted_step": None, "residual_before": norm, "rejected": rejected})
        if accepted is None:
            fail_reason = "line search found no residual-decreasing step"
            break
        u, omega, state = accepted
        res_hist.append(state[-1])
    else:
        fail_reason = "max_iter reached"

    deps, sig, dep, dkappa, region, eps_kelvin, _, norm = state
    if np.isfinite(norm) and norm <= tolerance:
        converged = True
    final_lu = None
    if converged:
        try:
            final_lu = spla.splu(system.mode_jacobian(
                mode, assembly.assemble_tangent_K(mesh, tangent(deps))))
        except RuntimeError as exc:
            converged = False
            fail_reason = f"final factorization failed: {exc}"
    diagnostics = {"res_hist": res_hist, "fail_reason": None if converged else fail_reason,
        "n_plastic": int((region > 0).sum()) if region is not None else 0,
        "region_counts": np.bincount(region, minlength=5).tolist() if region is not None else None,
        "dt": dt, "line_search": searches, "newton_tolerance": float(newton_tol),
        "load_reference": float(reference), "absolute_residual_tolerance": float(tolerance),
        "newton_iteration_budget": int(max_iter)}
    return fe.Trial(
        u=fe._frozen(u), sig=fe._frozen(sig), eps_p=fe._frozen(committed.eps_p + dep),
        kappa=fe._frozen(committed.kappa + dkappa), lam=float(lam_end), bc=bc,
        converged=converged, n_iter=len(res_hist) - 1, res_hist=res_hist,
        region=region, lu=final_lu,
        eps_K=fe._frozen(eps_kelvin) if system.viscous else None,
        t=committed.t + dt if system.viscous else None,
        mode={"kind": mode["kind"], "omega": omega.copy(),
              **({"free": mode["free"], "fixed": mode["fixed"], "g": mode["g"]}
                 if mode["kind"] == "dir" else {}),
              **({"f_out": mode["f_out"]} if mode["kind"] == "neu" else {}),
              **({"beta": mode["beta"], "f_mu": mode["f_mu"], "mu": np.asarray(bc.mu, float)}
                 if mode["kind"] == "rob" else {}),
              **({"K_bnd": np.asarray(bc.K_bnd, float)} if mode["kind"] == "cond" else {})},
        diagnostics=diagnostics)


def install_newton(*, max_backtracks=24, armijo=1e-4, residual_gradient=True):
    """Install once on the activated FE module and return its identity fields."""
    from fedev_src import solver as fe

    policy = {"version": "residual-decreasing-newton-v1", "max_backtracks": int(max_backtracks),
              "armijo": float(armijo), "residual_gradient": bool(residual_gradient),
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "frozen_solver_sha256": hashlib.sha256(Path(fe.__file__).read_bytes()).hexdigest()}
    current = getattr(fe.trial_increment, "geometry_revision_policy", None)
    if current is not None:
        if current != policy:
            raise RuntimeError("A different Newton policy is already installed; start a fresh process")
        return dict(policy)
    if Path(inspect.getfile(fe.trial_increment)).resolve() != Path(fe.__file__).resolve():
        raise RuntimeError("Refusing to replace an unrecognized FE trial solver")
    installed = partial(trial_increment, max_backtracks=max_backtracks, armijo=armijo,
                        residual_gradient=residual_gradient)
    installed.geometry_revision_policy = dict(policy)
    fe.trial_increment = installed
    return dict(policy)
