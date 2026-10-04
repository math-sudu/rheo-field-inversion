
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch

from . import assembly, material, material_visc
from .meshing import DTYPE


# ------------------------------------------------------------------ BCs

@dataclass(frozen=True)
class DirichletOuter:
    """Nodal Cartesian perturbation displacements on the outer ring."""
    values: object  # (n_t, 2)

    @staticmethod
    def zeros(n_t):
        return DirichletOuter(np.zeros((n_t, 2)))


@dataclass(frozen=True)
class NeumannOuter:
    """Outer perturbation traction: nodal values (via B_out) or a
    precomputed consistent load vector (n_dof,)."""
    nodal: object = None    # (n_t, 2) Cartesian nodal traction values
    load: object = None     # (n_dof,) consistent load vector

    @staticmethod
    def zeros(n_t):
        return NeumannOuter(nodal=np.zeros((n_t, 2)))


@dataclass(frozen=True)
class RobinOuter:
    """(K + beta*B_out) u = f + B_out mu; mu nodal Cartesian values."""
    beta: float
    mu: object              # (n_t, 2)

    @staticmethod
    def zeros(n_t, beta):
        return RobinOuter(beta, np.zeros((n_t, 2)))


@dataclass(frozen=True)
class CondensedOuter:
    """Constant dense boundary stiffness block added on outer-ring dofs
    (interleaved [x0, y0, x1, y1, ...] ordering)."""
    K_bnd: object           # (2 n_t, 2 n_t)


def interp_bc(bc0, bc1, t):
    """Linear interpolation of boundary data within a knot interval."""
    if t >= 1.0:
        return bc1
    if type(bc0) is not type(bc1):
        raise TypeError("BC type must be constant within a knot interval")
    if isinstance(bc1, DirichletOuter):
        return DirichletOuter((1 - t) * np.asarray(bc0.values)
                              + t * np.asarray(bc1.values))
    if isinstance(bc1, NeumannOuter):
        if (bc0.nodal is None) != (bc1.nodal is None):
            raise ValueError("NeumannOuter form must be constant")
        if bc1.nodal is not None:
            return NeumannOuter(nodal=(1 - t) * np.asarray(bc0.nodal)
                                + t * np.asarray(bc1.nodal))
        return NeumannOuter(load=(1 - t) * np.asarray(bc0.load)
                            + t * np.asarray(bc1.load))
    if isinstance(bc1, RobinOuter):
        if bc0.beta != bc1.beta:
            raise ValueError("Robin beta must be constant within a knot")
        return RobinOuter(bc1.beta, (1 - t) * np.asarray(bc0.mu)
                          + t * np.asarray(bc1.mu))
    if isinstance(bc1, CondensedOuter):
        return bc1
    raise TypeError(type(bc1))


# ---------------------------------------------------------------- state

def _frozen(a):
    a = np.array(a, dtype=float, copy=True)
    a.setflags(write=False)
    return a


@dataclass(frozen=True)
class State:
    """Committed quadrature-point state (immutable numpy arrays)."""
    u: np.ndarray           # (n_dof,) converged perturbation displacement
    sig: np.ndarray         # (nq, 4) TOTAL stress (sxx, syy, szz, sxy)
    eps_p: np.ndarray       # (nq, 4) accumulated plastic strain
    kappa: np.ndarray       # (nq,) internal variable: accumulated dkappa
    #                       # (perfect kinds: plastic-multiplier measure,
    #                       # dkappa = dlam; cwfs: equivalent plastic
    #                       # shear strain, dkappa = deps_p_max -
    #                       # deps_p_min per increment)
    lam: float
    eps_K: np.ndarray = None    # (nq, 4) committed Kelvin strain, eng.
    #                           # shear, trace-free (viscous kinds only;
    #                           # REPLACED per commit); None otherwise
    t: float = None             # committed physical time (viscous kinds
    #                           # only: accumulated sum of committed dt
    #                           # from t = 0); None otherwise


@dataclass
class Trial:
    """Result of one trial solve (uncommitted)."""
    u: np.ndarray
    sig: np.ndarray
    eps_p: np.ndarray
    kappa: np.ndarray
    lam: float
    bc: object
    converged: bool
    n_iter: int
    res_hist: list
    region: np.ndarray      # (nq,) return-map region codes at convergence
    #                       # (viscous kinds: 2-code 0 elastic /
    #                       # 1 viscoplastic, from dlam > 0)
    eps_K: np.ndarray = None    # (nq, 4) new Kelvin strain (viscous only)
    t: float = None             # trial-end physical time (viscous only)
    lu: object = None       # SuperLU of the final consistent system
    mode: dict = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)


class IncrementFailure(RuntimeError):
    def __init__(self, msg, diagnostics=None):
        super().__init__(msg)
        self.diagnostics = diagnostics or {}


# --------------------------------------------------------------- system

class FESystem:
    """Mesh + base parameters + material binding (forward numpy path)."""

    def __init__(self, mesh, base, mat):
        self.mesh = mesh
        self.base = base
        self.mat = mat
        self.kind = mat.kind
        self.viscous = mat.kind in material_visc.VISCOUS_KINDS
        self.sigma0 = base.sigma0()                       # (4,)
        self.f1 = assembly.inner_unit_load(mesh, base.sigma_v, base.K0)
        self.m_out = assembly.ring_mass(mesh, "outer")
        self.B_out = assembly.b_out_global(mesh, self.m_out)
        self.rigid = assembly.rigid_vectors(mesh)
        self.K_elastic = assembly.assemble_elastic_K(mesh, base.E, base.nu)
        # torch constants of the forward path
        self._E_t = torch.tensor(base.E, dtype=DTYPE)
        self._nu_t = torch.tensor(base.nu, dtype=DTYPE)
        self._mat_t = ({} if mat.kind == "elastic" else
                       {k: torch.tensor(v, dtype=DTYPE)
                        for k, v in mat.theta().items()})
        if mat.kind == "cwfs":
            # Perzyna regularization setting (params.CWFSMohrCoulomb
            # .visc_H): threaded as a theta-independent CONSTANT --
            # deliberately not a theta() parameter (the 7-key
            # differentiability contract; diff.replay mirrors this).
            self._mat_t["visc_H"] = torch.tensor(mat.visc_H,
                                                 dtype=DTYPE)
        self._sig0_in = self.sigma0[[0, 1, 3]]

    # ------------------------------------------------------------ state
    def initial_state(self):
        nq = self.mesh.n_qp
        return State(
            u=_frozen(np.zeros(self.mesh.n_dof)),
            sig=_frozen(np.tile(self.sigma0, (nq, 1))),
            eps_p=_frozen(np.zeros((nq, 4))),
            kappa=_frozen(np.zeros(nq)),
            lam=0.0,
            eps_K=_frozen(np.zeros((nq, 4))) if self.viscous else None,
            t=0.0 if self.viscous else None,
        )

    # -------------------------------------------------- material bridge
    def integrate_np(self, sig_committed, kappa_committed, deps):
        """Numpy-facing full-increment integration from committed state."""
        with torch.no_grad():
            out = material.integrate_full(
                torch.tensor(np.asarray(deps), dtype=DTYPE),
                torch.tensor(np.asarray(sig_committed), dtype=DTYPE),
                torch.tensor(np.asarray(kappa_committed), dtype=DTYPE),
                self._E_t, self._nu_t, self._mat_t, self.kind)
        return tuple(o.numpy() for o in out)

    def tangent_np(self, sig_committed, kappa_committed, deps):
        D = material.consistent_tangent(
            torch.tensor(np.asarray(deps), dtype=DTYPE),
            torch.tensor(np.asarray(sig_committed), dtype=DTYPE),
            torch.tensor(np.asarray(kappa_committed), dtype=DTYPE),
            self._E_t, self._nu_t, self._mat_t, self.kind)
        return D.detach().numpy()

    def integrate_visc_np(self, sig_committed, eps_K_committed,
                          kappa_committed, deps, dt):
        with torch.no_grad():
            out = material_visc.integrate_visc(
                torch.tensor(np.asarray(deps), dtype=DTYPE),
                torch.tensor(np.asarray(sig_committed), dtype=DTYPE),
                torch.tensor(np.asarray(eps_K_committed), dtype=DTYPE),
                torch.tensor(np.asarray(kappa_committed), dtype=DTYPE),
                dt, self._E_t, self._nu_t, self._mat_t, self.kind)
        return tuple(o.numpy() for o in out)

    def tangent_visc_np(self, sig_committed, eps_K_committed,
                        kappa_committed, deps, dt):
        D = material_visc.consistent_tangent_visc(
            torch.tensor(np.asarray(deps), dtype=DTYPE),
            torch.tensor(np.asarray(sig_committed), dtype=DTYPE),
            torch.tensor(np.asarray(eps_K_committed), dtype=DTYPE),
            torch.tensor(np.asarray(kappa_committed), dtype=DTYPE),
            dt, self._E_t, self._nu_t, self._mat_t, self.kind)
        return D.detach().numpy()

    # ----------------------------------------------------- mode helpers
    def _mode_setup(self, bc):
        mesh = self.mesh
        if isinstance(bc, DirichletOuter):
            fixed = mesh.ring_dofs("outer")
            free = np.setdiff1d(np.arange(mesh.n_dof), fixed)
            g = np.asarray(bc.values, dtype=float).ravel()
            return {"kind": "dir", "fixed": fixed, "free": free, "g": g}
        if isinstance(bc, NeumannOuter):
            if bc.load is not None:
                f_out = np.asarray(bc.load, dtype=float)
            else:
                mu = np.zeros(mesh.n_dof)
                mu[mesh.ring_dofs("outer")] = np.asarray(
                    bc.nodal, dtype=float).ravel()
                f_out = self.B_out @ mu
            return {"kind": "neu", "f_out": f_out, "C": self.rigid}
        if isinstance(bc, RobinOuter):
            mu = np.zeros(mesh.n_dof)
            mu[mesh.ring_dofs("outer")] = np.asarray(
                bc.mu, dtype=float).ravel()
            return {"kind": "rob", "beta": float(bc.beta),
                    "f_mu": self.B_out @ mu}
        if isinstance(bc, CondensedOuter):
            K_add = assembly.scatter_ring_block(mesh, bc.K_bnd)
            return {"kind": "cond", "K_add": K_add,
                    "C": self.rigid[:, :2]}
        raise TypeError(type(bc))

    def mode_residual(self, md, R_bulk, u, omega):
        """Full residual of the mode's nonlinear system (numpy)."""
        k = md["kind"]
        if k == "dir":
            return R_bulk[md["free"]]
        if k == "neu":
            return np.concatenate([R_bulk + md["C"] @ omega
                                   - md["f_out"], md["C"].T @ u])
        if k == "rob":
            return R_bulk + md["beta"] * (self.B_out @ u) - md["f_mu"]
        if k == "cond":
            return np.concatenate([R_bulk + md["K_add"] @ u
                                   + md["C"] @ omega, md["C"].T @ u])
        raise ValueError(k)

    def mode_jacobian(self, md, K_T):
        k = md["kind"]
        if k == "dir":
            free = md["free"]
            return K_T[free][:, free].tocsc()
        if k == "neu":
            C = sp.csc_matrix(md["C"])
            return sp.bmat([[K_T, C], [C.T, None]], format="csc")
        if k == "rob":
            return (K_T + md["beta"] * self.B_out).tocsc()
        if k == "cond":
            C = sp.csc_matrix(md["C"])
            return sp.bmat([[K_T + md["K_add"], C], [C.T, None]],
                           format="csc")
        raise ValueError(k)

    def load_reference(self, md, lam):
        """Scale used for the relative Newton convergence test."""
        ref = np.linalg.norm(self.f1)
        k = md["kind"]
        if k == "neu":
            ref = max(ref, np.linalg.norm(md["f_out"]))
        if k == "rob":
            ref = max(ref, np.linalg.norm(md["f_mu"]))
        return max(ref, 1e-30)


    def outer_traction_from_reactions(self, state):
        """Nodal traction values (n_t, 2) on Gamma from consistent
        reactions (variationally consistent; prototype semantics).

        reactions = f_int(sigma - sigma0) - lambda*f1 restricted to the
        outer ring, converted through the boundary mass matrix.
        """
        sig_in = state.sig[:, [0, 1, 3]] - self._sig0_in
        r_full = assembly.f_int_qp(self.mesh, sig_in) - state.lam * self.f1
        r = r_full[self.mesh.ring_dofs("outer")].reshape(self.mesh.n_t, 2)
        return np.linalg.solve(self.m_out, r)


# --------------------------------------------------------------- newton

def trial_increment(system, committed, lam_end, bc, newton_tol=1e-11,
                    max_iter=40, dt=None):
    """Trial solve of one increment from the committed state.

    Never mutates ``committed``; deterministic function of its inputs.
    ``dt`` is the physical-time increment of the trial (strict
    contract, module docstring: REQUIRED > 0 for viscous systems,
    REJECTED for rate-independent systems).
    """
    if system.viscous:
        if dt is None:
            raise ValueError(f"viscous system (kind {system.kind!r}) "
                             "requires a physical-time increment dt")
        dt = float(dt)
        if not (np.isfinite(dt) and dt > 0.0):
            raise ValueError(f"dt must be positive finite, got {dt!r}")
        if committed.eps_K is None or committed.t is None:
            raise ValueError("viscous trial needs a viscous committed "
                             "state (eps_K, t) -- use "
                             "FESystem.initial_state()")
    elif dt is not None:
        raise ValueError("rate-independent system (kind "
                         f"{system.kind!r}) rejects dt (strict time-"
                         "axis contract)")
    mesh = system.mesh
    md = system._mode_setup(bc)
    u = committed.u.copy()
    if md["kind"] == "dir":
        u[md["fixed"]] = md["g"]
    n_border = (3 if md["kind"] == "neu"
                else 2 if md["kind"] == "cond" else 0)
    omega = np.zeros(n_border)
    ref = system.load_reference(md, lam_end)
    tol = newton_tol * ref
    res_hist = []
    converged = False
    fail_reason = None

    def evaluate(u_v, om_v):
        deps = mesh.qp_strains(u_v - committed.u)
        if system.viscous:
            sig, dep, dlam, eps_K_new, dkappa = system.integrate_visc_np(
                committed.sig, committed.eps_K, committed.kappa, deps,
                dt)
            region = (dlam > 0.0).astype(np.int64)   # 2-code (module doc)
        else:
            sig, dep, dlam, region, dkappa = system.integrate_np(
                committed.sig, committed.kappa, deps)
            eps_K_new = None
        R_bulk = (assembly.f_int_qp(mesh, sig[:, [0, 1, 3]]
                                    - system._sig0_in)
                  - lam_end * system.f1)
        R = system.mode_residual(md, R_bulk, u_v, om_v)
        return (deps, sig, dep, dkappa, region, eps_K_new, R,
                float(np.linalg.norm(R)))

    def tangent_at(deps):
        if system.viscous:
            return system.tangent_visc_np(committed.sig, committed.eps_K,
                                          committed.kappa, deps, dt)
        return system.tangent_np(committed.sig, committed.kappa, deps)

    deps, sig, dep, dkappa, region, eps_K_new, R, rn = evaluate(u, omega)
    res_hist.append(rn)
    for it in range(max_iter):
        if not np.isfinite(rn):
            fail_reason = "non-finite residual"
            break
        if rn <= tol:
            converged = True
            break
        D = tangent_at(deps)
        K_T = assembly.assemble_tangent_K(mesh, D)
        J = system.mode_jacobian(md, K_T)
        try:
            lu = spla.splu(J)
            d = lu.solve(R)
        except RuntimeError as err:      # singular tangent -> halving path
            fail_reason = f"factorization failed: {err}"
            break
        if not np.isfinite(d).all():
            fail_reason = "non-finite Newton step"
            break
        # deterministic backtracking line search on ||R||
        accepted = None
        step = 1.0
        for ls in range(5):
            u_try = u.copy()
            om_try = omega.copy()
            if md["kind"] == "dir":
                u_try[md["free"]] -= step * d
            else:
                u_try -= step * d[:mesh.n_dof]
                if n_border:
                    om_try -= step * d[mesh.n_dof:]
            cand = evaluate(u_try, om_try)
            if cand[7] < rn or ls == 4:
                accepted = (u_try, om_try, cand)
                break
            step *= 0.5
        u, omega, (deps, sig, dep, dkappa, region, eps_K_new, R,
                   rn) = accepted
        res_hist.append(rn)
    else:
        fail_reason = fail_reason or "max_iter reached"
    if not converged and rn <= tol and np.isfinite(rn):
        converged = True                 # converged on the last evaluation
    # final consistent factorization at the accepted iterate (replay)
    lu_final = None
    if converged:
        D = tangent_at(deps)
        K_T = assembly.assemble_tangent_K(mesh, D)
        try:
            lu_final = spla.splu(system.mode_jacobian(md, K_T))
        except RuntimeError as err:
            converged = False
            fail_reason = f"final factorization failed: {err}"
    diagnostics = {
        "res_hist": res_hist,
        "fail_reason": None if converged else fail_reason,
        "n_plastic": int((region > 0).sum()) if region is not None else 0,
        "region_counts": (np.bincount(region, minlength=5).tolist()
                          if region is not None else None),
        "dt": dt,
    }
    return Trial(
        u=_frozen(u), sig=_frozen(sig), eps_p=_frozen(committed.eps_p + dep),
        kappa=_frozen(committed.kappa + dkappa), lam=float(lam_end), bc=bc,
        converged=converged, n_iter=len(res_hist) - 1, res_hist=res_hist,
        region=region, lu=lu_final,
        eps_K=(_frozen(eps_K_new) if system.viscous else None),
        t=(committed.t + dt if system.viscous else None),
        mode={"kind": md["kind"], "omega": omega.copy(),
              **({"free": md["free"], "fixed": md["fixed"], "g": md["g"]}
                 if md["kind"] == "dir" else {}),
              **({"f_out": md["f_out"]} if md["kind"] == "neu" else {}),
              **({"beta": md["beta"], "f_mu": md["f_mu"],
                  "mu": np.asarray(bc.mu, dtype=float)}
                 if md["kind"] == "rob" else {}),
              **({"K_bnd": np.asarray(bc.K_bnd, dtype=float)}
                 if md["kind"] == "cond" else {})},
        diagnostics=diagnostics)


def commit(trial):
    if not trial.converged:
        raise IncrementFailure("cannot commit a non-converged trial",
                               trial.diagnostics)
    return State(u=trial.u, sig=trial.sig, eps_p=trial.eps_p,
                 kappa=trial.kappa, lam=trial.lam,
                 eps_K=trial.eps_K, t=trial.t)


# --------------------------------------------------------------- driver

@dataclass
class ExecutedIncrement:
    knot: int               # interval index k: lambda in (lam_k, lam_{k+1}]
    t0: float               # start fraction within the interval
    t1: float               # end fraction within the interval
    committed_before: State
    trial: Trial


@dataclass
class RunResult:
    system: object
    lambdas: np.ndarray
    bcs: list
    executed: list
    final_state: State
    times: np.ndarray = None        # physical-time knots (viscous runs)
    failed_attempts: list = field(default_factory=list)
    #                               # non-converged attempt records
    #                               # (module docstring)

    @property
    def n_halvings(self):
        return len(self.executed) - (len(self.lambdas) - 1)


def validate_times(system, lambdas, times, t_start=0.0):
    """Strict dual-axis contract (module docstring).

    Viscous systems REQUIRE a physical-time knot array ``times``
    parallel to ``lambdas`` (times[0] = the committed start time,
    strictly increasing so every substep gets dt > 0); rate-independent
    systems REJECT one.  Returns the validated float array (or None).
    """
    if not system.viscous:
        if times is not None:
            raise ValueError("rate-independent system (kind "
                             f"{system.kind!r}) rejects a times axis "
                             "(strict time-axis contract)")
        return None
    if times is None:
        raise ValueError(f"viscous system (kind {system.kind!r}) "
                         "requires a times knot array parallel to "
                         "lambdas")
    times = np.asarray(times, dtype=float)
    if times.ndim != 1 or len(times) != len(lambdas):
        raise ValueError("times must be a 1-D knot array parallel to "
                         f"lambdas (got {len(times)} knots for "
                         f"{len(lambdas)} lambdas)")
    if abs(times[0] - t_start) > 1e-12 * max(1.0, abs(t_start)):
        raise ValueError(f"times[0] = {times[0]!r} must equal the "
                         f"committed start time {t_start!r}")
    if not np.all(np.diff(times) > 0.0):
        raise ValueError("times must be strictly increasing "
                         "(every increment needs dt > 0)")
    return times


def run_schedule(system, lambdas, bcs, newton_tol=1e-11, max_iter=40,
                 min_substep=1.0 / 64.0, times=None):
    """Drive the lambda schedule; halve increments on Newton failure.

    ``times``: physical-time knots parallel to ``lambdas`` (viscous
    systems only -- ``validate_times``).  Within a knot interval the
    (lambda, t) loading path is the straight segment between knots, so
    substep halving halves the release fraction AND dt JOINTLY; equal
    lambda knots with advancing time are a pure creep/relaxation hold.
    Non-converged attempts are recorded on ``RunResult.failed_attempts``
    (and in the ``IncrementFailure`` diagnostics as
    ``"failed_attempts"``): dicts with knot, t0/t1 substep fractions,
    lam_target, dt, the Newton ``res_hist`` and the fail reason.
    """
    lambdas = np.asarray(lambdas, dtype=float)
    if len(bcs) != len(lambdas):
        raise ValueError("need one BC per schedule knot (incl. lambda_0)")
    if abs(lambdas[0]) > 1e-15:
        raise ValueError("schedule must start at lambda = 0")
    times = validate_times(system, lambdas, times, t_start=0.0)
    committed = system.initial_state()
    executed = []
    failed = []
    for k in range(len(lambdas) - 1):
        t, step = 0.0, 1.0
        while t < 1.0 - 1e-12:
            t_next = min(t + step, 1.0)
            lam_end = (lambdas[k]
                       + t_next * (lambdas[k + 1] - lambdas[k]))
            dt = (None if times is None
                  else (t_next - t) * (times[k + 1] - times[k]))
            bc_end = interp_bc(bcs[k], bcs[k + 1], t_next)
            trial = trial_increment(system, committed, lam_end, bc_end,
                                    newton_tol=newton_tol,
                                    max_iter=max_iter, dt=dt)
            if trial.converged:
                executed.append(ExecutedIncrement(
                    k, t, t_next, committed, trial))
                committed = commit(trial)
                t = t_next
            else:
                failed.append({
                    "knot": k, "t0": t, "t1": t_next,
                    "lam_target": float(lam_end), "dt": dt,
                    "res_hist": list(trial.res_hist),
                    "fail_reason": trial.diagnostics["fail_reason"],
                })
                step *= 0.5
                if step < min_substep:
                    raise IncrementFailure(
                        f"increment (knot {k}, t = {t:.4f} -> "
                        f"{t_next:.4f}) failed below the substep floor "
                        f"{min_substep}; last residuals "
                        f"{trial.res_hist[-3:]}",
                        dict(trial.diagnostics,
                             failed_attempts=failed))
    return RunResult(system=system, lambdas=lambdas, bcs=bcs,
                     executed=executed, final_state=committed,
                     times=times, failed_attempts=failed)


# ------------------------------------------------------------------ QoI

def mean_inner_radial_disp(u, mesh):
    """Mean radial displacement over inner-ring nodes (numpy or torch)."""
    th = mesh.theta_nodes
    idx = mesh.ring_dofs("inner")
    if isinstance(u, torch.Tensor):
        ur = (u[torch.as_tensor(idx[0::2])] * torch.as_tensor(np.cos(th))
              + u[torch.as_tensor(idx[1::2])] * torch.as_tensor(np.sin(th)))
        return ur.mean()
    ux, uy = u[idx[0::2]], u[idx[1::2]]
    return float(np.mean(ux * np.cos(th) + uy * np.sin(th)))
