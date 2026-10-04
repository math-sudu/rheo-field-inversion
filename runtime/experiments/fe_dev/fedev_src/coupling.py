
import time
from dataclasses import dataclass, field

import numpy as np

from . import assembly, solver
from . import proto_bridge as pb


# ------------------------------------------------------- polar helpers

def polar_stacked_to_cart_nodal(x, theta):
    """Stacked polar [v_r; v_t] (2 n_t,) -> Cartesian nodal (n_t, 2)."""
    n_t = len(theta)
    c, s = np.cos(theta), np.sin(theta)
    vx = x[:n_t] * c - x[n_t:] * s
    vy = x[:n_t] * s + x[n_t:] * c
    return np.column_stack([vx, vy])


def cart_nodal_to_polar_stacked(v, theta):
    """Cartesian nodal (n_t, 2) -> stacked polar [v_r; v_t] (2 n_t,)."""
    c, s = np.cos(theta), np.sin(theta)
    vr = v[:, 0] * c + v[:, 1] * s
    vt = -v[:, 0] * s + v[:, 1] * c
    return np.concatenate([vr, vt])


# ----------------------------------------------------------- far zone

def make_far(system, carrier="laurent_ls", n_coeff=8, **kw):
    mesh = system.mesh
    if carrier == "laurent_ls":
        return pb.farfield.FarLaurentLS(mesh.R, mesh.theta_nodes,
                                        system.base, n_coeff=n_coeff)
    if carrier == "coeff_net":
        return pb.coeff_net.FarCoeffNet(mesh.R, mesh.theta_nodes,
                                        system.base, n_coeff=n_coeff, **kw)
    raise ValueError(f"unknown carrier {carrier!r}")


def traction_mismatch(system, state_or_trial, far, floor=1e-30):
    """RULE-2 criterion (ii): relative two-side traction mismatch.

    t_near: reaction-recovered nodal traction of the near solve;
    t_far: the carrier's traction trace at the interface nodes; both as
    stacked polar vectors.  Returns
    ``||t_near - t_far||_2 / max(||t_near||_2, floor)``.
    """
    th = system.mesh.theta_nodes
    t_near = cart_nodal_to_polar_stacked(
        system.outer_traction_from_reactions(state_or_trial), th)
    t_far = far.eval_kind("traction")
    return float(np.linalg.norm(t_near - t_far)
                 / max(np.linalg.norm(t_near), floor))


# ---------------------------------------------------------- strategies

@dataclass(frozen=True)
class Strategy:
    """Transmission x relaxation combination.

    transmission: "RR" (Robin-Robin, beta = beta_factor*G/R_Gamma) or
    "DN-FEdir" (FE side receives Dirichlet).  relax: float rho or the
    string "aitken".
    """
    transmission: str
    relax: object
    beta_factor: float = 2.0

    def __post_init__(self):
        if self.transmission not in ("RR", "DN-FEdir"):
            raise ValueError(self.transmission)

    @property
    def name(self):
        t = (f"RRb{self.beta_factor:g}" if self.transmission == "RR"
             else "DNdir")
        r = ("aitken" if self.relax == "aitken"
             else f"rho{float(self.relax):g}")
        return f"{t}_{r}"

    def beta(self, system):
        if self.transmission != "RR":
            return None
        return self.beta_factor * system.base.G / system.mesh.R


RR_B2_RHO1 = Strategy("RR", 1.0, 2.0)          # primary default
RR_B2_AITKEN = Strategy("RR", "aitken", 2.0)
DN_FEDIR_AITKEN = Strategy("DN-FEdir", "aitken")   # fallback default
DN_FEDIR_RHO1 = Strategy("DN-FEdir", 1.0)      # measured proto stall


# ----------------------------------------------------------- failures

class CouplingFailure(RuntimeError):
    """Raised when an increment cannot be advanced (RULE 3 exhausted) or
    the far-zone elasticity assumption is violated.  ``diagnostics``
    carries the schedule position, sweep/residual histories, rollback
    trail, strategy switches, plastic-QP counts and the last committed
    state."""

    def __init__(self, msg, diagnostics=None):
        super().__init__(msg)
        self.diagnostics = diagnostics or {}


# -------------------------------------------------------- plastic front

def plastic_front(mesh, kappa, tol=1e-12):
    """(n_plastic, r_front, margin, outermost_ring_hit) from QP kappa.

    margin = R_Gamma - r_front (np.inf when no QP is plastic);
    outermost_ring_hit flags plastic QPs in the outermost element ring
    (radial element index n_r - 1) -- the far-zone elasticity guard.
    """
    plastic = np.asarray(kappa) > tol
    n_plastic = int(plastic.sum())
    if n_plastic == 0:
        return 0, np.nan, np.inf, False
    r_front = float(mesh.r_qp[plastic].max())
    outer_elems = mesh.qp_elem >= (mesh.n_r - 1) * mesh.n_t
    hit = bool((plastic & outer_elems).any())
    return n_plastic, r_front, float(mesh.R - r_front), hit


# ------------------------------------------------------ sweep outcome

@dataclass
class SweepOutcome:
    """Result of one increment ATTEMPT (a Schwarz sweep loop)."""
    status: str             # converged | max_sweeps | rising | diverged
    #                       # | newton_failure
    sweeps: int
    trial: object           # last near trial (converged iff status ok)
    x: np.ndarray           # interface state entering the last sweep
    hist_u: list            # res_u per sweep (from sweep 2)
    hist_t: list            # res_t per sweep (aligned with hist_u)
    newton_iters: int       # near Newton iterations spent in the attempt


def schwarz_increment(system, committed, lam_end, strategy, far, x0, *,
                      tol_u=1e-6, tol_t=1e-4, t_floor=1e-30, k_max=20,
                      m_rise=3, div_guard=1e6, newton_tol=1e-11,
                      max_iter=40, dt=None):
    """One coupled increment attempt from ``committed`` to ``lam_end``.

    RULE 1: every near solve is a pure trial from ``committed``.
    Iteration/relaxation logic is the prototype's verbatim; the dual
    RULE-2 criteria and the RULE-3 divergence triggers (b) are checked
    per sweep.  Never commits; the caller owns commit/rollback.
    ``dt``: physical-time increment of the attempt (viscous systems
    only; forwarded to every near trial -- solver strict contract).
    """
    mesh = system.mesh
    th = mesh.theta_nodes
    outer = mesh.ring_dofs("outer")
    beta = strategy.beta(system)
    rr = strategy.transmission == "RR"

    x = np.array(x0, dtype=float, copy=True)
    aitken = strategy.relax == "aitken"
    rho_prev, r_prev, y_prev = 0.5, None, None
    hist_u, hist_t = [], []
    newton_total = 0
    trial = None

    for k in range(1, k_max + 1):
        # ------------------------- one sweep: near trial + far fit
        data_cart = polar_stacked_to_cart_nodal(x, th)
        bc = (solver.RobinOuter(beta, data_cart) if rr
              else solver.DirichletOuter(data_cart))
        trial = solver.trial_increment(system, committed, lam_end, bc,
                                       newton_tol=newton_tol,
                                       max_iter=max_iter, dt=dt)
        newton_total += trial.n_iter
        if not trial.converged:                      # RULE-3 trigger (c)
            return SweepOutcome("newton_failure", k, trial, x,
                                hist_u, hist_t, newton_total)
        if rr:
            y = cart_nodal_to_polar_stacked(
                np.asarray(trial.u)[outer].reshape(mesh.n_t, 2), th)
            far.fit("robin_minus", x - 2.0 * beta * y, beta=beta)
            x_hat = far.eval_kind("robin_plus", beta=beta)
        else:
            t_pol = cart_nodal_to_polar_stacked(
                system.outer_traction_from_reactions(trial), th)
            far.fit("traction", t_pol)
            x_hat = far.eval_kind("displacement")
            y = x.copy()             # near trace = the imposed state

        # ------------------------------- dual criteria (sweep >= 2)
        if k >= 2:
            res_u = (np.linalg.norm(y - y_prev)
                     / max(np.linalg.norm(y), 1e-300))
            res_t = traction_mismatch(system, trial, far, t_floor)
            hist_u.append(res_u)
            hist_t.append(res_t)
            if not np.isfinite(res_u) or res_u > div_guard:
                return SweepOutcome("diverged", k, trial, x,
                                    hist_u, hist_t, newton_total)
            if res_u <= tol_u and res_t <= tol_t:    # RULE 2 satisfied
                return SweepOutcome("converged", k, trial, x,
                                    hist_u, hist_t, newton_total)
            if m_rise and len(hist_u) >= m_rise + 1 and all(
                    hist_u[-i] > hist_u[-i - 1]
                    for i in range(1, m_rise + 1)):  # RULE-3 trigger (b)
                return SweepOutcome("rising", k, trial, x,
                                    hist_u, hist_t, newton_total)
        y_prev = y

        # --------------------------- relaxation (prototype verbatim)
        r = x_hat - x
        if aitken:
            if r_prev is None:
                rho = 0.5
            else:
                dr = r - r_prev
                den = float(dr @ dr)
                rho = (-rho_prev * float(r_prev @ dr) / den
                       if den > 0.0 else rho_prev)
                rho = float(np.clip(rho, 0.05, 2.0))
            rho_prev = rho
            r_prev = r
        else:
            rho = float(strategy.relax)
        x = x + rho * r
        if not np.all(np.isfinite(x)):
            return SweepOutcome("diverged", k, trial, x,
                                hist_u, hist_t, newton_total)
    return SweepOutcome("max_sweeps", k_max, trial, x,   # RULE-3 (a)
                        hist_u, hist_t, newton_total)


# ------------------------------------------------------------- records

@dataclass
class IncrementRecord:
    """Diagnostics of one COMMITTED increment (CSV-friendly)."""
    knot: int
    t0: float
    t1: float
    lam_end: float
    strategy: str
    sweeps: int
    res_u: float            # criterion (i) at convergence
    res_t: float            # criterion (ii) at convergence
    hist_u: list
    hist_t: list
    n_plastic: int
    r_front: float
    margin: float           # R_Gamma - r_front (inf if elastic)
    rollbacks_before: int   # consecutive rollbacks preceding this commit
    newton_iters: int
    wall_s: float
    far_params: np.ndarray  # carrier parameter vector at convergence
    state: object           # committed State AFTER this increment
    dt: float = None        # physical-time increment (viscous runs)

    def as_row(self):
        return {
            "knot": self.knot, "t0": self.t0, "t1": self.t1,
            "lam_end": self.lam_end, "strategy": self.strategy,
            "sweeps": self.sweeps, "res_u": self.res_u,
            "res_t": self.res_t, "n_plastic": self.n_plastic,
            "r_front": self.r_front, "margin": self.margin,
            "rollbacks_before": self.rollbacks_before,
            "newton_iters": self.newton_iters, "wall_s": self.wall_s,
            "dt": self.dt,
        }


@dataclass
class CouplingResult:
    system: object
    lambdas: np.ndarray
    strategy: object            # primary strategy
    records: list               # IncrementRecord per committed increment
    rollbacks: list             # rollback trail (dicts)
    strategy_switches: list     # switch events (dicts)
    final_state: object
    n_commits: int
    far: object
    warm_start: bool = True
    times: np.ndarray = None    # physical-time knots (viscous runs)

    @property
    def sweeps_per_increment(self):
        return [r.sweeps for r in self.records]


# -------------------------------------------------------------- driver

def run_coupled_schwarz(system, lambdas, strategy=RR_B2_RHO1, *,
                        fallback="auto", far=None, carrier="laurent_ls",
                        n_coeff=8, tol_u=1e-6, tol_t=1e-4,
                        t_floor=1e-30, k_max=20, m_rise=3,
                        div_guard=1e6, min_substep_frac=1.0 / 16.0,
                        warm_start=True, newton_tol=1e-11, max_iter=40,
                        initial_state=None, times=None):
    """Drive the lambda schedule through the three-rule Schwarz format.

    Outer loop over the schedule knots; within [lam_k, lam_{k+1}]
    Schwarz sweeps (``schwarz_increment``); commit only on dual-criteria
    convergence (RULE 2); rollback with substep halving, one-time
    fallback switch after 2 consecutive rollbacks, and explicit
    ``CouplingFailure`` at the substep floor (RULE 3).

    ``fallback``: a Strategy, None (no fallback -- the floor raises),
    or "auto" (default): DN-FEdir + Aitken for RR primaries -- the
    spec's default pairing -- and RRb2 + rho1 for DN-FEdir primaries.

    ``times``: physical-time knots parallel to ``lambdas`` (viscous
    systems only; ``solver.validate_times`` with the committed start
    time).  Rollback halves the release fraction AND dt jointly (the
    retried attempt covers the same straight (lambda, t) segment at
    half length); every rollback record carries the FULL interface
    residual histories (``hist_u``/``hist_t``) plus the near-Newton
    ``res_hist`` of its last trial, feeding the failure-row residual
    dump of the matrix runner.
    """
    lambdas = np.asarray(lambdas, dtype=float)
    if abs(lambdas[0]) > 1e-15:
        raise ValueError("schedule must start at lambda = 0")
    if isinstance(fallback, str):
        if fallback != "auto":
            raise ValueError(fallback)
        fallback = (DN_FEDIR_AITKEN if strategy.transmission == "RR"
                    else RR_B2_RHO1)
    if far is None:
        far = make_far(system, carrier=carrier, n_coeff=n_coeff)
    committed = (system.initial_state() if initial_state is None
                 else initial_state)
    times = solver.validate_times(
        system, lambdas, times,
        t_start=(committed.t if committed.t is not None else 0.0))
    n_t = system.mesh.n_t
    x_warm = np.zeros(2 * n_t)
    active = strategy
    switched = False
    records, rollbacks, switches = [], [], []
    n_commits = 0
    consec_rollbacks = 0
    sweep_kw = dict(tol_u=tol_u, tol_t=tol_t, t_floor=t_floor,
                    k_max=k_max, m_rise=m_rise, div_guard=div_guard,
                    newton_tol=newton_tol, max_iter=max_iter)

    def fail(msg, out, knot, t, t_next, step, dt=None):
        raise CouplingFailure(msg, diagnostics={
            "knot": knot, "t": t, "t_next": t_next, "step": step,
            "lam_target": (lambdas[knot]
                           + t_next * (lambdas[knot + 1] - lambdas[knot])),
            "dt": dt,
            "strategy": active.name,
            "last_status": out.status if out else None,
            "last_sweeps": out.sweeps if out else 0,
            "last_hist_u": list(out.hist_u) if out else [],
            "last_hist_t": list(out.hist_t) if out else [],
            "last_newton_res_hist": (list(out.trial.res_hist)
                                     if out and out.trial else []),
            "last_n_plastic": (out.trial.diagnostics.get("n_plastic")
                               if out and out.trial else None),
            "rollbacks": list(rollbacks),
            "strategy_switches": list(switches),
            "records": [r.as_row() for r in records],
            "committed_state": committed,
            "committed_lam": committed.lam,
        })

    for k in range(len(lambdas) - 1):
        t, step = 0.0, 1.0
        while t < 1.0 - 1e-12:
            t_next = min(t + step, 1.0)
            lam_end = (lambdas[k]
                       + t_next * (lambdas[k + 1] - lambdas[k]))
            dt = (None if times is None
                  else (t_next - t) * (times[k + 1] - times[k]))
            x0 = x_warm if warm_start else np.zeros(2 * n_t)
            t_wall = time.perf_counter()
            out = schwarz_increment(system, committed, lam_end, active,
                                    far, x0, dt=dt, **sweep_kw)
            wall = time.perf_counter() - t_wall
            if out.status == "converged":
                n_pl, r_front, margin, hit = plastic_front(
                    system.mesh, out.trial.kappa)
                if hit:                    # far-zone elasticity guard
                    fail("plastic front reached the outermost element "
                         f"ring (r_front = {r_front:.6g}, margin = "
                         f"{margin:.3g})", out, k, t, t_next, step, dt)
                committed = solver.commit(out.trial)
                n_commits += 1
                records.append(IncrementRecord(
                    knot=k, t0=t, t1=t_next, lam_end=float(lam_end),
                    strategy=active.name, sweeps=out.sweeps,
                    res_u=out.hist_u[-1], res_t=out.hist_t[-1],
                    hist_u=list(out.hist_u), hist_t=list(out.hist_t),
                    n_plastic=n_pl, r_front=r_front, margin=margin,
                    rollbacks_before=consec_rollbacks,
                    newton_iters=out.newton_iters, wall_s=wall,
                    far_params=np.array(far.p, copy=True),
                    state=committed, dt=dt))
                x_warm = np.array(out.x, copy=True)
                t = t_next
                consec_rollbacks = 0
            else:
                # ------------------------------- RULE 3: rollback
                consec_rollbacks += 1
                rollbacks.append({
                    "knot": k, "t": t, "t_next": t_next, "step": step,
                    "lam_target": float(lam_end), "dt": dt,
                    "strategy": active.name, "status": out.status,
                    "sweeps": out.sweeps,
                    "res_u_tail": list(out.hist_u[-3:]),
                    "res_t_tail": list(out.hist_t[-3:]),
                    "hist_u": list(out.hist_u),
                    "hist_t": list(out.hist_t),
                    "newton_res_hist": (list(out.trial.res_hist)
                                        if out.trial else []),
                    "consecutive": consec_rollbacks,
                })
                if (consec_rollbacks >= 2 and fallback is not None
                        and not switched):
                    switches.append({
                        "knot": k, "t": t, "step": step,
                        "after_rollbacks": consec_rollbacks,
                        "from": active.name, "to": fallback.name,
                    })
                    active = fallback
                    switched = True
                step *= 0.5
                if step < min_substep_frac - 1e-15:
                    fail(f"increment (knot {k}, t = {t:.4f}) failed "
                         f"below the substep floor {min_substep_frac} "
                         f"(last status: {out.status})",
                         out, k, t, t_next, step, dt)
    return CouplingResult(system=system, lambdas=lambdas,
                          strategy=strategy, records=records,
                          rollbacks=rollbacks,
                          strategy_switches=switches,
                          final_state=committed, n_commits=n_commits,
                          far=far, warm_start=warm_start, times=times)


# ------------------------------------------- condensed variant (prod)

def condensed_boundary_block(mesh, base):
    th = mesh.theta_nodes
    n_t = mesh.n_t
    S_pol = pb.dtn.nodal_dtn(th, mesh.R, base.G, base.kappa)
    c, s = np.cos(th), np.sin(th)
    j = np.arange(n_t)
    P = np.zeros((2 * n_t, 2 * n_t))
    P[j, 2 * j] = c
    P[j, 2 * j + 1] = s
    P[n_t + j, 2 * j] = -s
    P[n_t + j, 2 * j + 1] = c
    M = assembly.ring_mass(mesh, "outer")
    Bm = np.zeros((2 * n_t, 2 * n_t))
    Bm[0::2, 0::2] = M
    Bm[1::2, 1::2] = M
    return -(Bm @ P.T @ S_pol @ P)


def run_condensed(system, lambdas, K_bnd=None, newton_tol=1e-11,
                  max_iter=40, min_substep=1.0 / 64.0, check_front=True,
                  times=None):
    if K_bnd is None:
        K_bnd = condensed_boundary_block(system.mesh, system.base)
    bc = solver.CondensedOuter(K_bnd)
    run = solver.run_schedule(system, np.asarray(lambdas, dtype=float),
                              [bc] * len(lambdas), newton_tol=newton_tol,
                              max_iter=max_iter, min_substep=min_substep,
                              times=times)
    if check_front:
        for ex in run.executed:
            n_pl, r_front, margin, hit = plastic_front(system.mesh,
                                                       ex.trial.kappa)
            if hit:
                raise CouplingFailure(
                    f"condensed increment (knot {ex.knot}, t1 = "
                    f"{ex.t1:g}) plastic front reached the outermost "
                    f"element ring (r_front = {r_front:.6g})",
                    diagnostics={"knot": ex.knot, "t0": ex.t0,
                                 "t1": ex.t1,
                                 "lam": ex.trial.lam, "r_front": r_front,
                                 "margin": margin, "n_plastic": n_pl,
                                 "dt": ex.trial.diagnostics.get("dt"),
                                 "res_hist": list(ex.trial.res_hist),
                                 "failed_attempts":
                                     list(run.failed_attempts)})
    return run, K_bnd
