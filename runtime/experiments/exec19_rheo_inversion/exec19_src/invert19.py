"""Multi-start LM estimator on the viscous field forward (EXEC-19
batch 2: recovery grid / profile-likelihood machinery).

Structure port of ``exec8_src.invert`` (frozen protocol section 2 /
correction note 3): optimizer transforms (log for the multi-decade
coordinates, affine box-normalization for the sub-decade ones),
N_STARTS = 6 Sobol starts drawn uniformly in the PHYSICAL box then
transformed (exec8 convention), LM damping ladder (x10 / /3, 12 inner
trials), bounds by clamping with zero outside-chain, stationarity
criteria (relative step / gradient / SSE plateau / no-descent),
MAX_ITER = 60, best-final-SSE-of-starts kept.

Differences against exec8's analytic ``_pack`` (each forced by the
viscous surface, not a style choice):

* residual and Jacobian evaluations are SPLIT: a residual costs one
  FE forward on the fixed dense knot axis (~2-3 s at 8x32) while the
  Jacobian costs a reverse replay sweep (identify19.field_jacobian,
  ~20-25 s), so the LM inner trial loop re-evaluates residuals only
  and the linearization happens once per accepted iterate;
* the FE forward can FAIL at pathological iterates (Newton substep
  floor); a failed start evaluation marks the start failed and moves
  on, a failed trial step counts as a rejected step (damping x10) --
  exec8's closed-form operator could not fail so its port needs this
  guard;
* K_bnd is assembled once per evaluator at a reference E and scaled
  exactly by E/E_ref per evaluation (elastic far-field linearity,
  batch-1 lock row: kb_E_scaling_dev = 0.0).

Transforms (PROTOCOL section 2): log for s_star, r_K, tau_K, tau_M,
tau_vp; box-affine for K0, r_s, and lam0 (lam0 follows the exec8
invert lane -- its box spans less than a decade).

Optimization convergence is NOT parameter-identification success
(project discipline); recovery quality is read from the booked rows
only (no evaluative gates).
"""

from __future__ import annotations

import dataclasses
import time

import numpy as np
import torch

from . import _paths  # noqa: F401
from . import identify19
from .field19 import CHANNELS, FieldObservation, FieldStamps, \
    build_field_geometry, field_knot_axis
from .scenario import RheoScenario

from exec9_src import coupled as x9coupled                 # noqa: E402
from fedev_src import coupling as fcoupling                # noqa: E402
from fedev_src import noncirc as fnoncirc                  # noqa: E402

N_STARTS = 6
MAX_ITER = 60
SEED = 20260712

ALL_COORDS = ("s_star", "K0", "lam0", "r_K", "tau_K", "tau_M", "r_s",
              "tau_vp")
# log-transformed coordinate names; the batch-4 tier-B additions
# (r_c/chi_c/gamma_c/gamma_phi, pre-run note 5 item 2) are disjoint
# from the tier-C namespace, so the frozen tier-C transform behavior
# is bit-identical (locked by tests/test_invert19.py round-trip)
LOG_COORDS = ("s_star", "r_K", "tau_K", "tau_M", "tau_vp",
              "r_c", "chi_c", "gamma_c", "gamma_phi")


# ------------------------------------------------------- transforms

def z_of(name, v, box):
    lo, hi = box[name]
    if name in LOG_COORDS:
        return float(np.log(v))
    return float((v - lo) / (hi - lo))


def theta_and_chain(name, z, box):
    """Physical value + d theta / d z with clamping (zero outside)."""
    lo, hi = box[name]
    if name in LOG_COORDS:
        zl, zh = np.log(lo), np.log(hi)
        zc = min(max(z, zl), zh)
        th = float(np.exp(zc))
        return th, (th if zl < z < zh else 0.0)
    zc = min(max(z, 0.0), 1.0)
    th = lo + (hi - lo) * zc
    return float(th), ((hi - lo) if 0.0 < z < 1.0 else 0.0)


# -------------------------------------------------------- evaluator

class FieldEvaluator:
    """One (geometry, mesh, stamps, knot axis) context; theta ->
    model / residual / Jacobian on the field tier.

    ``stamps`` may be a truncated variant (observation-window stage);
    the knot axis defaults to the batch-1 fixed dense axis of the FULL
    section span so runs stay comparable across window truncations.
    """

    def __init__(self, section="DK0+935", n_r=8, n_t=32, stamps=None,
                 axis=None, axis_kw=None, sigma_section=None, wh=None,
                 geom=None, geometry_options=None):
        if geom is not None and (wh is not None or geometry_options):
            raise ValueError("Select a geometry object or geometry options, not both")
        self.geom = geom if geom is not None else build_field_geometry(
            wh=wh, **(geometry_options or {}))
        self.stamps = stamps if stamps is not None \
            else FieldStamps.load(section)
        self.n_r, self.n_t = n_r, n_t
        self.mesh = x9coupled.build_mesh(self.geom, n_r=n_r, n_t=n_t,
                                         r_level=x9coupled.R_LEVEL)
        from .field19 import channel_matrix
        self.W = channel_matrix(self.mesh, self.geom)
        self.obs = FieldObservation(self.stamps, self.W,
                                    sigma_section=sigma_section)
        if axis is None:
            span = FieldStamps.load(section).span() \
                if stamps is not None else self.stamps.span()
            axis = field_knot_axis(span, **(axis_kw or {}))
        self.lams, self.times = axis
        # reference K_bnd at an arbitrary anchor E (exact E-linearity)
        scn_ref = RheoScenario(s_star=3.0e-2, K0=1.0, r_K=0.35,
                               tau_K=2.0, tau_M=200.0)
        sys_ref = self._system(scn_ref)
        self._E_ref = float(sys_ref.base.E)
        self._K_ref = fcoupling.condensed_boundary_block(
            self.mesh, sys_ref.base)
        self.sigma = identify19.sigma_vector(self.obs)
        self.n_forward = 0
        self.n_jac = 0

    def _system(self, scn):
        return fnoncirc.NonCircFESystem(self.mesh,
                                        scn.base(a=self.geom.R),
                                        scn.material())

    # theta dict -> (scenario, lam0)
    @staticmethod
    def scenario_of(theta):
        kw = {k: float(v) for k, v in theta.items() if k != "lam0"}
        return RheoScenario(**kw), float(theta["lam0"])

    def forward(self, theta):
        """One committed FE run at theta (fixed dense axis)."""
        scn, _ = self.scenario_of(theta)
        system = self._system(scn)
        K_bnd = (float(scn.E) / self._E_ref) * self._K_ref
        run, _ = fcoupling.run_condensed(
            system, np.asarray(self.lams, float), K_bnd=K_bnd,
            times=np.asarray(self.times, float))
        self.n_forward += 1
        return run

    def model_np(self, run, lam0):
        """Flat own-clock observation vector from the committed run
        (numpy lane; equals the torch path -- locked in tests)."""
        from exec8_src import schedule as x8sched
        t_commit = np.array([ex.trial.t for ex in run.executed])
        U = np.stack([self.W @ np.asarray(ex.trial.u)
                      for ex in run.executed])
        t_off = x8sched.T_REL * (np.log(
            (1.0 - x8sched.LAM_F) / (1.0 - lam0))) ** (1.0
                                                       / x8sched.BETA)
        out = []
        for r, ch in enumerate(CHANNELS):
            st = self.stamps.stamps[ch]
            tq = t_off + (st - self.stamps.stamp0)
            v = np.interp(tq, t_commit, U[:, r])
            v0 = np.interp(t_off + float(st[0] - self.stamps.stamp0),
                           t_commit, U[:, r])
            out.append(v - v0)
        return np.concatenate(out)

    def model(self, theta):
        run = self.forward(theta)
        return self.model_np(run, float(theta["lam0"])), run

    def jacobian(self, run, theta, names):
        """(model_vec, J) via the reverse replay (batch-1 locked)."""
        scn, lam0 = self.scenario_of(theta)
        vals, J = identify19.field_jacobian(run, scn, lam0, names,
                                            self.obs)
        self.n_jac += 1
        return vals, J

    def split_by_channel(self, flat):
        out, k = {}, 0
        for ch in CHANNELS:
            n = len(self.stamps.stamps[ch])
            out[ch] = flat[k:k + n]
            k += n
        return out

    def synthesize(self, model_flat, seed):
        """model + iid per-channel Gaussian noise (SIGMA_MM table),
        drawn in CHANNELS order (frozen chain)."""
        rng = np.random.default_rng(seed)
        parts = []
        for ch, m in self.split_by_channel(model_flat).items():
            parts.append(m + self.obs.sig[ch]
                         * rng.standard_normal(len(m)))
        return np.concatenate(parts)


# -------------------------------------------------------------- fit

@dataclasses.dataclass
class FitResult:
    theta_hat: dict
    sse: float
    n_iter: int
    converged: bool
    start_sses: list
    n_forward: int
    n_jac: int
    n_fail: int
    wall_s: float


def _residual(ev, free, fixed, z, y, w, box):
    theta = dict(fixed)
    chain = np.zeros(len(free))
    for i, n in enumerate(free):
        theta[n], chain[i] = theta_and_chain(n, float(z[i]), box)
    m, run = ev.model(theta)
    return theta, chain, (m - y) * w, run


def fit(ev: FieldEvaluator, free, fixed, y_obs, box=None,
        n_starts=N_STARTS, seed=SEED, extra_starts=()) -> FitResult:
    """Multi-start LM over ``free`` (others fixed by ``fixed``).

    ``extra_starts``: iterable of {coord: value} dicts prepended to the
    Sobol starts (profile-continuation warm starts; ``n_starts=0``
    runs the warm starts alone).
    """
    from scipy.stats import qmc
    box = dict(identify19.BOX if box is None else box)
    y = np.asarray(y_obs, dtype=float)
    w = 1.0 / ev.sigma
    d = len(free)
    starts = []
    for th in extra_starts:
        starts.append(np.array([z_of(n, float(th[n]), box)
                                for n in free]))
    if n_starts:
        sob = qmc.Sobol(d, scramble=True, seed=seed)
        pts = sob.random(n_starts)
        for p in pts:
            z0 = []
            for i, n in enumerate(free):
                lo, hi = box[n]
                v = lo + (hi - lo) * p[i]  # uniform in physical box
                z0.append(z_of(n, v, box))
            starts.append(np.array(z0))
    t_wall = time.perf_counter()
    ev.n_forward = ev.n_jac = 0
    n_fail = 0
    best = None
    start_sses = []
    for z0 in starts:
        z = z0.copy()
        lam = 1e-3
        converged = False
        it = 0
        try:
            theta, chain, r, run = _residual(ev, free, fixed, z, y,
                                             w, box)
        except Exception:
            n_fail += 1
            start_sses.append(float("inf"))
            continue
        sse0 = float(r @ r)
        for it in range(1, MAX_ITER + 1):
            _, Jth = ev.jacobian(run, theta, tuple(free))
            J = Jth * w[:, None] * chain[None, :]
            g = J.T @ r
            H = J.T @ J
            sse_new = sse0
            stepped = False
            for _ in range(12):
                D = np.diag(np.maximum(np.diag(H), 1e-12))
                try:
                    step = np.linalg.solve(H + lam * D, -g)
                except np.linalg.LinAlgError:
                    lam *= 10.0
                    continue
                z_new = z + step
                try:
                    th_new, ch_new, r_new, run_new = _residual(
                        ev, free, fixed, z_new, y, w, box)
                except Exception:
                    n_fail += 1
                    lam *= 10.0
                    continue
                sse_new = float(r_new @ r_new)
                if sse_new < sse0:
                    lam = max(lam / 3.0, 1e-12)
                    stepped = True
                    break
                lam *= 10.0
            if not stepped:
                converged = True   # no descent direction: stationary
                break
            rel_step = float(np.linalg.norm(step)
                             / (np.linalg.norm(z) + 1e-12))
            z = z_new
            theta, chain, r, run = th_new, ch_new, r_new, run_new
            if (rel_step < 1e-10
                    or float(np.linalg.norm(g)) < 1e-12
                    or (sse0 - sse_new) < 1e-11 * max(sse0, 1e-300)):
                converged = True
                sse0 = sse_new
                break
            sse0 = sse_new
        sse_fin = float(r @ r)
        start_sses.append(sse_fin)
        res = FitResult(
            theta_hat={k: float(v) for k, v in theta.items()},
            sse=sse_fin, n_iter=it, converged=converged,
            start_sses=[], n_forward=0, n_jac=0, n_fail=0, wall_s=0.0)
        if best is None or res.sse < best.sse:
            best = res
    if best is None:
        raise RuntimeError("every start failed in the FE forward")
    best.start_sses = start_sses
    best.n_forward = ev.n_forward
    best.n_jac = ev.n_jac
    best.n_fail = n_fail
    best.wall_s = time.perf_counter() - t_wall
    return best


def truncate_stamps(stamps: FieldStamps, frac) -> FieldStamps:
    """Window truncation: keep per-channel stamps within ``frac`` of
    the FULL section span from the section clock zero (own-clock
    first stamps always survive -- they anchor the increments)."""
    span = stamps.span()
    cut = stamps.stamp0 + frac * span
    st, y = {}, {}
    for ch, t in stamps.stamps.items():
        keep = t <= cut + 1e-9
        keep[0] = True
        st[ch] = t[keep]
        y[ch] = stamps.y_mm[ch][keep]
    return FieldStamps(section=stamps.section, stamps=st, y_mm=y,
                       stamp0=stamps.stamp0)
