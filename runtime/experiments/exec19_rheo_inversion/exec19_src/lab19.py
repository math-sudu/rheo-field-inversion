
from __future__ import annotations

import csv
import dataclasses

import numpy as np

from . import _paths
from exec8_src.information import coordinate_crb, fisher_condition

import fit_creep_slate as labfit                          # noqa: E402

PCT = labfit.PCT
CHI2_95 = 3.841

PARAMS_CSV = _paths.RESULTS / "fe_dev" / "slate_creep_fit" \
    / "nishihara_params.csv"


CONTAINER_BOUND_1D_GPAH = {"A_beta0": 5.1e4, "C_beta90": 3.2e4}
WEAK_POINT_1D_GPAH = {"B_beta30": 1.184e4}

# reduced-frame optimiser bounds (fit script lane + self-set eta_vp)
LOG_EK_BOUNDS = (np.log(1e2), np.log(1e8))          # MPa
LOG_TAUK_BOUNDS = (np.log(labfit.TAU_K_FLOOR), np.log(labfit.TAU_K_CEIL))
LOG_EVP_BOUNDS = (np.log(1e2), np.log(1e12))        # MPa*h
OFFSET_BOUNDS = (-5.0, 5.0)                          # %


@dataclasses.dataclass
class LabGroup:
    key: str
    levels: list
    t_starts: np.ndarray
    t_abs: np.ndarray
    eps_obs: np.ndarray
    booked: dict           # MPa / MPa*h / h / % frame
    sigma_lab: float       # % (booked fit RMSE)
    sigma_s: float         # MPa (fixed; inf = branch off)
    vp_on: bool


def _read_booked():
    rows = {}
    with open(PARAMS_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows[row["group"]] = row
    return rows


def load_group(key) -> LabGroup:
    spec = {g.key: g for g in labfit.GROUPS}[key]
    levels = labfit.load_levels(spec, labfit.DATA_DIR)
    t_starts = labfit._stage_starts(levels)
    t_abs, eps_obs, _ = labfit._assemble_observations(levels, t_starts)
    b = _read_booked()[key]
    eta_vp = float(b["eta_vp_GPa_h"])
    sig_s = float(b["sigma_s_MPa"])
    vp_on = np.isfinite(eta_vp) and np.isfinite(sig_s)
    booked = {
        "E_K": float(b["E_K_GPa"]) * 1e3,
        "tau_K": float(b["tau_K_h"]),
        "eta_M": float(b["eta_M_GPa_h"]) * 1e3,
        "eta_vp": (eta_vp * 1e3 if vp_on else np.inf),
        "offset": float(b["offset_pct"]),
    }
    return LabGroup(key=key, levels=levels, t_starts=t_starts,
                    t_abs=t_abs, eps_obs=eps_obs, booked=booked,
                    sigma_lab=float(b["rmse_pct"]),
                    sigma_s=(sig_s if vp_on else np.inf), vp_on=vp_on)


def coord_names(g: LabGroup):
    base = ["ln_E_K", "ln_tau_K", "ln_eta_M"]
    if g.vp_on:
        base.append("ln_eta_vp")
    return base + ["offset"]


def model_and_jac(g: LabGroup, E_K, tau_K, eta_M, eta_vp, offset):
    """Cumulative creep model (%) + analytic Jacobian in the log
    frame; mirrors labfit._model_cumulative stage by stage (locked in
    tests against the script function itself)."""
    t = g.t_abs
    n = t.size
    m = np.full(n, offset, dtype=float)
    cols = {k: np.zeros(n) for k in
            ("ln_E_K", "ln_tau_K", "ln_eta_M", "ln_eta_vp")}
    sig_prev = 0.0
    for k, lv in enumerate(g.levels):
        dsig = lv.sigma - sig_prev
        sig_prev = lv.sigma
        tk = g.t_starts[k]
        act = t >= tk
        if not np.any(act):
            continue
        dt = t[act] - tk
        r = dt / tau_K
        kel = dsig / E_K * (1.0 - np.exp(-r))
        t_next = g.t_starts[k + 1] if k + 1 < len(g.levels) else np.inf
        hold = np.minimum(t[act], t_next) - tk
        mx = lv.sigma * hold / eta_M
        bg = (max(lv.sigma - g.sigma_s, 0.0) * hold / eta_vp
              if g.vp_on else 0.0)
        m[act] += (kel + mx + bg) / PCT
        cols["ln_E_K"][act] += -kel / PCT
        cols["ln_tau_K"][act] += dsig / E_K * (-r * np.exp(-r)) / PCT
        cols["ln_eta_M"][act] += -mx / PCT
        if g.vp_on:
            cols["ln_eta_vp"][act] += -bg / PCT
    names = coord_names(g)
    J = np.zeros((n, len(names)))
    for j, nm in enumerate(names):
        J[:, j] = cols[nm] if nm in cols else 1.0   # offset column
    return m, J


def model_at(g: LabGroup, **kw):
    p = dict(g.booked)
    p.update(kw)
    return model_and_jac(g, p["E_K"], p["tau_K"], p["eta_M"],
                         p["eta_vp"], p["offset"])


def fisher(g: LabGroup, at=None):
    """(names, sv, cond, crb_log) at ``at`` (default booked point).
    crb is the log-frame (relative) sigma per coordinate; the offset
    row is in % units."""
    m, J = model_at(g, **(at or {}))
    Jn = J / g.sigma_lab
    sv = np.linalg.svd(Jn, compute_uv=False)
    crb = coordinate_crb(Jn)
    cond = fisher_condition(Jn)
    sse = float(np.sum((m - g.eps_obs) ** 2))
    return coord_names(g), sv, cond, crb, sse


def _refit(g: LabGroup, eta_M, x0):
    from scipy.optimize import least_squares
    lb = [LOG_EK_BOUNDS[0], LOG_TAUK_BOUNDS[0]]
    ub = [LOG_EK_BOUNDS[1], LOG_TAUK_BOUNDS[1]]
    if g.vp_on:
        lb.append(LOG_EVP_BOUNDS[0])
        ub.append(LOG_EVP_BOUNDS[1])
    lb.append(OFFSET_BOUNDS[0])
    ub.append(OFFSET_BOUNDS[1])

    def unpack(x):
        E_K, tau_K = np.exp(x[0]), np.exp(x[1])
        eta_vp = np.exp(x[2]) if g.vp_on else np.inf
        return E_K, tau_K, eta_vp, x[-1]

    def resid(x):
        E_K, tau_K, eta_vp, off = unpack(x)
        m, _ = model_and_jac(g, E_K, tau_K, eta_M, eta_vp, off)
        return m - g.eps_obs

    sol = least_squares(resid, np.clip(x0, lb, ub), bounds=(lb, ub),
                        method="trf", max_nfev=2000)
    return float(np.sum(sol.fun ** 2)), sol.x


def profile_eta_M(g: LabGroup, grid_mpah=None, n_jitter=2, seed=0):
    """(grid, sse) profile of ln eta_M with the remaining coordinates
    re-optimised per point (booked + jittered starts, min kept)."""
    if grid_mpah is None:
        grid_mpah = np.geomspace(1e5, 1e11, 25)     # 1e2..1e8 GPa*h
    rng = np.random.default_rng(seed)
    b = g.booked
    x0 = [np.log(b["E_K"]), np.log(b["tau_K"])]
    if g.vp_on:
        x0.append(np.log(b["eta_vp"]))
    x0.append(b["offset"])
    x0 = np.array(x0)
    out = []
    for eta in grid_mpah:
        best = None
        starts = [x0]
        for _ in range(n_jitter):
            j = x0.copy()
            j[:-1] += rng.uniform(-0.5, 0.5, size=len(x0) - 1)
            j[-1] += rng.uniform(-0.05, 0.05)
            starts.append(j)
        for s in starts:
            sse, _ = _refit(g, float(eta), s)
            if best is None or sse < best:
                best = sse
        out.append(best)
    return np.asarray(grid_mpah), np.asarray(out)


def profile_read(grid, sse, sigma_lab):
    """Lower-crossing / openness read of the eta_M profile.

    Returns (sse_min, dchi2 array, lo_cross_mpah, hi_open) with
    lo_cross the Delta-chi2 = 3.841 crossing on the LOW side (linear
    interpolation in ln eta_M; None when the curve never exceeds the
    threshold there) and hi_open True when the high end stays below
    the threshold (lower-bound-only shape)."""
    sse_min = float(np.min(sse))
    d = (sse - sse_min) / sigma_lab ** 2
    i_min = int(np.argmin(sse))
    lo_cross = None
    for i in range(i_min, 0, -1):        # walk left, find last below
        if d[i - 1] > CHI2_95 >= d[i]:
            a0, a1 = np.log(grid[i]), np.log(grid[i - 1])
            c0, c1 = d[i], d[i - 1]
            a = a0 + (CHI2_95 - c0) * (a1 - a0) / (c1 - c0)
            lo_cross = float(np.exp(a))
            break
    hi_open = bool(d[-1] <= CHI2_95)
    return sse_min, d, lo_cross, hi_open
