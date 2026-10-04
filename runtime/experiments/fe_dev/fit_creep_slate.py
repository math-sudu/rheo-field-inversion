"""Fit staged axial-creep data with a Maxwell-extended Nishihara chain."""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

DATA_DIR = Path(__file__).resolve().parent / "data" / "slate_creep"

PCT = 0.01  # strain % -> fraction
RESOLUTION_PCT_PER_H = 2.0e-4  # digitisation floor for a 25 h tail rate (%/h)
TAU_K_FLOOR, TAU_K_CEIL = 1.0e-3, 50.0  # h, optimiser bounds


@dataclass(frozen=True)
class GroupSpec:
    key: str
    beta_deg: float
    staged_fid: str        # staged axial creep figure (fit input)
    ss_fid: str            # stress-strain figure (E0 source)
    failure_stress: float
    peak_strain_pct: float


GROUPS = (
    GroupSpec("A_beta0", 0.0, "slate_0_creep", "slate_0_stress_strain", 115.0, 0.51),
    GroupSpec("B_beta30", 30.0, "slate_30_creep", "slate_30_stress_strain", 52.0, 0.52),
    GroupSpec("C_beta90", 90.0, "slate_90_creep", "slate_90_stress_strain", 166.0, 1.21),
)


@dataclass
class Level:
    sigma: float               # stage axial stress, MPa
    tau: np.ndarray            # within-stage time, h (as digitised)
    eps_pct: np.ndarray        # digitised strain, % (cumulative axis)
    plateau_from: int = 0      # first index of the plateau branch
    accelerating: bool = False
    tail_rate: float = 0.0     # %/h, last-half secant
    used: bool = True


@dataclass
class GroupFit:
    spec: GroupSpec
    levels: list[Level] = field(default_factory=list)
    E0_MPa: float = np.nan
    E_K: float = np.nan        # MPa
    tau_K: float = np.nan      # h
    eta_K: float = np.nan      # MPa*h
    inv_eta_M: float = np.nan  # 1/(MPa*h)
    inv_eta_vp: float = np.nan
    sigma_s: float = np.nan    # MPa
    offset: float = np.nan     # %, global baseline (bedding + digitisation)
    bedding_pct: float = np.nan  # level-1 plateau surplus over the chain
    r2: float = np.nan
    rmse_pct: float = np.nan
    notes: list[str] = field(default_factory=list)


_STRESS_RE = re.compile(r"(\d+(?:\.\d+)?)\s*MPa")


def _read_series(path: Path) -> dict[str, list[tuple[float, float]]]:
    out: dict[str, list[tuple[float, float]]] = {}
    with open(path, encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            out.setdefault(row["series"], []).append(
                (float(row["x"]), float(row["y"])))
    for pts in out.values():
        pts.sort()
    return out


def load_levels(spec: GroupSpec, data_dir: Path = DATA_DIR) -> list[Level]:
    """Parse the staged figure into stress-ordered levels.

    ``plateau_from`` marks the first sample of the plateau branch: the
    rise prefix (loading marker + mid-rise points) is every leading
    sample whose strain lies below the previous stage tail plus half
    the stage's plateau step.
    """
    series = _read_series(data_dir / f"{spec.staged_fid}.csv")
    levels = []
    for name, pts in series.items():
        m = _STRESS_RE.search(name)
        if not m:
            raise ValueError(f"{spec.staged_fid}: no stress in series {name!r}")
        arr = np.array(pts, dtype=float)
        tau = np.clip(arr[:, 0], 0.0, None)  # digitisation may give tiny t<0
        tau[0] = 0.0  # first point = loading-instant marker
        lv = Level(sigma=float(m.group(1)), tau=tau, eps_pct=arr[:, 1])
        lv.tail_rate = _tail_rate(lv)
        levels.append(lv)
    levels.sort(key=lambda lv: lv.sigma)

    prev_tail = 0.0
    for lv in levels:
        tail = float(lv.eps_pct[-1])
        cut = prev_tail + 0.5 * max(tail - prev_tail, 0.0)
        idx = 0
        while idx < lv.eps_pct.size - 1 and lv.eps_pct[idx] < cut:
            idx += 1
        lv.plateau_from = idx
        prev_tail = tail
        if abs(lv.sigma - spec.failure_stress) < 0.5:
            lv.accelerating = True
            lv.used = False
    return levels


def _tail_rate(lv: Level) -> float:
    n2 = max(2, lv.tau.size // 2)
    t, e = lv.tau[-n2:], lv.eps_pct[-n2:]
    if t[-1] <= t[0]:
        return float("nan")
    return float((e[-1] - e[0]) / (t[-1] - t[0]))


def fit_E0(spec: GroupSpec, data_dir: Path = DATA_DIR,
           frac_of_peak: float = 0.4) -> float:
    """Low-stress secant of the loading branch of the sigma-eps curve (MPa).

    The loading branch is the prefix up to peak stress; the secant runs
    from the first point to the last point below ``frac_of_peak`` * peak
    stress.  Under-estimates E0 (inter-stage creep is embedded in the
    strain axis) -- reported with that caveat.
    """
    series = _read_series(data_dir / f"{spec.ss_fid}.csv")
    (name, pts), = series.items()  # single curve per figure
    arr = np.array(pts, dtype=float)  # x = strain %, y = stress MPa
    stress = arr[:, 1]
    k_peak = int(np.argmax(stress))
    load = arr[: k_peak + 1]
    cap = frac_of_peak * stress[k_peak]
    seg = load[load[:, 1] <= cap]
    if seg.shape[0] < 2:
        seg = load[: max(2, load.shape[0] // 3)]
    dsig = seg[-1, 1] - seg[0, 1]
    deps = (seg[-1, 0] - seg[0, 0]) * PCT
    return float(dsig / deps)


def _stage_starts(levels: list[Level]) -> np.ndarray:
    """Absolute stage start times, h (stages assumed back-to-back)."""
    starts, t = [], 0.0
    for lv in levels:
        starts.append(t)
        t += float(lv.tau[-1])
    return np.array(starts)


def _assemble_observations(levels: list[Level], t_starts: np.ndarray):
    """Plateau-branch samples of used levels on the absolute time axis."""
    t_abs, eps, owner = [], [], []
    for k, lv in enumerate(levels):
        if not lv.used:
            continue
        for i in range(lv.plateau_from, lv.tau.size):
            t_abs.append(t_starts[k] + lv.tau[i])
            eps.append(lv.eps_pct[i])
            owner.append(k)
    return np.array(t_abs), np.array(eps), np.array(owner)


def _model_cumulative(t_abs: np.ndarray, levels: list[Level],
                      t_starts: np.ndarray, E_K: float, tau_K: float,
                      inv_eta_M: float, inv_eta_vp: float, sigma_s: float,
                      offset_pct: float) -> np.ndarray:
    """Cumulative creep strain (%), Boltzmann over stage increments."""
    out = np.full(t_abs.shape, offset_pct, dtype=float)
    sig_prev = 0.0
    for k, lv in enumerate(levels):
        dsig = lv.sigma - sig_prev
        sig_prev = lv.sigma
        tk = t_starts[k]
        act = t_abs >= tk
        if not np.any(act):
            continue
        dt = t_abs[act] - tk
        kel = dsig / E_K * (1.0 - np.exp(-dt / tau_K))
        # steady terms integrate the CURRENT stage stress over the stage
        # occupancy of [tk, t]; done incrementally: this stage holds
        # sigma_k from tk until the next stage start (or t if earlier).
        t_next = t_starts[k + 1] if k + 1 < len(levels) else np.inf
        hold = np.minimum(t_abs[act], t_next) - tk
        steady = (lv.sigma * inv_eta_M
                  + max(lv.sigma - sigma_s, 0.0) * inv_eta_vp) * hold
        out[act] += (kel + steady) / PCT
    return out


def _residuals(x: np.ndarray, t_abs, eps_obs, levels, t_starts,
               sigma_s: float) -> np.ndarray:
    log_EK, log_tauK, inv_eM, inv_evp, off = x
    model = _model_cumulative(t_abs, levels, t_starts, np.exp(log_EK),
                              np.exp(log_tauK), inv_eM, inv_evp,
                              sigma_s, off)
    return model - eps_obs


def fit_group(spec: GroupSpec, data_dir: Path = DATA_DIR) -> GroupFit:
    g = GroupFit(spec=spec, levels=load_levels(spec, data_dir))
    g.E0_MPa = fit_E0(spec, data_dir)
    g.notes.append("E0 from sigma-eps loading-branch low-stress secant; "
                   "inter-stage creep embedded -> under-estimate")

    used = [lv for lv in g.levels if lv.used]
    if len(used) < 2:
        g.notes.append("fewer than 2 non-accelerating levels; fit skipped")
        return g
    t_starts = _stage_starts(g.levels)
    t_abs, eps_obs, _ = _assemble_observations(g.levels, t_starts)

    # sigma_s grid: 0 .. failure stress, 1 MPa steps (finer than level gaps)
    grid = np.arange(0.0, spec.failure_stress + 0.5, 1.0)

    tail = np.array([lv.tail_rate for lv in used])
    rate_scale = max(np.nanmax(tail), RESOLUTION_PCT_PER_H) * PCT
    sig_max = used[-1].sigma
    # Multi-start over the Kelvin time constant (the loss is multi-modal
    # in tau_K; a single start can fall into a degenerate basin).
    starts = [np.array([np.log(2.0e5), np.log(tau0),
                        0.25 * rate_scale / sig_max,
                        0.5 * rate_scale / sig_max, 0.0])
              for tau0 in (0.02, 0.2, 2.0)]
    lb = np.array([np.log(1e2), np.log(TAU_K_FLOOR), 0.0, 0.0, -5.0])
    ub = np.array([np.log(1e8), np.log(TAU_K_CEIL), 1e-2, 1e-2, 5.0])

    best = None
    for sigma_s in grid:
        for x0 in starts:
            try:
                sol = least_squares(
                    _residuals, x0, bounds=(lb, ub),
                    args=(t_abs, eps_obs, g.levels, t_starts,
                          float(sigma_s)),
                    method="trf", max_nfev=2000)
            except Exception:
                continue
            sse = float(np.sum(sol.fun ** 2))
            if best is None or sse < best[0]:
                best = (sse, float(sigma_s), sol)
    if best is None:
        g.notes.append("least-squares failed on every sigma_s candidate")
        return g

    sse, g.sigma_s, sol = best
    g.E_K = float(np.exp(sol.x[0]))
    g.tau_K = float(np.exp(sol.x[1]))
    g.eta_K = g.E_K * g.tau_K
    g.inv_eta_M, g.inv_eta_vp = float(sol.x[2]), float(sol.x[3])
    g.offset = float(sol.x[4])

    g.rmse_pct = float(np.sqrt(sse / eps_obs.size))
    ss_tot = float(np.sum((eps_obs - eps_obs.mean()) ** 2))
    g.r2 = 1.0 - sse / ss_tot if ss_tot > 0 else np.nan

    # Level-1 bedding surplus: observed first plateau tail minus the
    # linear-chain prediction WITHOUT the global offset.
    lv1 = used[0]
    pred1 = _model_cumulative(
        np.array([t_starts[g.levels.index(lv1)] + lv1.tau[-1]]), g.levels,
        t_starts, g.E_K, g.tau_K, g.inv_eta_M, g.inv_eta_vp, g.sigma_s,
        0.0)[0]
    g.bedding_pct = float(lv1.eps_pct[-1] - pred1)

    # ---- identifiability annotations (reported, not absorbed) ----
    floor = RESOLUTION_PCT_PER_H * PCT
    accel = [lv for lv in g.levels if lv.accelerating]
    hi = accel[0].sigma if accel else np.inf

    if g.tau_K <= 1.5 * TAU_K_FLOOR:
        first_plateau_tau = min(float(lv.tau[lv.plateau_from])
                                for lv in used if lv.plateau_from < lv.tau.size)
        g.notes.append(f"tau_K collapsed to the optimiser floor: the "
                       f"primary transient hides inside the compressed "
                       f"rise; upper bound ~{first_plateau_tau:.2f} h "
                       f"(first plateau sample)")
    if g.inv_eta_M * sig_max < floor:
        eta_lb = sig_max * float(used[-1].tau[-1]) / (0.005 * PCT)
        g.notes.append(f"eta_M under-constrained by the lab hold window "
                       f"(fitted steady term below digitisation floor); "
                       f"lower bound ~{eta_lb:.1e} MPa*h")

    vp_rate = max(sig_max - g.sigma_s, 0.0) * g.inv_eta_vp
    if vp_rate < floor:
        # The Bingham branch fitted to nothing: any sigma_s on the flat
        # SSE plateau is arbitrary -- do NOT report a point estimate.
        g.sigma_s = float("nan")
        g.inv_eta_vp = 0.0
        g.notes.append(f"sigma_s / eta_vp not identifiable from "
                       f"plateau-branch rates (viscoplastic steady term "
                       f"below digitisation floor); upper bound from the "
                       f"failure stage: sigma_s <= {hi:g} MPa "
                       f"(sub-resolution activation at lower stages "
                       f"cannot be excluded)")
    elif not np.isnan(g.sigma_s) and g.sigma_s <= used[0].sigma:
        g.notes.append("fitted sigma_s at/below all used stage stresses: "
                       "eta_M/eta_vp collinear on this design; steady "
                       "compliance is the identified combination")
    if abs(g.bedding_pct) > 0.02:
        g.notes.append(f"level-1 bedding-in surplus {g.bedding_pct:+.3f} % "
                       f"absorbed by the global offset (first-loading "
                       f"compaction; not a chain parameter)")
    if accel:
        g.notes.append(f"tertiary creep at {hi:g} MPa excluded from fit "
                       f"domain (classical chain has no tertiary branch); "
                       f"viscoplastic activation evidence: accelerating "
                       f"tail rate {accel[0].tail_rate:.2e} %/h")
    return g


def write_outputs(fits: list[GroupFit], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "nishihara_params.csv", "w", newline="",
              encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["group", "beta_deg", "E0_GPa", "E_K_GPa", "eta_K_GPa_h",
                    "tau_K_h", "eta_M_GPa_h", "eta_vp_GPa_h", "sigma_s_MPa",
                    "sigma_s_over_failure", "offset_pct", "bedding_pct",
                    "n_levels_used", "r2", "rmse_pct", "notes"])
        for g in fits:
            eta_M = (1.0 / g.inv_eta_M / 1e3) if g.inv_eta_M and g.inv_eta_M > 0 else np.inf
            eta_vp = (1.0 / g.inv_eta_vp / 1e3) if g.inv_eta_vp and g.inv_eta_vp > 0 else np.inf
            w.writerow([
                g.spec.key, g.spec.beta_deg, f"{g.E0_MPa / 1e3:.2f}",
                f"{g.E_K / 1e3:.3f}", f"{g.eta_K / 1e3:.4f}",
                f"{g.tau_K:.4f}", f"{eta_M:.3e}", f"{eta_vp:.3e}",
                f"{g.sigma_s:.1f}",
                f"{g.sigma_s / g.spec.failure_stress:.2f}",
                f"{g.offset:.4f}", f"{g.bedding_pct:.4f}",
                sum(lv.used for lv in g.levels),
                f"{g.r2:.4f}", f"{g.rmse_pct:.4f}",
                " | ".join(g.notes),
            ])

    with open(out_dir / "level_rates.csv", "w", newline="",
              encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["group", "sigma_MPa", "n_points", "plateau_from",
                    "hold_h", "tail_rate_pct_per_h", "accelerating",
                    "used_in_fit"])
        for g in fits:
            for lv in g.levels:
                w.writerow([g.spec.key, lv.sigma, lv.tau.size,
                            lv.plateau_from, f"{lv.tau[-1]:.2f}",
                            f"{lv.tail_rate:.4e}", int(lv.accelerating),
                            int(lv.used)])

    with open(out_dir / "fit_curves.csv", "w", newline="",
              encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["group", "sigma_MPa", "t_abs_h", "tau_h",
                    "eps_obs_pct", "eps_fit_pct"])
        for g in fits:
            if np.isnan(g.E_K):
                continue
            t_starts = _stage_starts(g.levels)
            t_abs, eps_obs, owner = _assemble_observations(g.levels,
                                                           t_starts)
            model = _model_cumulative(t_abs, g.levels, t_starts, g.E_K,
                                      g.tau_K, g.inv_eta_M, g.inv_eta_vp,
                                      np.nan_to_num(g.sigma_s, nan=np.inf),
                                      g.offset)
            for t, k, eo, ef in zip(t_abs, owner, eps_obs, model):
                w.writerow([g.spec.key, g.levels[k].sigma, f"{t:.4f}",
                            f"{t - t_starts[k]:.4f}", f"{eo:.5f}",
                            f"{ef:.5f}"])


def main(argv=None) -> list[GroupFit]:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", type=Path, default=DATA_DIR)
    ap.add_argument("--out", type=Path,
                    default=Path("results/fe_dev/slate_creep_fit"))
    args = ap.parse_args(argv)

    fits = [fit_group(spec, args.data) for spec in GROUPS]
    write_outputs(fits, args.out)

    for g in fits:
        print(f"== {g.spec.key} (beta={g.spec.beta_deg:g} deg)")
        print(f"   E0 ~ {g.E0_MPa / 1e3:.1f} GPa (loading-secant, low bias)")
        print(f"   E_K = {g.E_K / 1e3:.2f} GPa  tau_K = {g.tau_K:.4f} h  "
              f"eta_K = {g.eta_K / 1e3:.4f} GPa*h")
        eM = 1.0 / g.inv_eta_M / 1e3 if g.inv_eta_M > 0 else np.inf
        evp = 1.0 / g.inv_eta_vp / 1e3 if g.inv_eta_vp > 0 else np.inf
        print(f"   eta_M = {eM:.3e} GPa*h  eta_vp = {evp:.3e} GPa*h  "
              f"sigma_s = {g.sigma_s:.0f} MPa "
              f"({g.sigma_s / g.spec.failure_stress:.2f} of failure)")
        print(f"   offset = {g.offset:.4f} %  bedding(level-1) = "
              f"{g.bedding_pct:.4f} %")
        print(f"   R2 = {g.r2:.4f}  RMSE = {g.rmse_pct:.4f} %  "
              f"levels used = {sum(lv.used for lv in g.levels)}")
        for n in g.notes:
            print(f"   note: {n}")
    return fits


if __name__ == "__main__":
    main()
