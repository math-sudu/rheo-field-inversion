
from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline

from . import _paths  # noqa: F401  (sys.path side effect)

from exec8_src import geometry as geo                      # noqa: E402
from exec8_src import schedule                             # noqa: E402
from fedev_src import coupling as fcoupling                # noqa: E402
from fedev_src import noncirc                              # noqa: E402
from fedev_src.params import BaseParams, MohrCoulomb       # noqa: E402



# arch map is strongly ECCENTRIC (origin at the invert foot, wall
# radius extrema 0.002 .. 10.50 m at R_map = 5.02), so r_level = 2

# r_level = 2.5 tangles the blend mesh (min detJ < 0 at every probed
# refinement).  The guard clauses outrank the numeric level; the first
# guard-clean level is r_level = 3.0 (clearance 1.435, min detJ
# 2.87e-2 at 16x64) -- exec7d's own R3 sensitivity lane.  Measured
# probe table + erratum note in the run report.
N_R, N_T = 16, 64
R_LEVEL = 3.0            # R_Gamma / R_map (erratum note above)
N_INC = 8
ELASTIC_C_MPA = 1.0e9    # MC with huge cohesion == elastic (asserted)
MC_PHI_DEG, MC_PSI_DEG = 30.0, 10.0   # [convention] params.py L370
MC_C_OVER_SIGMA_V = 0.3  # [convention] keeps exec7d's c/sigma_v = 1.5/5


def build_map(geom: "geo.FrozenGeometry") -> noncirc.LaurentMap:
    m = noncirc.LaurentMap(geom.R, tuple(geom.c))
    th = 2.0 * np.pi * np.arange(4096) / 4096
    zeta = np.exp(1j * th)
    # independent recompute of the exec6b/exec8 boundary formula
    z_ref = geom.R * (zeta + sum(cj * zeta ** (-j)
                                 for j, cj in enumerate(geom.c)))
    dev = float(np.max(np.abs(m.wall_points(th) - z_ref)))
    if dev > 1e-12:
        raise AssertionError(f"map twin cross-check failed: {dev:.3e}")
    # Projection error is numerical geometry error. It is recorded separately
    # from observation noise and does not establish engineering applicability.
    dev_obs = float(np.max(np.abs(
        m.wall_points(geom.theta_points) - geom.z_points)))
    if dev_obs > 1.05 * max(geom.fit_max_dist_m,
                           geom.provenance["point_projection_max_m"]):
        raise AssertionError(
            f"observation points deviate from the mapped wall by "
            f"{dev_obs:.3e} m exceeds the mapping tolerance "
            f"{geom.fit_max_dist_m:.3e} m")
    return m


def assert_engineering_upright(geom: "geo.FrozenGeometry") -> None:
    """Crown on top, consistently ordered chord endpoints in the model frame."""
    m = noncirc.LaurentMap(geom.R, tuple(geom.c))
    th = 2.0 * np.pi * np.arange(4096) / 4096
    wall = m.wall_points(th)
    crown = geom.z_points[-1]
    if not crown.imag >= wall.imag.max() - max(geom.fit_max_dist_m, 1e-6):
        raise AssertionError("crown is not the topmost wall point")
    zl, zr = geom.z_points[:3], geom.z_points[3:6]
    if not np.all(zl.real < zr.real):
        raise AssertionError("chord endpoints are not ordered left to right")


def build_mesh(geom: "geo.FrozenGeometry", n_r: int = N_R,
               n_t: int = N_T,
               r_level: float = R_LEVEL) -> noncirc.MappedAnnulusMesh:
    sm = build_map(geom)
    return noncirc.MappedAnnulusMesh(sm, r_level * geom.R, n_r, n_t)


def channels_from_u(u: np.ndarray, mesh: noncirc.MappedAnnulusMesh,
                    geom: "geo.FrozenGeometry") -> dict[str, float]:
    idx = mesh.ring_dofs("inner")
    ux = np.asarray(u)[idx[0::2]]
    uy = np.asarray(u)[idx[1::2]]
    th = mesh.theta_nodes
    thp = np.concatenate([th, [2.0 * np.pi]])

    def interp(vals, t):
        vp = np.concatenate([vals, vals[:1]])
        return CubicSpline(thp, vp, bc_type="periodic")(
            np.mod(t, 2.0 * np.pi))

    uv = np.r_[interp(ux, geom.theta_points), interp(uy, geom.theta_points)]
    values = geo.observation_matrix(geom, returned_frame="physical") @ uv * 1000.0
    return dict(zip(geo.CHANNELS, map(float, values)))


def run_condensed_case(mesh, geom, E_mpa: float, nu: float, K0: float,
                       sigma_v_mpa: float, material: str = "elastic",
                       lambdas=None) -> dict:
    """One condensed-path coupled run; channels + diagnostics.

    material = "elastic" (MC, huge cohesion, asserted zero plastic QPs)
    or "mc" (perfect MC at the [convention] strength lane)."""
    if lambdas is None:
        lambdas = np.linspace(0.0, 1.0, N_INC + 1)
    base = BaseParams(sigma_v=sigma_v_mpa, K0=K0, E=E_mpa, nu=nu)
    if material == "elastic":
        c = ELASTIC_C_MPA
    elif material == "mc":
        c = MC_C_OVER_SIGMA_V * sigma_v_mpa
    else:
        raise ValueError(material)
    mat = MohrCoulomb(c=c, phi=float(np.deg2rad(MC_PHI_DEG)),
                      psi=float(np.deg2rad(MC_PSI_DEG)))
    system = noncirc.NonCircFESystem(mesh, base, mat)
    try:
        run, _ = fcoupling.run_condensed(system, np.asarray(lambdas))
    except fcoupling.CouplingFailure as exc:
        return {"status": "failed", "fail_reason": str(exc),
                "channels": None}
    st = run.final_state
    n_pl, r_front, margin, _ = fcoupling.plastic_front(mesh, st.kappa)
    if material == "elastic" and n_pl != 0:
        raise AssertionError(
            f"elastic row developed {n_pl} plastic QPs -- cohesion "
            "convention broken")
    met = noncirc.wall_disp_metrics(st.u, mesh)
    newtons = [ex.trial.n_iter for ex in run.executed]
    return {
        "status": "completed", "fail_reason": "",
        "channels": channels_from_u(st.u, mesh, geom),
        "n_plastic": int(n_pl),
        "r_front_over_a": float(r_front / mesh.a),
        "margin_over_a": float(margin / mesh.a),
        "newton_total": int(sum(newtons)),
        "newton_max": int(max(newtons)),
        "u_wall_normal_mean_m": met["u_wall_normal_mean"],
        "u_wall_normal_max_m": met["u_wall_normal_max"],
    }


def reduced_static_channels(op, s_star: float, K0: float,
                            nu: float) -> dict[str, float]:
    """Reduced-model static channels keyed by channel name (mm)."""
    import torch
    vals = op.static_channels_mm(
        torch.tensor(s_star), torch.tensor(K0), torch.tensor(nu))
    return {ch.channel: float(v) for ch, v in zip(op.channels, vals)}


def release_tail_factor(dt_end_days: float, lam0: float) -> float:
    """g(dt_end; lam0): model factor static -> observed tail."""
    return float(schedule.release_increment(
        np.asarray([dt_end_days]), lam0)[0])
