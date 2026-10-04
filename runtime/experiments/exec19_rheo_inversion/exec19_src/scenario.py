
from __future__ import annotations

import dataclasses

import numpy as np

from . import _paths  # noqa: F401

from exec8_src import schedule as x8sched                  # noqa: E402
from fedev_src import params as fparams                    # noqa: E402

GPAH_TO_MPAD = 1000.0 / 24.0

SIGMA_V_MPA = 12.96
NU_FIXED = 0.255         # exec9 continuity
A_M = 5.0                # G-C circle radius (exec7b demo constant)

PHI = float(fparams.SLATE_CONV_PHI)
PSI = float(fparams.SLATE_CONV_PSI)


@dataclasses.dataclass(frozen=True)
class RheoScenario:
    """Quotient-coordinate scenario (all floats; days/MPa units)."""
    s_star: float
    K0: float
    r_K: float
    tau_K: float
    tau_M: float
    r_s: float | None = None      # None => viscoplastic leg disabled
    tau_vp: float | None = None   # (huge threshold keeps MC inactive)
    sigma_v: float = SIGMA_V_MPA
    nu: float = NU_FIXED

    # ---------------------------------------------------- conversions
    @property
    def G0(self):
        return self.sigma_v / (2.0 * self.s_star)

    @property
    def E(self):
        return 2.0 * (1.0 + self.nu) * self.G0

    @property
    def G_K(self):
        return self.r_K * self.G0

    @property
    def E_K(self):
        return 2.0 * (1.0 + self.nu) * self.G_K

    @property
    def eta_K(self):
        return self.tau_K * self.G_K

    @property
    def eta_M(self):
        return self.tau_M * 2.0 * self.G0

    def material(self):
        n_phi = (1.0 + np.sin(PHI)) / (1.0 - np.sin(PHI))
        if self.r_s is None:
            c = 1.0e9
            eta_vp = 1.0e12
        else:
            sigma_s_1d = self.r_s * self.sigma_v
            c = sigma_s_1d / (2.0 * np.sqrt(n_phi))
            eta_vp_1d = (self.tau_vp if self.tau_vp is not None
                         else 30.0) * 2.0 * self.G0
            eta_vp = eta_vp_1d * 0.5 * (1.0 - np.sin(PHI)) \
                * (1.0 - np.sin(PSI))
        return fparams.NishiharaMC(
            c=float(c), phi=PHI, psi=PSI, E_K=float(self.E_K),
            eta_K=float(self.eta_K), eta_M=float(self.eta_M),
            eta_vp=float(eta_vp))

    def base(self, a=A_M):
        return fparams.BaseParams(a=a, sigma_v=self.sigma_v, K0=self.K0,
                                  E=self.E, nu=self.nu)

    def names_values(self, names):
        return np.array([getattr(self, n) for n in names], dtype=float)

    def replace(self, **kw):
        return dataclasses.replace(self, **kw)


def from_slate_group(group, s_star, K0, scenario_overrides=None):
    ov = dict(scenario_overrides or {})
    mat = group.nishihara(NU_FIXED, **ov)
    G0 = SIGMA_V_MPA / (2.0 * s_star)
    G_K = mat.E_K / (2.0 * (1.0 + NU_FIXED))
    scn = RheoScenario(
        s_star=s_star, K0=K0,
        r_K=float(G_K / G0),
        tau_K=float(mat.eta_K / G_K),
        tau_M=float(mat.eta_M / (2.0 * G0)),
        r_s=None, tau_vp=None)
    return scn, mat


# ------------------------------------------------------------ schedule

def lam_abs(t_day):
    t = np.asarray(t_day, dtype=float)
    return 1.0 - (1.0 - x8sched.LAM_F) * np.exp(
        -(np.maximum(t, 0.0) / x8sched.T_REL) ** x8sched.BETA)


def check_lam_abs_against_release_increment(lam0=0.55, dt=np.array(
        [0.0, 1.0, 5.0, 20.0, 60.0])):
    """Lock: g(dt; lam0) == lam_abs(t0+dt) - lam0 with t0 = lam^-1(lam0)."""
    t0 = x8sched.T_REL * (np.log((1 - x8sched.LAM_F)
                                 / (1 - lam0))) ** (1 / x8sched.BETA)
    lhs = x8sched.release_increment(dt, lam0)
    rhs = lam_abs(t0 + dt) - lam0
    return float(np.max(np.abs(lhs - rhs)))


RAMP_D = 0.05      # face-passage ramp 0 -> LAM_F over 0.05 d (declared
#                    knot-design constant; creep over 0.05 d is
#                    negligible at the scenario tau scales)


def knot_path(t_obs_days, t_end=None, n_release=10, n_creep=8):
    """(lambdas, times) knot arrays for the viscous run.

    Path: t=0 face passage; ramp 0 -> LAM_F over RAMP_D; release law
    lam_abs on a dense grid to 4*T_REL; then lambda ~ holds (the law's
    residual tail is carried exactly -- knots keep lam_abs(t), which
    is monotone), creep-dominated log-spaced knots to t_end; the
    observation stamps are UNION-ed into the knot axis so committed
    increments land exactly on them.
    """
    t_obs = np.asarray(t_obs_days, dtype=float)
    if t_end is None:
        t_end = float(t_obs.max())
    t_rel = x8sched.T_REL
    t_release = RAMP_D + (4.0 * t_rel - RAMP_D) * (
        np.arange(1, n_release + 1) / n_release)
    t_creep = np.geomspace(4.0 * t_rel, max(t_end, 4.0 * t_rel + 1.0),
                           n_creep + 1)[1:]
    knots = np.unique(np.concatenate(
        [[0.0, RAMP_D], t_release, t_creep,
         t_obs[(t_obs > RAMP_D)]]))
    knots = knots[knots <= max(t_end, knots[0]) + 1e-12]
    if knots[-1] < t_end - 1e-12:
        knots = np.append(knots, t_end)
    times = np.concatenate([[0.0], knots[1:]])
    lams = np.concatenate([[0.0], lam_abs(knots[1:] - RAMP_D)])
    # ramp knot: lambda(RAMP_D) = LAM_F exactly
    lams[np.searchsorted(times, RAMP_D)] = x8sched.LAM_F
    if not np.all(np.diff(lams) >= -1e-12):
        raise AssertionError("release path must be monotone")
    lams = np.maximum.accumulate(lams)
    return lams, times
