"""Event-sequence step-waveform probe (EXEC-19 batch 5, pre-run
note 6; SUPPLEMENTARY position -- readings only, no claim upgrade,
official batch-3 rows untouched).

Two-wave release family (note 6 item 1; law SHAPE frozen -- the
exec8 constants beta/T_REL/LAM_F are shared by both waves)::

    lam_2w(t; w, t_e) = w * Lr(t) + (1 - w) * Lr(t - t_e)

with Lr the single-wave law INCLUDING the face-passage ramp
(0 -> LAM_F over RAMP_D, then lam_abs(t - RAMP_D)); physical image:
upper-bench passage at t = 0 releases share w, lower-bench passage at
t_e releases the remainder along the same law shape -- the (1-w)*LAM_F
rise across [t_e, t_e + RAMP_D] IS the step morphology carrier.

Degeneracy anchors (V-E1): w = 1 and t_e = 0 both reduce to the
single-wave law exactly (array identity); both waves are monotone so
the mixture is monotone (guarded).

Probe frame (note 6 item 2): t_off (face-passage -> monitoring-start,
days) is a DIRECT coordinate here -- the lam0 quotient's inverse-law
image is single-wave-specific; the material lane is the batch-3
tier-C official theta (module separation, D-055).  No AD channel for
(w, t_e): replay19 treats the schedule as constants, so the fit lane
is gradient-free (grid + Nelder-Mead), declared in note 6 item 5.
"""

from __future__ import annotations

import numpy as np

from . import _paths  # noqa: F401
from .field19 import CHANNELS, FieldStamps, build_field_geometry, \
    channel_matrix, lam_inv
from .scenario import RAMP_D, RheoScenario, lam_abs

from exec8_src import schedule as x8sched                  # noqa: E402
from exec8_src.observe import SIGMA_MM                     # noqa: E402
from exec9_src import coupled as x9coupled                 # noqa: E402
from fedev_src import coupling as fcoupling                # noqa: E402
from fedev_src import noncirc as fnoncirc                  # noqa: E402


def lam_ramped(t):
    """Single-wave release value with the face-passage ramp: 0 for
    t <= 0, linear to LAM_F over RAMP_D, lam_abs(t - RAMP_D) after
    (continuous at the ramp knot: lam_abs(0) = LAM_F)."""
    t = np.asarray(t, dtype=float)
    out = np.where(
        t <= 0.0, 0.0,
        np.where(t < RAMP_D, x8sched.LAM_F * t / RAMP_D,
                 lam_abs(t - RAMP_D)))
    return out


def lam_two_wave(t, w, t_e):
    """Two-wave mixture lam_2w(t; w, t_e) (note 6 item 1)."""
    return w * lam_ramped(t) + (1.0 - w) * lam_ramped(
        np.asarray(t, dtype=float) - t_e)


def event_time_axis(span_days, t_e, t_off_hi=None, n_release=14,
                    n_creep=12):
    """Fixed knot-time axis covering BOTH waves' release transients
    plus the observation span (mirror of field19.field_knot_axis with
    the wave-2 knots unioned in; knots include t_e and t_e + RAMP_D
    so the step rise is committed exactly)."""
    if t_off_hi is None:
        t_off_hi = lam_inv(0.85)
    t_end = float(t_off_hi + span_days + 1.0)
    t_rel = x8sched.T_REL
    rel = RAMP_D + (4.0 * t_rel - RAMP_D) * (
        np.arange(1, n_release + 1) / n_release)
    wave1 = np.concatenate([[0.0, RAMP_D], rel])
    wave2 = t_e + np.concatenate([[0.0, RAMP_D], rel])
    t_creep = np.geomspace(4.0 * t_rel, t_end, n_creep + 1)[1:]
    knots = np.unique(np.concatenate([wave1, wave2, t_creep]))
    knots = knots[knots <= t_end + 1e-12]
    if knots[-1] < t_end - 1e-9:
        knots = np.append(knots, t_end)
    return knots


class EventContext:
    """One (section, geometry, mesh, W, sigma, K_ref) context for the
    probe; the knot axis is (t_e)-dependent and rebuilt per call
    (gradient-free lane -- no replay contract on the axis)."""

    def __init__(self, section="DK0+935", n_r=8, n_t=32,
                 n_release=14, n_creep=12):
        self.geom = build_field_geometry()
        self.stamps = FieldStamps.load(section)
        self.mesh = x9coupled.build_mesh(self.geom, n_r=n_r, n_t=n_t,
                                         r_level=x9coupled.R_LEVEL)
        self.W = channel_matrix(self.mesh, self.geom)
        self.sig = {ch: float(SIGMA_MM[(section, ch)])
                    for ch in CHANNELS}
        self.n_release, self.n_creep = n_release, n_creep
        scn_ref = RheoScenario(s_star=3.0e-2, K0=1.0, r_K=0.35,
                               tau_K=2.0, tau_M=200.0)
        sys_ref = fnoncirc.NonCircFESystem(
            self.mesh, scn_ref.base(a=self.geom.R), scn_ref.material())
        self._E_ref = float(sys_ref.base.E)
        self._K_ref = fcoupling.condensed_boundary_block(
            self.mesh, sys_ref.base)
        self.span = self.stamps.span()

    def forward_obs(self, scn: RheoScenario, w, t_e, t_off):
        """Viscous forward on the two-wave axis + own-clock model at
        the direct t_off; returns (model_flat, run)."""
        times = event_time_axis(self.span, t_e, t_off_hi=t_off,
                                n_release=self.n_release,
                                n_creep=self.n_creep)
        lams = lam_two_wave(times, w, t_e)
        if not np.all(np.diff(lams) >= -1e-12):
            raise AssertionError("two-wave path must be monotone")
        lams = np.maximum.accumulate(lams)
        system = fnoncirc.NonCircFESystem(
            self.mesh, scn.base(a=self.geom.R), scn.material())
        K_bnd = (float(scn.E) / self._E_ref) * self._K_ref
        run, _ = fcoupling.run_condensed(
            system, lams, K_bnd=K_bnd, times=times)
        t_commit = np.array([ex.trial.t for ex in run.executed])
        U = np.stack([self.W @ np.asarray(ex.trial.u)
                      for ex in run.executed])
        out = []
        for r, ch in enumerate(CHANNELS):
            st = self.stamps.stamps[ch]
            tq = t_off + (st - self.stamps.stamp0)
            v = np.interp(tq, t_commit, U[:, r])
            v0 = np.interp(t_off + float(st[0] - self.stamps.stamp0),
                           t_commit, U[:, r])
            out.append(v - v0)
        return np.concatenate(out), run

    def y_flat(self):
        return np.concatenate([self.stamps.y_mm[ch] for ch in CHANNELS])

    def sigma_flat(self):
        return np.concatenate([
            np.full(len(self.stamps.stamps[ch]), self.sig[ch])
            for ch in CHANNELS])

    def split(self, flat):
        out, k = {}, 0
        for ch in CHANNELS:
            n = len(self.stamps.stamps[ch])
            out[ch] = flat[k:k + n]
            k += n
        return out

    def lam0_eff_2w(self, w, t_e, t_off):
        return lam0_eff_two_wave(self.stamps, w, t_e, t_off)


def lam0_eff_two_wave(stamps: FieldStamps, w, t_e, t_off):
    """Generalized per-channel derived phase reading (note 6 item 2):
    lam_2w at each channel's first own-clock stamp."""
    out = {}
    for ch in CHANNELS:
        dt0 = float(stamps.stamps[ch][0] - stamps.stamp0)
        out[ch] = float(lam_two_wave(t_off + dt0, w, t_e))
    return out


# ------------------------------------------------- data-side readings

def jump_table(stamps: FieldStamps, channel="SL05-SL06", k=5):
    """Top-k own-clock jump times/magnitudes of a measured series
    (note 6 item 4; pinned data direct read)."""
    t = np.asarray(stamps.stamps[channel], dtype=float)
    y = np.asarray(stamps.y_mm[channel], dtype=float)
    dy = np.diff(y)
    t_mid = 0.5 * (t[1:] + t[:-1]) - t[0]
    order = np.argsort(-np.abs(dy))[:k]
    order = order[np.argsort(t_mid[order])]
    return t_mid[order], dy[order]


def step_alignment(stamps: FieldStamps, t_e, channel="SL05-SL06",
                   k=5, halfwin=1.0):
    """(min_k |t_jump - t_e|, single-event mass share): share of the
    channel's total increment carried inside t_e +/- halfwin (own
    clock), against the top-k jump times (note 6 item 4)."""
    tj, _ = jump_table(stamps, channel=channel, k=k)
    t = np.asarray(stamps.stamps[channel], dtype=float)
    y = np.asarray(stamps.y_mm[channel], dtype=float)
    dy = np.diff(y)
    t_mid = 0.5 * (t[1:] + t[:-1]) - t[0]
    total = float(y[-1] - y[0])
    inwin = float(dy[np.abs(t_mid - t_e) <= halfwin].sum())
    share = inwin / total if total != 0.0 else float("nan")
    return float(np.min(np.abs(tj - t_e))), share


def shape_dev(y, m):
    """Normalized shape deviation max |y/y_end - m/m_end| (EXEC-9
    section 2-2 inherited metric)."""
    y = np.asarray(y, float)
    m = np.asarray(m, float)
    if abs(y[-1]) < 1e-12 or abs(m[-1]) < 1e-12:
        return float("nan")
    return float(np.max(np.abs(y / y[-1] - m / m[-1])))


def sign_run(r):
    """Fraction of consecutive same-sign residual pairs (self-set;
    white noise ~0.5, fully systematic -> 1)."""
    s = np.sign(np.asarray(r, float))
    s = s[s != 0]
    if len(s) < 2:
        return float("nan")
    return float(np.mean(s[1:] == s[:-1]))
