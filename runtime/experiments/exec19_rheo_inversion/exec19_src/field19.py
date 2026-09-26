"""Field tier (G-F): 935 mapped section, viscous FE forward, channel
observation operator with own-clock increments (family-B lineage).

Geometry/pipeline = exec9_src.coupled read-only (LaurentMap +
MappedAnnulusMesh on the frozen exec8 FrozenGeometry; channel
conventions of channels_from_u).  The channel extraction is
precompiled into a sparse linear operator W (channels = W @ u): the
periodic cubic-spline interpolation is LINEAR in the nodal values, so
W is exact -- locked element-wise against channels_from_u.

Observation model (PROTOCOL.md section 1 + correction note 1):

    obs_i(stamp) = U_i(t_off + (stamp - stamp0_sec))
                   - U_i(t_off + (stamp_i0 - stamp0_sec))

with U_i the per-committed-increment channel series of ONE viscous
run on a FIXED dense knot axis (theta-independent schedule), linearly
interpolated in time; t_off = Lambda^{-1}(lam0) the section's
face-passage -> monitoring-start offset carried by the lam0 quotient
coordinate (exec8 lam0 box semantics).  Per-channel phases are the
data-known stamp offsets (own clocks); the EXEC-9b per-channel
lam0^ch becomes the DERIVED reading lam_abs(t at channel install).
"""

from __future__ import annotations

import dataclasses

import numpy as np
import torch

from . import _paths  # noqa: F401
from .scenario import RheoScenario, lam_abs
from . import scenario as scn_mod

from exec8_src import data_io, geometry as x8geo           # noqa: E402
from exec8_src import schedule as x8sched                  # noqa: E402
from exec8_src.observe import SIGMA_MM                     # noqa: E402
from exec9_src import coupled as x9coupled                 # noqa: E402
from fedev_src import coupling as fcoupling                # noqa: E402
from fedev_src import noncirc as fnoncirc                  # noqa: E402
from fedev_src.meshing import DTYPE                        # noqa: E402

CHANNELS = ("GD", "SL01-SL02", "SL03-SL04", "SL05-SL06")


def lam_inv(lam0):
    """t since face passage at which lam_abs(t) = lam0 (frozen law)."""
    return x8sched.T_REL * (np.log((1.0 - x8sched.LAM_F)
                                   / (1.0 - lam0))
                            ) ** (1.0 / x8sched.BETA)


def build_field_geometry(wh=None, **geometry_options):
    geom = x8geo.build_geometry(wh, **geometry_options)
    x9coupled.assert_engineering_upright(geom)
    return geom


def build_field_system(scn: RheoScenario, geom, n_r=8, n_t=32,
                       r_level=x9coupled.R_LEVEL):
    mesh = x9coupled.build_mesh(geom, n_r=n_r, n_t=n_t, r_level=r_level)
    base = scn.base(a=geom.R)     # BaseParams.a is bookkeeping here;
    # the load/mesh geometry is fully carried by the mapped mesh
    return fnoncirc.NonCircFESystem(mesh, base, scn.material()), mesh


def channel_matrix(mesh, geom):
    """Exact (4, n_dof) channel operator; rows ordered CHANNELS."""
    idx = mesh.ring_dofs("inner")
    n_dof = mesh.n_dof
    W = np.zeros((len(CHANNELS), n_dof))
    base = np.zeros(n_dof)
    for dof in idx:
        base[:] = 0.0
        base[dof] = 1.0
        ch = x9coupled.channels_from_u(base, mesh, geom)
        for r, name in enumerate(CHANNELS):
            W[r, dof] = ch[name]
    # lock: random vector against the reference implementation
    rng = np.random.default_rng(0)
    u = rng.standard_normal(n_dof)
    ref = x9coupled.channels_from_u(u, mesh, geom)
    dev = max(abs(float(W[r] @ u) - ref[name])
              for r, name in enumerate(CHANNELS))
    if dev > 1e-9 * max(1.0, max(abs(v) for v in ref.values())):
        raise AssertionError(f"channel operator lock failed: {dev}")
    return W


@dataclasses.dataclass(frozen=True)
class FieldStamps:
    """Per-channel observation stamps of one section (days, absolute
    day-of-year floats as digitized; own clock = first stamp)."""
    section: str
    stamps: dict          # {channel: np.ndarray}
    y_mm: dict            # {channel: np.ndarray} measured series
    stamp0: float         # section clock zero = min over channels

    @classmethod
    def load(cls, section):
        data = data_io.load_all()
        stamps, y = {}, {}
        for s in data[section]:
            stamps[s.channel] = np.asarray(s.t_day, dtype=float)
            y[s.channel] = np.asarray(s.y_mm, dtype=float)
        stamp0 = min(float(v[0]) for v in stamps.values())
        return cls(section=section, stamps=stamps, y_mm=y,
                   stamp0=stamp0)

    def span(self):
        return max(float(v[-1]) for v in self.stamps.values()) \
            - self.stamp0


def field_knot_axis(span_days, lam0_box=(0.30, 0.90), n_release=14,
                    n_creep=12):
    """Fixed dense (lambdas, times) axis covering every candidate
    t_off in the lam0 box plus the observation span."""
    t_off_max = lam_inv(lam0_box[1])
    t_end = t_off_max + span_days + 1.0
    t_rel = x8sched.T_REL
    ramp = scn_mod.RAMP_D
    t_release = ramp + (4.0 * t_rel - ramp) * (
        np.arange(1, n_release + 1) / n_release)
    t_creep = np.geomspace(4.0 * t_rel, t_end, n_creep + 1)[1:]
    knots = np.unique(np.concatenate([[0.0, ramp], t_release, t_creep]))
    if knots[-1] < t_end - 1e-9:
        knots = np.append(knots, t_end)
    times = knots
    lams = np.concatenate([[0.0], lam_abs(knots[1:] - ramp)])
    lams[np.searchsorted(times, ramp)] = x8sched.LAM_F
    lams = np.maximum.accumulate(lams)
    return lams, times


def run_field_forward(system, lams, times, K_bnd=None):
    return fcoupling.run_condensed(system, np.asarray(lams, float),
                                   K_bnd=K_bnd,
                                   times=np.asarray(times, float))


# ------------------------------------------------- torch observation

def torch_interp(tq, tk, vk):
    """Piecewise-linear interpolation, differentiable in tq AND vk.

    tk: (n,) ascending float tensor (constants); vk: (n, ...) values;
    tq: (m,) query tensor.  Clamped at the ends (readings outside the
    committed span raise upstream)."""
    j = torch.searchsorted(tk, tq.detach(), right=True)
    j = torch.clamp(j, 1, len(tk) - 1)
    t0, t1 = tk[j - 1], tk[j]
    w = (tq - t0) / (t1 - t0)
    w = torch.clamp(w, 0.0, 1.0)
    v0, v1 = vk[j - 1], vk[j]
    return v0 + w.unsqueeze(-1) * (v1 - v0)


class FieldObservation:
    """Own-clock incremental observation of one section (torch)."""

    def __init__(self, stamps: FieldStamps, W, sigma_section=None):
        self.stamps = stamps
        self.W_t = torch.tensor(W, dtype=DTYPE)
        self.sig = {ch: float(SIGMA_MM[(stamps.section, ch)])
                    for ch in CHANNELS} if sigma_section is None \
            else dict(sigma_section)

    def channel_series(self, u_list):
        """(n_commit, 4) differentiable channel series from replay u."""
        U = torch.stack([self.W_t @ u for u in u_list])
        return U

    def model(self, t_commit, U, lam0_t):
        """Per-channel own-clock increments at the data stamps.

        t_commit: (n_commit,) float tensor of committed times;
        U: (n_commit, 4) channel series; lam0_t: scalar tensor.
        Returns dict {channel: tensor (m_ch,)} in mm.
        """
        t_off = x8sched.T_REL * (torch.log(
            (1.0 - x8sched.LAM_F) / (1.0 - lam0_t))
        ) ** (1.0 / x8sched.BETA)
        out = {}
        for r, ch in enumerate(CHANNELS):
            st = self.stamps.stamps[ch]
            tq = t_off + torch.tensor(st - self.stamps.stamp0,
                                      dtype=DTYPE)
            t0 = t_off + float(st[0] - self.stamps.stamp0)
            vals = torch_interp(tq, t_commit, U[:, r:r + 1])[:, 0]
            v0 = torch_interp(t0.reshape(1), t_commit,
                              U[:, r:r + 1])[0, 0]
            out[ch] = vals - v0
        return out

    def residual_vector(self, model_out, y=None):
        """Noise-weighted residual stack (WLS, sigma tables)."""
        rs = []
        for ch in CHANNELS:
            m = model_out[ch]
            yy = torch.tensor((self.stamps.y_mm[ch] if y is None
                               else y[ch]), dtype=DTYPE)
            rs.append((m - yy) / self.sig[ch])
        return torch.cat(rs)

    def flat(self, model_out):
        return torch.cat([model_out[ch] for ch in CHANNELS])

    def n_obs(self):
        return sum(len(self.stamps.stamps[ch]) for ch in CHANNELS)
