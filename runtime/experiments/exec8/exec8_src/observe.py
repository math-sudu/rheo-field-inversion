
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from . import schedule
from .geometry import observation_matrix, CHANNELS
from .data_io import CHORD_CHANNELS, CROWN_CHANNEL, Series
from .forward import DTYPE, ForwardOperator


SIGMA_MM = {
    ("DK0+935", "GD"): 3.484,
    ("DK0+935", "SL01-SL02"): 4.776,
    ("DK0+935", "SL03-SL04"): 6.220,
    ("DK0+935", "SL05-SL06"): 8.709,
    ("DK0+915", "GD"): 4.223,
    ("DK0+915", "SL01-SL02"): 6.669,
    ("DK0+915", "SL03-SL04"): 10.078,
    ("DK0+915", "SL05-SL06"): 3.498,
}


@dataclass(frozen=True)
class ObsChannel:
    section: str
    channel: str            # SL01-SL02 / SL03-SL04 / SL05-SL06 / GD
    dt_day: np.ndarray      # time since section t0 (>= 0)
    sigma_mm: float


def build_channels(series_by_section: dict[str, list[Series]],
                   sections: tuple[str, ...]) -> list[ObsChannel]:
    """Real-grid observation channels of the requested sections."""
    out = []
    for sec in sections:
        series = series_by_section[sec]
        t0 = min(float(s.t_day[0]) for s in series)
        for s in series:
            if s.channel not in CHORD_CHANNELS + (CROWN_CHANNEL,):
                continue
            out.append(ObsChannel(
                section=sec, channel=s.channel,
                dt_day=np.maximum(s.t_day - t0, 0.0),
                sigma_mm=SIGMA_MM[(sec, s.channel)]))
    return out


class ObservationOperator:
    """theta -> stacked model observations (mm) on the real grids.

    Static channel functionals (signs asserted positive at the box
    center by tests/test_forward_observe.py).  The exec6b displacement
    convention is GLOBALLY SIGN-FLIPPED excavation-induced motion
    ((u_x, u_y) = -(tension-positive displacement), general_map
    docstring), so in the RETURNED frame inward wall motion carries
    u_x(right) > 0 > u_x(left) and crown settlement carries
    u_y(crown) > 0; hence

    * chord j: c_j = +(u_x(R_j) - u_x(L_j))  [closing positive]
    * crown:   s   = +u_y(crown)             [settlement positive]

    Wall-point order in ForwardOperator: [L1 L2 L3 R1 R2 R3 crown]
    (geometry.build_geometry), u stacked (u_x(0..P-1), u_y(0..P-1)).
    """

    def __init__(self, fwd: ForwardOperator, channels: list[ObsChannel]):
        self.fwd = fwd
        self.channels = channels
        self.dt = [torch.as_tensor(ch.dt_day, dtype=DTYPE)
                   for ch in channels]
        self.sigma = torch.as_tensor([ch.sigma_mm for ch in channels],
                                     dtype=DTYPE)
        self.n_points = int(sum(len(ch.dt_day) for ch in channels))
        self._chord_idx = {"SL01-SL02": 0, "SL03-SL04": 1, "SL05-SL06": 2}

    def static_channels_mm(self, s_star, K0, nu) -> torch.Tensor:
        """(n_channels,) static (full-release) channel values, mm."""
        u = self.fwd.wall_displacements(s_star, K0, nu)  # (2P,) metres
        W = torch.as_tensor(observation_matrix(self.fwd.geom, returned_frame="km"),
                            dtype=DTYPE)
        vals = W @ u
        return torch.stack([vals[CHANNELS.index(ch.channel)]
                            for ch in self.channels]) * 1000.0

    def model(self, theta: dict[str, torch.Tensor]) -> torch.Tensor:
        """Stacked model observations (mm) for parameter dict theta.

        theta keys: s_star, K0, nu, and one lam0 per section, named
        ``lam0__<section>`` (a single shared key ``lam0`` is accepted
        when only one section is present).
        """
        stat = self.static_channels_mm(theta["s_star"], theta["K0"],
                                       theta["nu"])
        out = []
        for i, ch in enumerate(self.channels):
            key = (f"lam0__{ch.section}"
                   if f"lam0__{ch.section}" in theta else "lam0")
            g = schedule.release_increment(self.dt[i], theta[key])
            out.append(g * stat[i])
        return torch.cat(out)

    def sigma_vector(self) -> torch.Tensor:
        """(n_points,) per-point noise std (mm), channel-wise."""
        return torch.cat([
            torch.full((len(ch.dt_day),), ch.sigma_mm, dtype=DTYPE)
            for ch in self.channels])

    def synthesize(self, theta_true: dict[str, float], seed: int,
                   noisy: bool = True) -> np.ndarray:
        th = {k: torch.tensor(v, dtype=DTYPE)
              for k, v in theta_true.items()}
        clean = self.model(th).numpy()
        if not noisy:
            return clean
        rng = np.random.default_rng(seed)
        return clean + rng.normal(0.0, self.sigma_vector().numpy())

    # ------------------------------------------------------------------
    # analytic fast path (exact derivative of the same solve-path
    # composition; equality with the torch through-system AD asserted
    # in tests -- the model is bilinear in (kappa, K0) with the two
    # precomputed unit-sigma_v LS solutions x0 = x_hat(K0=0) and
    # xK = x_hat(1) - x_hat(0), so all channel functionals collapse to
    # four scalars per channel)
    # ------------------------------------------------------------------
    def _functionals(self):
        if hasattr(self, "_fun"):
            return self._fun
        with torch.no_grad():
            x0 = self.fwd.solve_xhat(torch.tensor(0.0, dtype=DTYPE))
            x1 = self.fwd.solve_xhat(torch.tensor(1.0, dtype=DTYPE))
            xK = x1 - x0
            A = self.fwd.A.numpy()
            B = self.fwd.B.numpy()
            x0 = x0.numpy()
            xK = xK.numpy()
        W = observation_matrix(self.fwd.geom, returned_frame="km")
        rows = []
        for ch in self.channels:
            f = W[CHANNELS.index(ch.channel)]
            rows.append((float(f @ (A @ x0)), float(f @ (A @ xK)),
                         float(f @ (B @ x0)), float(f @ (B @ xK))))
        self._fun = rows
        return rows

    def _static_np(self, s_star: float, K0: float, nu: float):
        """(n_channels,) statics + their (dK0, dnu) partials, mm."""
        kap = 3.0 - 4.0 * nu
        out, dK, dn = [], [], []
        for (a0, aK, b0, bK) in self._functionals():
            out.append(1000.0 * s_star * (kap * (a0 + K0 * aK)
                                          - (b0 + K0 * bK)))
            dK.append(1000.0 * s_star * (kap * aK - bK))
            dn.append(1000.0 * s_star * (-4.0) * (a0 + K0 * aK))
        return np.array(out), np.array(dK), np.array(dn)

    def model_np(self, theta: dict[str, float]) -> np.ndarray:
        stat, _, _ = self._static_np(theta["s_star"], theta["K0"],
                                     theta["nu"])
        parts = []
        for i, ch in enumerate(self.channels):
            key = (f"lam0__{ch.section}"
                   if f"lam0__{ch.section}" in theta else "lam0")
            g = schedule.release_increment(ch.dt_day, theta[key])
            parts.append(g * stat[i])
        return np.concatenate(parts)

    def model_and_jac_np(self, theta: dict[str, float],
                         names: list[str]):
        """(m, J) with J[:, a] = d m / d theta[names[a]] -- the exact
        closed-form derivative of the solve-path composition."""
        s, K0, nu = theta["s_star"], theta["K0"], theta["nu"]
        stat, dK, dn = self._static_np(s, K0, nu)
        m_parts, jac_parts = [], []
        for i, ch in enumerate(self.channels):
            key = (f"lam0__{ch.section}"
                   if f"lam0__{ch.section}" in theta else "lam0")
            g = schedule.release_increment(ch.dt_day, theta[key])
            gp = schedule.release_increment_dlam0(ch.dt_day, theta[key])
            m_i = g * stat[i]
            cols = []
            for n in names:
                if n == "s_star":
                    cols.append(m_i / s)
                elif n == "K0":
                    cols.append(g * dK[i])
                elif n == "nu":
                    cols.append(g * dn[i])
                elif n == key:
                    cols.append(gp * stat[i])
                else:            # lam0 of another section
                    cols.append(np.zeros_like(m_i))
            m_parts.append(m_i)
            jac_parts.append(np.stack(cols, axis=1))
        return np.concatenate(m_parts), np.vstack(jac_parts)
