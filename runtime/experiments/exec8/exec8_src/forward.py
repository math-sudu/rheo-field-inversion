
from __future__ import annotations

import pickle
import hashlib
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from . import geometry as geo

DTYPE = torch.float64

# gitignored run cache (root .gitignore line 69: experiments/**/runs/)
_CACHE_DIR = Path(__file__).resolve().parents[1] / "runs" / "op_cache"


@dataclass
class ForwardOperator:

    geom: "geo.FrozenGeometry"
    G_scaled: torch.Tensor      # (2M, 4N) column-scaled system
    col_scale: torch.Tensor     # (4N,) column scales
    Rq: torch.Tensor            # QR factors of G_scaled
    Qt_h0: torch.Tensor         # Q^T h0
    Qt_hK: torch.Tensor         # Q^T hK
    A: torch.Tensor             # (2P, 4N) kappa-part displacement matrix
    B: torch.Tensor             # (2P, 4N) remainder matrix
    n_order: int

    # ------------------------------------------------------------------
    @classmethod
    def build_cached(cls, WH=None, *, geom=None, **geometry_options) -> "ForwardOperator":
        """Cache by geometry, observations, discretization and source code.

        Old width/height caches are never opened. The algebra twin lock does
        not establish engineering applicability; provenance is checked first.
        """
        if geom is None:
            geom = geo.build_geometry(WH, **geometry_options)
        elif WH is not None or geometry_options:
            raise ValueError("Pass geom or geometry options, not both")
        code_hash = hashlib.sha256(b"".join(Path(p).read_bytes() for p in (
            __file__, geo.__file__, geo.general_map.__file__,
            Path(__file__).with_name("observe.py")))).hexdigest()
        key = f"{geom.fingerprint}_{code_hash}_N{geo.N_ORDER}_M{geo.N_COLLOCATION}.pkl"
        path = _CACHE_DIR / "cad_v1" / key
        if path.exists():
            with open(path, "rb") as f:
                op = pickle.load(f)
            if op.geom.fingerprint != geom.fingerprint:
                raise ValueError("Cached geometry/observation identity mismatch")
        else:
            op = cls.build(geom)
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as f:
                pending = Path(f.name)
                pickle.dump(op, f)
            os.replace(pending, path)
        lock = op.twin_lock(n_trials=1)
        if lock > 1e-10:
            raise AssertionError(
                f"cached operator failed the twin lock: {lock:.3e}")
        return op

    @classmethod
    def build(cls, geom: "geo.FrozenGeometry") -> "ForwardOperator":
        N = geo.N_ORDER
        M = geo.N_COLLOCATION
        theta_wall = 2.0 * np.pi * np.arange(M) / M

        # system matrix + RHS split through the exec6b public API
        rep0 = geo.make_rep(geom, sigma_v=1.0, K0=0.0)
        G_mat, h0 = rep0.boundary_system(theta_wall)
        rep1 = geo.make_rep(geom, sigma_v=1.0, K0=1.0)
        G_mat1, h01 = rep1.boundary_system(theta_wall)
        if not np.allclose(G_mat, G_mat1, rtol=0, atol=1e-13):
            raise AssertionError("system matrix depends on (sigma_v, K0); "
                                 "the RHS split assumption is broken")
        hK = h01 - h0
        # verify RHS K0-linearity against a direct assembly
        kv, sv = 0.7321, 3.117
        rep_t = geo.make_rep(geom, sigma_v=sv, K0=kv)
        _, h_t = rep_t.boundary_system(theta_wall)
        dev = np.max(np.abs(h_t - sv * (h0 + kv * hK)))
        if dev > 1e-10 * max(1.0, np.max(np.abs(h_t))):
            raise AssertionError(f"RHS K0-linearity check failed: {dev}")

        # displacement matrices by unit-coefficient probing (2G = 1)
        P = len(geom.theta_points)
        rho = np.ones(P)
        th = geom.theta_points
        A = np.zeros((2 * P, 4 * N))
        B = np.zeros((2 * P, 4 * N))
        for k in range(4 * N):
            xa = np.zeros(2 * N)
            xb = np.zeros(2 * N)
            if k < 2 * N:
                xa[k] = 1.0
            else:
                xb[k - 2 * N] = 1.0
            ca = xa[:N] + 1j * xa[N:]
            cb = xb[:N] + 1j * xb[N:]
            # nu = 0 -> kappa = 3; nu = 0.5 -> kappa = 1; E = 1 + nu
            ux3, uy3 = rep0.displacements_cartesian(rho, th, ca, cb,
                                                    E=1.0, nu=0.0)
            ux1, uy1 = rep0.displacements_cartesian(rho, th, ca, cb,
                                                    E=1.5, nu=0.5)
            u3 = np.concatenate([ux3, uy3])
            u1 = np.concatenate([ux1, uy1])
            A[:, k] = 0.5 * (u3 - u1)
            B[:, k] = 3.0 * A[:, k] - u3    # u3 = 3A - B -> B = 3A - u3
        # column scaling for QR conditioning (solution invariant)
        Gt = torch.as_tensor(G_mat, dtype=DTYPE)
        col = torch.linalg.vector_norm(Gt, dim=0)
        col = torch.where(col > 0, col, torch.ones_like(col))
        Gs = Gt / col
        Q, Rq = torch.linalg.qr(Gs, mode="reduced")
        rank = int(torch.linalg.matrix_rank(Rq))
        if rank != 4 * N:
            raise AssertionError(
                f"collocation system rank {rank} != {4*N}: full-rank "
                "assumption of the QR least-squares route is broken")
        return cls(geom=geom, G_scaled=Gs, col_scale=col, Rq=Rq,
                   Qt_h0=Q.T @ torch.as_tensor(h0, dtype=DTYPE),
                   Qt_hK=Q.T @ torch.as_tensor(hK, dtype=DTYPE),
                   A=torch.as_tensor(A, dtype=DTYPE),
                   B=torch.as_tensor(B, dtype=DTYPE),
                   n_order=N)

    # ------------------------------------------------------------------
    def solve_xhat(self, K0: torch.Tensor) -> torch.Tensor:
        """Unit-sigma_v LS coefficient vector x_hat(K0), inside autograd."""
        rhs = self.Qt_h0 + K0 * self.Qt_hK
        y = torch.linalg.solve_triangular(self.Rq, rhs.unsqueeze(-1),
                                          upper=True).squeeze(-1)
        return y / self.col_scale

    def wall_displacements(self, s_star: torch.Tensor, K0: torch.Tensor,
                           nu: torch.Tensor) -> torch.Tensor:
        xh = self.solve_xhat(K0)
        kappa = 3.0 - 4.0 * nu
        return s_star * ((kappa * self.A - self.B) @ xh)

    # numpy reference (twin-lock partner), exec6b path verbatim
    def wall_displacements_numpy(self, sigma_v: float, K0: float,
                                 E: float, nu: float) -> np.ndarray:
        rep = geo.make_rep(self.geom, sigma_v=sigma_v, K0=K0)
        sol = geo.general_map.solve_ls(rep, n_collocation=geo.N_COLLOCATION)
        P = len(self.geom.theta_points)
        ux, uy = rep.displacements_cartesian(
            np.ones(P), self.geom.theta_points,
            sol["coef_a"], sol["coef_b"], E, nu)
        return np.concatenate([ux, uy])

    def twin_lock(self, n_trials: int = 3, seed: int = 20260712) -> float:
        """Max relative deviation torch vs exec6b numpy path."""
        rng = np.random.default_rng(seed)
        worst = 0.0
        for _ in range(n_trials):
            sv = float(rng.uniform(9.5, 25.94))
            K0 = float(rng.uniform(0.69, 1.18))
            nu = float(rng.uniform(0.16, 0.35))
            E = float(rng.uniform(560.15, 25250.0))
            G2 = E / (1.0 + nu)
            s_star = sv / G2
            u_np = self.wall_displacements_numpy(sv, K0, E, nu)
            u_t = self.wall_displacements(
                torch.tensor(s_star, dtype=DTYPE),
                torch.tensor(K0, dtype=DTYPE),
                torch.tensor(nu, dtype=DTYPE)).numpy()
            dev = np.max(np.abs(u_t - u_np)) / np.max(np.abs(u_np))
            worst = max(worst, float(dev))
        return worst
