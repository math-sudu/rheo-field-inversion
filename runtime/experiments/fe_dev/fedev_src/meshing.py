"""Structured Q4 annulus mesh, mirroring the prototype's discretization.

Identical conventions to ``coupling_schwarz_proto/proto_src/fe_annulus.Mesh``:
node id = i_r * n_t + j_theta, geometric radial grading, uniform theta,
element connectivity [i*n_t+j, (i+1)*n_t+j, (i+1)*n_t+jp, i*n_t+jp],
2x2 Gauss stiffness quadrature, 3-point Gauss consistent boundary loads.

Adds precomputed B-matrices / Jacobian weights at all 2x2 Gauss points in
both numpy and torch form (quadrature-point-flat layout, element-major:
qp index q = e * 4 + gp with gp enumerating the prototype's
``for xi in G2: for eta in G2`` loop order) plus edge-Gauss geometry for
differentiable ring loads.
"""

import numpy as np
import torch

DTYPE = torch.float64

G2 = (-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0))
G3_P = (-np.sqrt(0.6), 0.0, np.sqrt(0.6))
G3_W = (5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0)


def shape_derivs(xi, eta):
    return 0.25 * np.array([
        [-(1.0 - eta), -(1.0 - xi)],
        [(1.0 - eta), -(1.0 + xi)],
        [(1.0 + eta), (1.0 + xi)],
        [-(1.0 + eta), (1.0 - xi)],
    ])


class Mesh:
    """Structured annulus mesh; node id = i_r * n_t + j_theta."""

    def __init__(self, a, R, n_r, n_t):
        self.a, self.R, self.n_r, self.n_t = float(a), float(R), int(n_r), int(n_t)
        ratio = (self.R / self.a) ** (1.0 / self.n_r)
        self.radii = self.a * ratio ** np.arange(self.n_r + 1)
        self.radii[-1] = self.R
        self.theta_nodes = 2.0 * np.pi * np.arange(self.n_t) / self.n_t
        rr = np.repeat(self.radii, self.n_t)
        tt = np.tile(self.theta_nodes, self.n_r + 1)
        self.r_node, self.th_node = rr, tt
        self.xy = np.column_stack([rr * np.cos(tt), rr * np.sin(tt)])
        self.n_nodes = (self.n_r + 1) * self.n_t
        self.n_dof = 2 * self.n_nodes
        i = np.repeat(np.arange(self.n_r), self.n_t)
        j = np.tile(np.arange(self.n_t), self.n_r)
        jp = (j + 1) % self.n_t
        self.elems = np.column_stack([
            i * self.n_t + j,
            (i + 1) * self.n_t + j,
            (i + 1) * self.n_t + jp,
            i * self.n_t + jp,
        ])
        self.inner_nodes = np.arange(self.n_t)
        self.outer_nodes = self.n_r * self.n_t + np.arange(self.n_t)
        self.n_elem = self.elems.shape[0]

        # element dof map (Cartesian interleaved, node-major)
        self.edof = np.empty((self.n_elem, 8), dtype=np.int64)
        self.edof[:, 0::2] = 2 * self.elems
        self.edof[:, 1::2] = 2 * self.elems + 1

        self._precompute_quadrature()
        self._precompute_torch()

    # ------------------------------------------------------------ rings
    def ring_nodes(self, ring):
        return self.inner_nodes if ring == "inner" else self.outer_nodes

    def ring_dofs(self, ring):
        n = self.ring_nodes(ring)
        return np.column_stack([2 * n, 2 * n + 1]).ravel()

    # ------------------------------------------------------ quadrature
    def _bmat_at(self, xi, eta):
        """B matrices (ne,3,8) and det(J) (ne,) at one parent point."""
        dN = shape_derivs(xi, eta)                      # (4,2)
        xy = self.xy[self.elems]                        # (ne,4,2)
        J = np.einsum("ak,eai->eki", dN, xy)            # (ne,2,2)
        detJ = J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0]
        Jinv = np.empty_like(J)
        Jinv[:, 0, 0] = J[:, 1, 1]
        Jinv[:, 0, 1] = -J[:, 0, 1]
        Jinv[:, 1, 0] = -J[:, 1, 0]
        Jinv[:, 1, 1] = J[:, 0, 0]
        Jinv /= detJ[:, None, None]
        dNxy = np.einsum("eik,ak->eai", Jinv, dN)       # (ne,4,2)
        ne = self.n_elem
        B = np.zeros((ne, 3, 8))
        B[:, 0, 0::2] = dNxy[:, :, 0]
        B[:, 1, 1::2] = dNxy[:, :, 1]
        B[:, 2, 0::2] = dNxy[:, :, 1]
        B[:, 2, 1::2] = dNxy[:, :, 0]
        return B, detJ

    def _precompute_quadrature(self):
        """2x2 Gauss data in the prototype's loop order (weights = 1)."""
        Bs, dets, coords = [], [], []
        xy = self.xy[self.elems]
        for xi in G2:
            for eta in G2:
                B, detJ = self._bmat_at(xi, eta)
                Bs.append(B)
                dets.append(detJ)
                N = 0.25 * np.array([
                    (1 - xi) * (1 - eta), (1 + xi) * (1 - eta),
                    (1 + xi) * (1 + eta), (1 - xi) * (1 + eta)])
                coords.append(np.einsum("a,eai->ei", N, xy))
        # gp-major stacks (4, ne, ...) kept for elastic assembly mirroring
        self.B_gp = np.stack(Bs)                          # (4, ne, 3, 8)
        self.detJ_gp = np.stack(dets)                     # (4, ne)
        # quadrature-point-flat, ELEMENT-major: q = e*4 + gp
        self.n_qp = 4 * self.n_elem
        self.B_qp = np.ascontiguousarray(
            self.B_gp.transpose(1, 0, 2, 3).reshape(self.n_qp, 3, 8))
        self.w_qp = np.ascontiguousarray(
            self.detJ_gp.T.reshape(self.n_qp))            # weight * detJ (w=1)
        xy_qp = np.stack(coords).transpose(1, 0, 2).reshape(self.n_qp, 2)
        self.xy_qp = xy_qp
        self.r_qp = np.hypot(xy_qp[:, 0], xy_qp[:, 1])
        self.th_qp = np.arctan2(xy_qp[:, 1], xy_qp[:, 0])
        self.qp_elem = np.repeat(np.arange(self.n_elem), 4)

        # element-center geometry (Barlow points for stress recovery)
        Bc, detc = self._bmat_at(0.0, 0.0)
        self.B_center = Bc
        xc = xy.mean(axis=1)
        self.r_center = np.hypot(xc[:, 0], xc[:, 1])
        self.th_center = np.arctan2(xc[:, 1], xc[:, 0])

    def _precompute_torch(self):
        t = lambda x: torch.as_tensor(x, dtype=DTYPE)
        self.B_qp_t = t(self.B_qp)
        self.w_qp_t = t(self.w_qp)
        self.edof_t = torch.as_tensor(self.edof, dtype=torch.int64)
        self.ring_dofs_inner_t = torch.as_tensor(
            self.ring_dofs("inner"), dtype=torch.int64)
        self.ring_dofs_outer_t = torch.as_tensor(
            self.ring_dofs("outer"), dtype=torch.int64)
        # inner-ring edge Gauss geometry for differentiable ring loads
        for ring in ("inner", "outer"):
            nodes = self.ring_nodes(ring)
            n0 = nodes
            n1 = nodes[np.r_[1:self.n_t, 0]]
            x0, x1 = self.xy[n0], self.xy[n1]
            L = np.hypot(*(x1 - x0).T)
            setattr(self, f"_edge_{ring}", (n0, n1, x0, x1, L))
            setattr(self, f"_edge_{ring}_t",
                    (torch.as_tensor(2 * n0), torch.as_tensor(2 * n0 + 1),
                     torch.as_tensor(2 * n1), torch.as_tensor(2 * n1 + 1),
                     t(x0), t(x1), t(L)))

    # ------------------------------------------------------ strain eval
    def qp_strains(self, u):
        """In-plane engineering strains (nq, 3) of nodal vector u (numpy)."""
        ue = u[self.edof]                                 # (ne, 8)
        ue_q = np.repeat(ue, 4, axis=0)                   # (nq, 8)
        return np.einsum("qab,qb->qa", self.B_qp, ue_q)

    def qp_strains_t(self, u_t):
        ue = u_t[self.edof_t.reshape(-1)].reshape(self.n_elem, 8)
        ue_q = torch.repeat_interleave(ue, 4, dim=0)
        return torch.einsum("qab,qb->qa", self.B_qp_t, ue_q)
