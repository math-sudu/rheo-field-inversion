"""Assembly: stiffness/tangent matrices, boundary loads, boundary mass.

Mirrors the prototype's discrete operators exactly (same Gauss rules,
same accumulation order for the elastic stiffness, same 3-point Gauss
consistent ring loads, same P1 boundary mass and rigid-vector
normalization) and adds:

* per-quadrature-point consistent-tangent assembly for Newton,
* internal perturbation force f_int = sum B^T (sigma - sigma0) dV,
  in numpy (forward path) and torch (differentiable replay path),
* differentiable inner-ring unit perturbation load (function of the
  in-situ parameters sigma_v, K0): the staged excavation traction on
  r = a is  t_total = (1-lambda) * (sigma0 . n),  n = -e_r,  so the
  PERTURBATION load is  lambda * (sigma0 . e_r)  (tension-positive);
  at lambda = 1 it reproduces the prototype's ``inner_load`` exactly.
"""

import numpy as np
import scipy.sparse as sp
import torch

from .meshing import G2, G3_P, G3_W, DTYPE


def dmat_plane_strain(E, nu):
    c = E / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return c * np.array([
        [1.0 - nu, nu, 0.0],
        [nu, 1.0 - nu, 0.0],
        [0.0, 0.0, 0.5 * (1.0 - 2.0 * nu)],
    ])


def assemble_elastic_K(mesh, E, nu):
    """Elastic stiffness, mirroring the prototype's accumulation order."""
    D = dmat_plane_strain(E, nu)
    ne = mesh.n_elem
    Ke = np.zeros((ne, 8, 8))
    for g in range(4):                       # prototype's xi/eta loop order
        B, detJ = mesh.B_gp[g], mesh.detJ_gp[g]
        Ke += np.einsum("eka,kl,elb->eab", B, D, B) * detJ[:, None, None]
    rows = np.repeat(mesh.edof, 8, axis=1).ravel()
    cols = np.tile(mesh.edof, (1, 8)).ravel()
    return sp.coo_matrix((Ke.ravel(), (rows, cols)),
                         shape=(mesh.n_dof, mesh.n_dof)).tocsr()


def assemble_tangent_K(mesh, D_qp):
    """Tangent stiffness from per-QP consistent moduli D_qp (nq, 3, 3)."""
    BW = mesh.B_qp * mesh.w_qp[:, None, None]
    Kq = np.einsum("qka,qkl,qlb->qab", BW, D_qp, mesh.B_qp)
    Ke = Kq.reshape(mesh.n_elem, 4, 8, 8).sum(axis=1)
    rows = np.repeat(mesh.edof, 8, axis=1).ravel()
    cols = np.tile(mesh.edof, (1, 8)).ravel()
    return sp.coo_matrix((Ke.ravel(), (rows, cols)),
                         shape=(mesh.n_dof, mesh.n_dof)).tocsr()


# ------------------------------------------------------- internal force

def f_int_qp(mesh, sig_inplane_qp):
    """Consistent internal force of a QP stress field (numpy, (nq,3))."""
    fq = np.einsum("qka,qk->qa", mesh.B_qp,
                   sig_inplane_qp * mesh.w_qp[:, None])
    fe = fq.reshape(mesh.n_elem, 4, 8).sum(axis=1)
    f = np.zeros(mesh.n_dof)
    np.add.at(f, mesh.edof, fe)
    return f


def f_int_qp_t(mesh, sig_inplane_qp_t):
    """Torch twin of :func:`f_int_qp` (differentiable)."""
    fq = torch.einsum("qka,qk->qa", mesh.B_qp_t,
                      sig_inplane_qp_t * mesh.w_qp_t[:, None])
    fe = fq.reshape(mesh.n_elem, 4, 8).sum(dim=1)
    f = torch.zeros(mesh.n_dof, dtype=DTYPE)
    return f.index_add(0, mesh.edof_t.reshape(-1), fe.reshape(-1))


# ----------------------------------------------------------- ring loads

def ring_load(mesh, ring, traction_xy_fn):
    """Consistent nodal loads of a traction FIELD t(x, y) on a ring.

    Numpy mirror of the prototype's ``ring_load`` (3-point Gauss).
    """
    n0, n1, x0, x1, L = getattr(mesh, f"_edge_{ring}")
    f = np.zeros(mesh.n_dof)
    for p, w in zip(G3_P, G3_W):
        N0, N1 = 0.5 * (1.0 - p), 0.5 * (1.0 + p)
        xp = N0 * x0 + N1 * x1
        tx, ty = traction_xy_fn(xp[:, 0], xp[:, 1])
        c = w * 0.5 * L
        np.add.at(f, 2 * n0, c * N0 * tx)
        np.add.at(f, 2 * n0 + 1, c * N0 * ty)
        np.add.at(f, 2 * n1, c * N1 * tx)
        np.add.at(f, 2 * n1 + 1, c * N1 * ty)
    return f


def inner_unit_load(mesh, sigma_v, K0):
    """Unit (lambda = 1) perturbation load on r = a: t = sigma0 . e_r.

    sigma0 tension-positive (see ``params.BaseParams.sigma0``); with
    s_xy0 = 0 the Cartesian traction is (s_xx0*cos, s_yy0*sin) theta.
    """
    sxx0, syy0 = -K0 * sigma_v, -sigma_v

    def t_fn(x, y):
        r = np.hypot(x, y)
        return sxx0 * x / r, syy0 * y / r

    return ring_load(mesh, "inner", t_fn)


def inner_unit_load_t(mesh, sigma_v_t, K0_t):
    """Differentiable torch twin of :func:`inner_unit_load`."""
    i0x, i0y, i1x, i1y, x0, x1, L = mesh._edge_inner_t
    sxx0, syy0 = -K0_t * sigma_v_t, -sigma_v_t
    f = torch.zeros(mesh.n_dof, dtype=DTYPE)
    for p, w in zip(G3_P, G3_W):
        N0, N1 = 0.5 * (1.0 - p), 0.5 * (1.0 + p)
        xp = N0 * x0 + N1 * x1
        r = torch.hypot(xp[:, 0], xp[:, 1])
        tx = sxx0 * xp[:, 0] / r
        ty = syy0 * xp[:, 1] / r
        c = w * 0.5 * L
        f = f.index_add(0, i0x, c * N0 * tx)
        f = f.index_add(0, i0y, c * N0 * ty)
        f = f.index_add(0, i1x, c * N1 * tx)
        f = f.index_add(0, i1y, c * N1 * ty)
    return f


# ------------------------------------------------- boundary mass / misc

def ring_mass(mesh, ring):
    """Scalar P1 boundary mass matrix (n_t, n_t), prototype mirror."""
    M = np.zeros((mesh.n_t, mesh.n_t))
    nodes = list(mesh.ring_nodes(ring))
    index = {n: i for i, n in enumerate(nodes)}
    n0s, n1s = getattr(mesh, f"_edge_{ring}")[:2]
    for n0, n1 in zip(n0s, n1s):
        i, j = index[n0], index[n1]
        L = np.hypot(*(mesh.xy[n1] - mesh.xy[n0]))
        M[i, i] += L / 3.0
        M[j, j] += L / 3.0
        M[i, j] += L / 6.0
        M[j, i] += L / 6.0
    return M


def b_out_global(mesh, m_out):
    """Outer-ring boundary mass as a global sparse matrix (proto mirror)."""
    nodes = mesh.outer_nodes
    n_t = mesh.n_t
    rows, cols, vals = [], [], []
    for c in (0, 1):
        for i in range(n_t):
            for j in range(n_t):
                if m_out[i, j] != 0.0:
                    rows.append(2 * nodes[i] + c)
                    cols.append(2 * nodes[j] + c)
                    vals.append(m_out[i, j])
    return sp.coo_matrix((vals, (rows, cols)),
                         shape=(mesh.n_dof, mesh.n_dof)).tocsr()


def rigid_vectors(mesh):
    """Normalized rigid-body vectors (n_dof, 3), prototype mirror."""
    tx = np.zeros(mesh.n_dof)
    tx[0::2] = 1.0
    ty = np.zeros(mesh.n_dof)
    ty[1::2] = 1.0
    rot = np.zeros(mesh.n_dof)
    rot[0::2] = -mesh.xy[:, 1]
    rot[1::2] = mesh.xy[:, 0]
    return np.column_stack([v / np.linalg.norm(v) for v in (tx, ty, rot)])


def scatter_ring_block(mesh, K_bnd):
    """Scatter a dense (2 n_t, 2 n_t) outer-ring dof block to global CSR.

    ``K_bnd`` acts on the outer-ring Cartesian dofs in interleaved
    node-major order [x_0, y_0, x_1, y_1, ...] (= ``ring_dofs('outer')``
    ordering), exactly the layout of the prototype's condensed boundary
    contribution.
    """
    n2 = 2 * mesh.n_t
    if K_bnd.shape != (n2, n2):
        raise ValueError("K_bnd must be (2 n_t, 2 n_t)")
    dofs = mesh.ring_dofs("outer")
    rows = np.repeat(dofs, n2)
    cols = np.tile(dofs, n2)
    return sp.coo_matrix((np.asarray(K_bnd, dtype=float).ravel(),
                          (rows, cols)),
                         shape=(mesh.n_dof, mesh.n_dof)).tocsr()
