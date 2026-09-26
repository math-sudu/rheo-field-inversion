"""Viscous differentiable replay + Jacobians (AD-through-FE, EXEC-19).

Mirrors the implicit-differentiation replay of ``fedev_src.diff``
(tier A/B contract, READ-ONLY) and extends it to the tier-C viscous
state chain -- the extension lives HERE; ``diff.py``'s
NotImplementedError boundary is untouched (exec7b section 10-3 /
exec8 section 9-4 lineage: "another row builds it outside fedev_src").

Chain per executed increment n (converged detached iterate u*_n, its
final consistent SuperLU J_n, physical-time step dt_n)::

    u_n     = u*_n - J_n^{-1} R(u*_n; u_{n-1}, z_{n-1}, theta)
    z_n     = IntegrateVisc(strains(u_n - u_{n-1}), z_{n-1}; theta, dt_n)

with z = (sig, eps_K, kappa) and R the condensed-mode residual
(R_bulk + K_bnd u + C omega; C^T u) evaluated in torch with full
theta/state dependence.  At convergence R = O(newton_tol) so u_n
equals the implicit-function value; gradients flow through the state
chain (path dependence) exactly as in the tier-A/B replay.

Differences against ``diff.replay`` (each a REQUIRED correctness item
for this parameter surface, not a style choice):

* material bridge = ``material_visc.integrate_visc`` with the Kelvin
  state threaded increment-to-increment (REPLACED per commit, solver
  State contract) and dt_n = (t1-t0)(times[k+1]-times[k]) recomputed
  from the executed record;
* the condensed boundary block K_bnd is the ELASTIC far-field DtN --
  linear in E at fixed nu -- so the replay threads
  K_bnd(E) = (E / E_run) K_bnd_run exactly instead of a constant
  (diff.py's constant treatment is correct only when E is not a
  differentiation target; here s_star -> E is primary).  Verified by
  a rebuild lock in the tests;
* the excavation unit load f1(sigma_v, K0) is rebuilt from TWO numpy
  loader calls as an exact affine basis
  f1 = sigma_v (K0 A + B),  A = f1(1,1) - f1(1,0),  B = f1(1,0)
  -- valid for the circular AND the mapped-wall loader (both are
  linear in sigma_v and affine in K0 by construction), which lifts
  the circular-twin restriction that makes diff.replay reject
  non-circular systems.  Locked element-wise against system.f1.

Quotient-coordinate graph: ``theta_graph`` builds the physical torch
parameters (E, K0, material 7-key dict) FROM quotient leaves
(s_star, K0, r_K, tau_K, tau_M, r_s, tau_vp), so autograd returns
quotient-coordinate derivatives directly.
"""

from __future__ import annotations

import numpy as np
import torch

from . import _paths  # noqa: F401
from .scenario import PHI, PSI, NU_FIXED, SIGMA_V_MPA

from fedev_src import assembly, material, material_visc    # noqa: E402
from fedev_src import noncirc as fnoncirc                  # noqa: E402
from .directional import lu_solve_t                       # noqa: E402
from fedev_src.meshing import DTYPE                        # noqa: E402

QUOTIENT_NAMES = ("s_star", "K0", "r_K", "tau_K", "tau_M", "r_s",
                  "tau_vp")


def theta_graph(scn, requires_grad=(), overrides=None):
    """Quotient leaves + derived physical tensors.

    Returns (leaves, phys): ``leaves[name]`` are the quotient torch
    leaf tensors; ``phys`` is a dict with E, nu, sigma_v, K0 and the
    NishiharaMC 7-key material dict, all built as graph functions of
    the leaves (constants where the scenario leg is disabled).
    """
    vals = {n: getattr(scn, n) for n in QUOTIENT_NAMES}
    if overrides:
        vals.update(overrides)
    leaves = {}
    for n in QUOTIENT_NAMES:
        v = vals[n]
        if v is None:
            leaves[n] = None
            continue
        leaves[n] = (v if torch.is_tensor(v) else
                     torch.tensor(float(v), dtype=DTYPE,
                                  requires_grad=(n in requires_grad)))
    sv = torch.tensor(scn.sigma_v, dtype=DTYPE)
    nu = torch.tensor(scn.nu, dtype=DTYPE)
    G0 = sv / (2.0 * leaves["s_star"])
    E = 2.0 * (1.0 + nu) * G0
    G_K = leaves["r_K"] * G0
    E_K = 2.0 * (1.0 + nu) * G_K
    eta_K = leaves["tau_K"] * G_K
    eta_M = leaves["tau_M"] * 2.0 * G0
    n_phi = (1.0 + np.sin(PHI)) / (1.0 - np.sin(PHI))
    if leaves["r_s"] is None:
        c = torch.tensor(1.0e9, dtype=DTYPE)
        eta_vp = torch.tensor(1.0e12, dtype=DTYPE)
    else:
        c = leaves["r_s"] * sv / (2.0 * np.sqrt(n_phi))
        eta_vp = (leaves["tau_vp"] * 2.0 * G0
                  * 0.5 * (1.0 - np.sin(PHI)) * (1.0 - np.sin(PSI)))
    mat_t = {"c": c,
             "phi": torch.tensor(PHI, dtype=DTYPE),
             "psi": torch.tensor(PSI, dtype=DTYPE),
             "E_K": E_K, "eta_K": eta_K, "eta_M": eta_M,
             "eta_vp": eta_vp}
    phys = {"E": E, "nu": nu, "sigma_v": sv, "K0": leaves["K0"],
            "mat": mat_t}
    return leaves, phys


def load_basis(system):
    """Exact affine excavation-load basis (A, B): f1 = sv (K0 A + B)."""
    mesh = system.mesh
    loader = (fnoncirc.noncirc_inner_unit_load
              if getattr(system, "noncircular", False)
              else assembly.inner_unit_load)
    f11 = loader(mesh, 1.0, 1.0)
    f10 = loader(mesh, 1.0, 0.0)
    A = torch.tensor(f11 - f10, dtype=DTYPE)
    B = torch.tensor(f10, dtype=DTYPE)
    return A, B


def replay_visc(run, phys, collect_u=True):
    """Differentiable viscous replay.

    ``phys`` from :func:`theta_graph`.  Returns (t_list, u_list,
    aux) with u_list the per-committed-increment differentiable
    displacement tensors (t_list floats).  ``aux['sig_last']`` is the
    final committed stress tensor (graph-connected).
    """
    system = run.system
    mesh = system.mesh
    kind = system.kind
    if kind not in material_visc.VISCOUS_KINDS:
        raise ValueError("replay_visc is the tier-C lane; use "
                         "fedev_src.diff.replay for tier A/B")
    if run.times is None:
        raise ValueError("viscous RunResult must carry times")

    E_t, nu_t = phys["E"], phys["nu"]
    sv_t, K0_t = phys["sigma_v"], phys["K0"]
    mat_t = phys["mat"]

    zero = torch.zeros((), dtype=DTYPE)
    in_plane_t = torch.tensor(material.IN_PLANE_IDX, dtype=torch.int64)
    sig0_t = torch.stack([-K0_t * sv_t, -sv_t, -K0_t * sv_t, zero])
    sig0_in_t = sig0_t.index_select(0, in_plane_t)
    A_l, B_l = load_basis(system)
    f1_t = sv_t * (K0_t * A_l + B_l)

    outer_dofs_t = mesh.ring_dofs_outer_t
    E_run = float(system.base.E)

    lam_knots = torch.as_tensor(run.lambdas, dtype=DTYPE)
    times = np.asarray(run.times, dtype=float)

    sig_committed = sig0_t.unsqueeze(0).expand(mesh.n_qp, 4)
    epsK_committed = torch.zeros(mesh.n_qp, 4, dtype=DTYPE)
    kap_committed = torch.zeros(mesh.n_qp, dtype=DTYPE)
    u_prev = torch.zeros(mesh.n_dof, dtype=DTYPE)

    t_list, u_list = [], []
    for ex in run.executed:
        tr = ex.trial
        md = tr.mode
        if md["kind"] != "cond":
            raise NotImplementedError(
                f"replay_visc covers the condensed production mode; "
                f"got {md['kind']!r}")
        k = ex.knot
        lam_t = (lam_knots[k]
                 + ex.t1 * (lam_knots[k + 1] - lam_knots[k]))
        dt_n = (ex.t1 - ex.t0) * (times[k + 1] - times[k])
        u_det = torch.tensor(np.asarray(tr.u), dtype=DTYPE)

        deps_t = mesh.qp_strains_t(u_det - u_prev)
        sig_new = material_visc.integrate_visc(
            deps_t, sig_committed, epsK_committed, kap_committed,
            dt_n, E_t, nu_t, mat_t, kind)[0]
        R_bulk = (assembly.f_int_qp_t(
            mesh, sig_new.index_select(1, in_plane_t) - sig0_in_t)
            - lam_t * f1_t)

        C_t = torch.as_tensor(system.rigid[:, :2], dtype=DTYPE)
        om_t = torch.as_tensor(md["omega"], dtype=DTYPE)
        Kb_t = torch.as_tensor(md["K_bnd"], dtype=DTYPE) \
            * (E_t / E_run)
        ur = u_det[outer_dofs_t]
        Kb_u = torch.zeros(mesh.n_dof, dtype=DTYPE).index_add(
            0, outer_dofs_t, Kb_t @ ur)
        R_aug = torch.cat([R_bulk + Kb_u + C_t @ om_t,
                           C_t.T @ u_det])
        delta = lu_solve_t(tr.lu, R_aug)
        u_new = u_det - delta[:mesh.n_dof]

        deps_new = mesh.qp_strains_t(u_new - u_prev)
        out = material_visc.integrate_visc(
            deps_new, sig_committed, epsK_committed, kap_committed,
            dt_n, E_t, nu_t, mat_t, kind)
        sig_committed = out[0]
        epsK_committed = out[3]
        kap_committed = kap_committed + out[4]
        u_prev = u_new
        t_list.append(float(tr.t))
        if collect_u:
            u_list.append(u_new)

    aux = {"sig_last": sig_committed, "kappa_last": kap_committed}
    return t_list, u_list, aux


def qoi_series_and_jacobian(run, scn, names, qoi_of_u, at_indices):
    """Reverse-mode Jacobian of a per-increment scalar QoI series.

    ``qoi_of_u(u_t) -> scalar tensor``; ``at_indices`` selects
    committed increments.  Returns (values (m,), J (m, p)) in the
    QUOTIENT coordinates ``names``.
    """
    leaves, phys = theta_graph(scn, requires_grad=names)
    t_list, u_list, _ = replay_visc(run, phys)
    params = [leaves[n] for n in names]
    vals, rows = [], []
    sel = list(at_indices)
    for j, idx in enumerate(sel):
        q = qoi_of_u(u_list[idx])
        vals.append(float(q.detach()))
        grads = torch.autograd.grad(
            q, params, retain_graph=(j < len(sel) - 1),
            allow_unused=False)
        rows.append([float(g) for g in grads])
    return np.asarray(vals), np.asarray(rows)


def kb_scaling_lock(system, factor=2.0):
    """Verify K_bnd(E) = (E/E_ref) K_bnd(E_ref) exactly (test hook)."""
    from fedev_src import coupling as fcoupling
    Kb1 = fcoupling.condensed_boundary_block(system.mesh, system.base)
    base2 = type(system.base)(
        a=system.base.a, sigma_v=system.base.sigma_v,
        K0=system.base.K0, E=factor * system.base.E, nu=system.base.nu)
    Kb2 = fcoupling.condensed_boundary_block(system.mesh, base2)
    return float(np.max(np.abs(Kb2 - factor * Kb1))
                 / max(1e-30, np.max(np.abs(Kb1))))
