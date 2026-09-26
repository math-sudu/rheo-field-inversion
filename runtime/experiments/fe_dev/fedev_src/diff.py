"""End-to-end differentiability through the multi-increment solve.

Mechanism (documented package choice): IMPLICIT DIFFERENTIATION of the
converged increment residuals, realized as a differentiable replay.
The forward Newton solve runs outside the autograd graph (numpy/scipy
sparse LU).  The replay then rebuilds, increment by increment, the
differentiable chain

    u_n     = u_n* - J_n^{-1} R(u_n*; state_{n-1}, u_{n-1}, theta)
    state_n = ReturnMap(state_{n-1}, eps(u_n - u_{n-1}); theta)

where u_n* is the (detached) converged iterate, J_n the consistent
Jacobian factorized at u_n* (detached SuperLU, applied through a custom
autograd function whose backward solves with the TRANSPOSED factors --
exact for the nonsymmetric tangents of non-associated flow), and
R(.) is evaluated in torch with full parameter/state dependence.  At
exact convergence R(u_n*) = 0, so d u_n / d theta equals the implicit-
function derivative; the residual of the correction is O(newton_tol).
State propagation across increments (the path dependence of
plasticity) flows through the torch ReturnMap chain, which carries the
committed pair (sig, kappa) -- for kind "cwfs" the softening laws
consume kappa, so gradients flow through the internal-variable history.

Differentiable inputs (theta dict, all optional except the material's):
  'E', 'nu', 'sigma_v', 'K0', 'lambdas' (full knot tensor, len M+1),
  MC: 'c', 'phi', 'psi';  HB: 'sigma_ci', 'm_b', 's', 'a_hb', 'psi_hb';
  CWFS: 'c_peak', 'c_res', 'gamma_c', 'phi_peak', 'phi_res',
  'gamma_phi', 'psi';  Robin: 'beta'.
Boundary DATA (Dirichlet g, Neumann loads, Robin mu, condensed K_bnd)
are treated as theta-independent constants in this batch.
"""

import numpy as np
import torch

from . import assembly, material, material_visc
from .meshing import DTYPE


class _SpluSolve(torch.autograd.Function):
    """Differentiable action of a fixed (detached) sparse LU inverse."""

    @staticmethod
    def forward(ctx, rhs, lu):
        ctx.lu = lu
        x = lu.solve(np.ascontiguousarray(rhs.detach().numpy()))
        return torch.from_numpy(x)

    @staticmethod
    def backward(ctx, grad_out):
        g = ctx.lu.solve(np.ascontiguousarray(grad_out.detach().numpy()),
                         trans="T")
        return torch.from_numpy(g), None


def lu_solve_t(lu, rhs_t):
    return _SpluSolve.apply(rhs_t, lu)


def make_theta(system, requires_grad=(), overrides=None):
    """Build the theta dict from the system's parameters.

    ``requires_grad``: iterable of keys to mark differentiable.
    ``overrides``: optional {key: float or array} replacing base values.
    """
    base, mat = system.base, system.mat
    vals = {"E": base.E, "nu": base.nu,
            "sigma_v": base.sigma_v, "K0": base.K0}
    if mat.kind != "elastic":
        vals.update(mat.theta())
    if overrides:
        vals.update(overrides)
    theta = {}
    for k, v in vals.items():
        theta[k] = torch.tensor(v, dtype=DTYPE,
                                requires_grad=(k in requires_grad))
    return theta


def replay(run, theta, qoi_fn):
    """Differentiable replay of a forward RunResult; returns scalar QoI.

    ``theta`` must contain the elastic + material keys; 'lambdas' is
    optional (defaults to the run's schedule as constants); 'beta'
    optional for Robin increments.  ``qoi_fn(u_t, sig_t, system)``
    returns a scalar torch tensor.
    """
    system = run.system
    mesh = system.mesh
    kind = system.kind
    if kind in material_visc.VISCOUS_KINDS:
        raise NotImplementedError(
            "replay differentiation of viscous (tier C) runs is out of "
            "scope for this batch (solver.py module docstring): the "
            "condensed replay contract stays tier A/B")
    if getattr(system, "noncircular", False):
        raise NotImplementedError(
            "replay differentiation of non-circular-section runs is out "
            "of scope (EXEC-7d): replay rebuilds the excavation load "
            "through the circular-wall twin assembly.inner_unit_load_t, "
            "which does not represent the mapped-wall normal-field load "
            "of fedev_src.noncirc.NonCircFESystem")
    E_t, nu_t = theta["E"], theta["nu"]
    sv_t, K0_t = theta["sigma_v"], theta["K0"]
    mat_t = ({} if kind == "elastic" else
             {k: theta[k] for k in system.mat.theta()})
    if kind == "cwfs":
        # Perzyna regularization setting of the replayed run
        # (params.CWFSMohrCoulomb.visc_H): a theta-independent
        # CONSTANT, like the default lambdas -- the replay reproduces
        # the forward operator, it does not differentiate w.r.t. the
        # regularization.
        mat_t["visc_H"] = torch.tensor(system.mat.visc_H, dtype=DTYPE)
    lam_knots = theta.get("lambdas")
    if lam_knots is None:
        lam_knots = torch.as_tensor(run.lambdas, dtype=DTYPE)

    zero = torch.zeros((), dtype=DTYPE)
    in_plane_t = torch.tensor(material.IN_PLANE_IDX, dtype=torch.int64)
    sig0_t = torch.stack([-K0_t * sv_t, -sv_t, -K0_t * sv_t, zero])
    sig0_in_t = sig0_t.index_select(0, in_plane_t)
    f1_t = assembly.inner_unit_load_t(mesh, sv_t, K0_t)

    outer_dofs_t = mesh.ring_dofs_outer_t
    m_out_t = torch.as_tensor(system.m_out, dtype=DTYPE)

    sig_committed = sig0_t.unsqueeze(0).expand(mesh.n_qp, 4)
    kap_committed = torch.zeros(mesh.n_qp, dtype=DTYPE)
    u_prev = torch.zeros(mesh.n_dof, dtype=DTYPE)

    def b_out_apply(vec_t):
        vr = vec_t[outer_dofs_t].reshape(mesh.n_t, 2)
        out = torch.zeros(mesh.n_dof, dtype=DTYPE)
        return out.index_add(0, outer_dofs_t, (m_out_t @ vr).reshape(-1))

    for ex in run.executed:
        tr = ex.trial
        md = tr.mode
        k = ex.knot
        lam_t = (lam_knots[k]
                 + ex.t1 * (lam_knots[k + 1] - lam_knots[k]))
        u_det = torch.tensor(np.asarray(tr.u), dtype=DTYPE)

        deps_t = mesh.qp_strains_t(u_det - u_prev)
        sig_new = material.integrate_full(deps_t, sig_committed,
                                          kap_committed, E_t, nu_t,
                                          mat_t, kind)[0]
        R_bulk = (assembly.f_int_qp_t(
            mesh, sig_new.index_select(1, in_plane_t) - sig0_in_t)
            - lam_t * f1_t)

        mk = md["kind"]
        if mk == "dir":
            free_t = torch.as_tensor(md["free"])
            fixed_t = torch.as_tensor(md["fixed"])
            g_t = torch.as_tensor(md["g"], dtype=DTYPE)
            delta = lu_solve_t(tr.lu, R_bulk[free_t])
            u_new = torch.zeros(mesh.n_dof, dtype=DTYPE)
            u_new = u_new.index_copy(0, free_t, u_det[free_t] - delta)
            u_new = u_new.index_copy(0, fixed_t, g_t)
        elif mk == "neu":
            C_t = torch.as_tensor(system.rigid, dtype=DTYPE)
            om_t = torch.as_tensor(md["omega"], dtype=DTYPE)
            f_out_t = torch.as_tensor(md["f_out"], dtype=DTYPE)
            R_aug = torch.cat([R_bulk + C_t @ om_t - f_out_t,
                               C_t.T @ u_det])
            delta = lu_solve_t(tr.lu, R_aug)
            u_new = u_det - delta[:mesh.n_dof]
        elif mk == "rob":
            beta_t = theta.get("beta")
            if beta_t is None:
                beta_t = torch.tensor(md["beta"], dtype=DTYPE)
            f_mu_t = torch.as_tensor(md["f_mu"], dtype=DTYPE)
            R_rob = R_bulk + beta_t * b_out_apply(u_det) - f_mu_t
            delta = lu_solve_t(tr.lu, R_rob)
            u_new = u_det - delta
        elif mk == "cond":
            C_t = torch.as_tensor(system.rigid[:, :2], dtype=DTYPE)
            om_t = torch.as_tensor(md["omega"], dtype=DTYPE)
            Kb_t = torch.as_tensor(md["K_bnd"], dtype=DTYPE)
            ur = u_det[outer_dofs_t]
            Kb_u = torch.zeros(mesh.n_dof, dtype=DTYPE).index_add(
                0, outer_dofs_t, Kb_t @ ur)
            R_aug = torch.cat([R_bulk + Kb_u + C_t @ om_t,
                               C_t.T @ u_det])
            delta = lu_solve_t(tr.lu, R_aug)
            u_new = u_det - delta[:mesh.n_dof]
        else:
            raise ValueError(mk)

        deps_new = mesh.qp_strains_t(u_new - u_prev)
        out = material.integrate_full(deps_new, sig_committed,
                                      kap_committed, E_t, nu_t, mat_t,
                                      kind)
        sig_committed = out[0]
        kap_committed = kap_committed + out[4]
        u_prev = u_new

    return qoi_fn(u_prev, sig_committed, system)


def qoi_and_grads(run, keys, qoi_fn, overrides=None):
    """Convenience: replay + autograd gradients for the given keys."""
    theta = make_theta(run.system, requires_grad=keys,
                       overrides=overrides)
    if "lambdas" in keys:
        theta["lambdas"] = torch.tensor(run.lambdas, dtype=DTYPE,
                                        requires_grad=True)
    q = replay(run, theta, qoi_fn)
    params = [theta[k] for k in keys]
    grads = torch.autograd.grad(q, params, allow_unused=False)
    return float(q.detach()), {k: g.detach().numpy() if g.dim() else
                               float(g) for k, g in zip(keys, grads)}
