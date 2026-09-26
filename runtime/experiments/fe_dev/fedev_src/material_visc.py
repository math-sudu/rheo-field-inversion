"""Nishihara visco-elasto-plastic material point (tier C), torch f64.

Maxwell-extended (Burgers-type) Nishihara chain around the tier-A
perfect-MC return (series chain: one total stress, strains add;
tension-positive)::

    eps = C_e(E, nu)^-1 : sig  +  eps_K  +  eps_M  +  eps_vp

NAMING (see params.NishiharaMC): the CLASSICAL Nishihara model has NO
free Maxwell dashpot (instantaneous elastic + Kelvin + thresholded
viscoplastic element only) and is recovered exactly as eta_M -> inf;
the classical/dtPINN convention's thresholded dashpot eta_3
corresponds to eta_vp here, NEVER to eta_M.

with DEVIATORIC-ONLY viscous flow -- the standard engineering
assumption; the volumetric response stays elastic with the bulk
modulus K of the elastic spring, and eps_K, eps_M are trace-free::

    Kelvin:   eps_K_dot = (s - 2 G_K eps_K) / (2 eta_K),  s = dev(sig)
    Maxwell:  eps_M_dot = s / (2 eta_M)
    Perzyna:  eps_vp_dot = <f_MC>/eta_vp * dg/dsig
              (MC-type potential, constant psi -- same as tier A)

G_K = E_K / (2 (1 + nu)) at the elastic Poisson ratio (see
params.NishiharaMC).  The Maxwell dashpot needs NO stored state: its
increment depends only on the converged stress, so the material state
is (sig, eps_K, kappa) only.

Backward-Euler condensation (the module's derivation contract)
---------------------------------------------------------------

Over a step dt from the committed state (sig_n, eps_K_n) with total
strain increment deps (deviatoric part de, volumetric part tr(deps)),
the fully implicit viscous updates are ::

    eps_K_(n+1) = (eps_K_n + dt/(2 eta_K) s_(n+1)) / (1 + dt G_K/eta_K)
    Delta eps_M = dt/(2 eta_M) s_(n+1)

so with  alpha_K = dt/(eta_K + dt G_K)  and  alpha_M = dt/eta_M ::

    Delta eps_K = (alpha_K/2) s_(n+1) - G_K alpha_K eps_K_n .

Substituting into the deviatoric strain balance
de = Delta s/(2 G0) + Delta eps_K + Delta eps_M + dev(Delta eps_vp)
and collecting s_(n+1) ::

    s_(n+1) = 2 G_eff [ de + s_n/(2 G0) + G_K alpha_K eps_K_n ]
              - 2 G_eff dev(Delta eps_vp) ,
    1/G_eff = 1/G0 + alpha_K + alpha_M .

The volumetric balance (viscous elements deviatoric) is
p_(n+1) = p_n + K (tr deps - tr Delta eps_vp).  Together these are
EXACTLY the rate-independent relation
sig_(n+1) = sig_tr_eff - C_eff : Delta eps_vp on the effective
isotropic elasticity C_eff built from (K, G_eff(dt)) -- converted to
(E_eff, nu_eff) for the principal-space return -- around the
viscoelastically corrected trial ::

    sig_tr_eff = [p_n + K tr(deps)] I
                 + 2 G_eff [ de + s_n/(2 G0) + G_K alpha_K eps_K_n ] .

The Perzyna consistency f(sig_(n+1)) = (eta_vp/dt) * dlam is carried
by the return itself (material._mc_sorted with visc_H = eta_vp/dt),
so ONE rate-independent-style return solves the coupled viscoelastic-
viscoplastic backward-Euler system exactly.  After the return the
Kelvin state is reconstructed from the CONVERGED deviatoric stress by
the same implicit formula, hence (sig_(n+1), eps_K_(n+1)) satisfy the
coupled system to round-off; on elastic steps (f_tr <= 0) the viscous
strains still evolve (stress relaxation / creep with dlam = 0).

Layouts (identical to material.integrate_full): stress 4-vectors
(s_xx, s_yy, s_zz, s_xy); strain-like 4-vectors -- eps_K state and the
returned plastic increment -- use ENGINEERING shear
(e_xx, e_yy, e_zz, gamma_xy); deps3 = (deps_xx, deps_yy, dgamma_xy) is
the plane-strain increment (deps_zz = 0).  ``integrate_visc4`` is the
general-kinematics core on a full strain increment
(deps_xx, deps_yy, deps_zz, dgamma_xy); the plane-strain entry
``integrate_visc`` restricts deps_zz = 0.  The core is exposed because
a constant-stress UNIAXIAL hold -- the classical creep-test gold
anchor -- is kinematically infeasible under deps_zz = 0 (the lateral
viscous contraction of the true uniaxial solution requires
deps_zz != 0, and freezing it makes the out-of-plane stress drift), so
the analytic uniaxial anchor drives the core, while the plane-strain
entry gets its own exact anchor on the PURE-SHEAR closed form, which
plane strain does represent (tests/test_nishihara_material.py).

Everything is vectorized torch float64, branchless (torch.where
masks inside the shared return machinery), and differentiable end to
end; the Nishihara-MC path is CLOSED FORM throughout (the VE
condensation is algebraic and the regularized MC face/edge/apex
returns stay closed form), so no detached iterative solve is needed
anywhere.  Consistent algorithmic tangents are exact reverse-mode
Jacobians of the update (torch.func.jacrev + vmap).  The nonsmooth
caveats of material.py apply unchanged (region-boundary trials,
repeated in-plane trial eigenvalues); derivative-based tests must not
sit exactly on them.

Requires dt > 0 and finite eta_K, eta_M, eta_vp > 0; eps_K4 must be
trace-free (the update preserves this to round-off).  ``kappa`` is
carried for API uniformity with the state-carrying tier-A/B entry and
reported back as dkappa = dlam (perfect-plasticity bookkeeping,
exactly the tier-A meaning of the internal-variable increment).
"""

import torch

from .material import (DTYPE, IN_PLANE_IDX, _mc_sorted, _principal4,
                       _recompose_strain, _recompose_stress, _sort3,
                       _unsort3)

VISCOUS_KINDS = ("nishihara",)
"""Material kinds whose system-level solves carry the physical-time
axis (state eps_K + committed t, strict dt contract -- see the
solver.py module docstring)."""

_I4 = torch.tensor([1.0, 1.0, 1.0, 0.0], dtype=DTYPE)


def _as_t(v):
    return torch.as_tensor(v, dtype=DTYPE)


def effective_moduli(dt, E, nu, E_K, eta_K, eta_M):
    """Step-condensed moduli of the viscoelastic sub-chain.

    Returns (G0, K, G_K, alpha_K, alpha_M, G_eff, E_eff, nu_eff); see
    the module docstring for the derivation.  (E_eff, nu_eff) is the
    (K, G_eff) pair re-expressed as engineering constants for the
    principal-space return (K unchanged -- volumetric response stays
    elastic).
    """
    dt, E, nu = _as_t(dt), _as_t(E), _as_t(nu)
    E_K, eta_K, eta_M = _as_t(E_K), _as_t(eta_K), _as_t(eta_M)
    G0 = E / (2.0 * (1.0 + nu))
    K = E / (3.0 * (1.0 - 2.0 * nu))
    G_K = E_K / (2.0 * (1.0 + nu))
    alpha_K = dt / (eta_K + dt * G_K)
    alpha_M = dt / eta_M
    G_eff = 1.0 / (1.0 / G0 + alpha_K + alpha_M)
    E_eff = 9.0 * K * G_eff / (3.0 * K + G_eff)
    nu_eff = (3.0 * K - 2.0 * G_eff) / (2.0 * (3.0 * K + G_eff))
    return G0, K, G_K, alpha_K, alpha_M, G_eff, E_eff, nu_eff


def _dev4(t4):
    """Deviator and mean of a TENSOR-component 4-vector (..., 4)."""
    m = (t4[..., 0] + t4[..., 1] + t4[..., 2]) / 3.0
    return t4 - m.unsqueeze(-1) * _I4, m


def _to_tensor_shear(v4):
    """(xx, yy, zz, gamma_xy) -> (xx, yy, zz, e_xy)."""
    return torch.stack([v4[..., 0], v4[..., 1], v4[..., 2],
                        0.5 * v4[..., 3]], dim=-1)


def _to_engineering_shear(v4):
    """(xx, yy, zz, e_xy) -> (xx, yy, zz, gamma_xy)."""
    return torch.stack([v4[..., 0], v4[..., 1], v4[..., 2],
                        2.0 * v4[..., 3]], dim=-1)


def integrate_visc4(deps4, sig4, eps_K4, kappa, dt, E, nu, mat):
    """General-kinematics Nishihara backward-Euler step (batch-free core).

    ``deps4 = (deps_xx, deps_yy, deps_zz, dgamma_xy)`` (engineering
    shear); all other arguments and the returns are those of
    :func:`integrate_visc` (which restricts deps_zz = 0).  All tensors
    broadcast on leading dimensions.
    """
    E, nu, dt = _as_t(E), _as_t(nu), _as_t(dt)
    mat = {k: _as_t(v) for k, v in mat.items()}
    deps4, sig4, eps_K4 = _as_t(deps4), _as_t(sig4), _as_t(eps_K4)

    G0, K, G_K, alpha_K, alpha_M, G_eff, E_eff, nu_eff = effective_moduli(
        dt, E, nu, mat["E_K"], mat["eta_K"], mat["eta_M"])
    visc_H = mat["eta_vp"] / dt

    deps_t = _to_tensor_shear(deps4)
    epsK_t = _to_tensor_shear(eps_K4)
    de, dm = _dev4(deps_t)                  # dm = tr(deps)/3
    s_n, p_n = _dev4(sig4)

    # viscoelastically corrected trial on the effective elasticity
    s_tr = 2.0 * G_eff * (de + s_n / (2.0 * G0)
                          + (G_K * alpha_K) * epsK_t)
    p_tr = p_n + 3.0 * K * dm
    sig_tr = s_tr + p_tr.unsqueeze(-1) * _I4

    # Perzyna-regularized MC return (visc_H = eta_vp/dt)
    sA, sB, sZ, c2, s2 = _principal4(sig_tr)
    s1, s2s, s3, m0, m2 = _sort3(sA, sB, sZ)
    t1, t2, t3, e1, e2, e3, dlam, region = _mc_sorted(
        s1, s2s, s3, E_eff, nu_eff, mat["c"], mat["phi"], mat["psi"],
        visc_H=visc_H)
    vA, vB, vZ = _unsort3(t1, t2, t3, m0, m2)
    eA, eB, eZ = _unsort3(e1, e2, e3, m0, m2)
    sig_pl = _recompose_stress(vA, vB, vZ, c2, s2)
    dep_pl = _recompose_strain(eA, eB, eZ, c2, s2)
    plast4 = (region > 0).unsqueeze(-1)
    sig_new = torch.where(plast4, sig_pl, sig_tr)
    deps_p = torch.where(plast4, dep_pl, torch.zeros_like(dep_pl))

    # Kelvin state from the CONVERGED deviatoric stress (same implicit
    # formula as the condensation -- the coupled system is satisfied)
    s_new, _ = _dev4(sig_new)
    epsK_new_t = ((epsK_t + (dt / (2.0 * mat["eta_K"])) * s_new)
                  / (1.0 + dt * G_K / mat["eta_K"]))
    eps_K4_new = _to_engineering_shear(epsK_new_t)

    return sig_new, deps_p, dlam, eps_K4_new, dlam


def integrate_visc(deps3, sig4, eps_K4, kappa, dt, E, nu, mat, kind):
    """One Nishihara backward-Euler step from the committed state.

    Canonical state-carrying entry (layouts identical to
    material.integrate_full; all tensors broadcast on leading
    dimensions; ``mat`` is a dict of scalars, params.NishiharaMC.theta)::

        integrate_visc(deps3, sig4, eps_K4, kappa, dt, E, nu, mat, kind)
        -> (sig4_new, deps_p4, dlam, eps_K4_new, dkappa)

    ``deps3`` is the plane-strain increment (deps_xx, deps_yy,
    dgamma_xy); ``eps_K4`` the committed (trace-free) Kelvin strain in
    (e_xx, e_yy, e_zz, gamma_xy) engineering-shear layout; ``deps_p4``
    the VISCOPLASTIC strain increment only (Kelvin/Maxwell strains are
    not plastic strain); ``dkappa = dlam`` (perfect-kind bookkeeping,
    tier-A meaning).  The committed inputs are never mutated (pure
    functional update -- trial purity).
    """
    if kind != "nishihara":
        raise ValueError(f"unknown viscous material kind {kind!r}")
    deps3 = _as_t(deps3)
    deps4 = torch.stack([deps3[..., 0], deps3[..., 1],
                         torch.zeros_like(deps3[..., 0]),
                         deps3[..., 2]], dim=-1)
    return integrate_visc4(deps4, sig4, eps_K4, kappa, dt, E, nu, mat)


def consistent_tangent_visc(deps3_qp, sig4_qp, eps_K4_qp, kappa_qp, dt,
                            E, nu, mat, kind):
    """Exact algorithmic tangent d(sig_inplane)/d(deps3), (nq, 3, 3).

    Mirrors material.consistent_tangent: reverse-mode Jacobian of the
    full update (viscoelastic condensation + regularized return) at
    FIXED committed (sig, eps_K, kappa) and fixed dt, vmapped over the
    leading QP batch.
    """
    idx = torch.tensor(IN_PLANE_IDX, dtype=torch.int64)

    def f(d, s, ek, k):
        return integrate_visc(d, s, ek, k, dt, E, nu, mat, kind)[0][idx]

    return torch.func.vmap(torch.func.jacrev(f, argnums=0))(
        _as_t(deps3_qp), _as_t(sig4_qp), _as_t(eps_K4_qp),
        _as_t(kappa_qp))
