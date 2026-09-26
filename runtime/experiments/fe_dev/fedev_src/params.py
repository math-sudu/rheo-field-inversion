"""Problem and material parameter containers (tension-positive frame).

Sign conventions (fixed by the trusted Kirsch reference and the frozen
prototype -- see ``proto_bridge.kirsch_ref``):

* Internal frame is TENSION-POSITIVE; x horizontal (direction of the
  horizontal in-situ stress sigma_h = K0*sigma_v), y vertical (direction
  of sigma_v), theta counter-clockwise from +x.
* In-situ inputs ``sigma_v`` (vertical) and ``K0`` are geomechanics
  COMPRESSION-POSITIVE magnitudes, exactly as in the prototype's
  ``Params``.  The uniform initial (geostatic) TOTAL stress in the
  tension-positive frame is

      s_xx0 = -K0*sigma_v,  s_yy0 = -sigma_v,  s_xy0 = 0,
      s_zz0 = -K0*sigma_v   (out-of-plane).

  Out-of-plane choice: both horizontal directions carry the same lateral
  geostatic stress (tunnel axis horizontal), i.e. sigma_z0 = K0*sigma_v
  compression-positive.  For K0 = 1 this is the fully hydrostatic state
  of the classic axisymmetric tunnel benchmarks.  This is a documented
  modeling choice of this package (the elastic prototype carries no
  out-of-plane state).

* Perfect plasticity (tier A): no hardening/softening.  The state and
  parameter containers keep an explicit internal-variable slot
  (``kappa``: cumulative plastic-multiplier measure) so a later
  softening extension extends rather than rewrites the API.

Angles (phi, psi) are stored in RADIANS.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BaseParams:
    """Geometry + elasticity + in-situ field (compression-positive inputs).

    Field names and defaults mirror the prototype's ``Params`` so the
    two parameter sets are interchangeable in equivalence tests.
    """

    a: float = 2.0
    sigma_v: float = 5.0
    K0: float = 0.5
    E: float = 1800.0
    nu: float = 0.30

    @property
    def G(self):
        return self.E / (2.0 * (1.0 + self.nu))

    @property
    def kappa(self):
        """Plane-strain Kolosov constant 3 - 4 nu (far-zone carrier)."""
        return 3.0 - 4.0 * self.nu

    @property
    def m(self):
        return 0.5 * (self.sigma_v + self.K0 * self.sigma_v)

    @property
    def d(self):
        return 0.5 * (self.sigma_v - self.K0 * self.sigma_v)

    def sigma0(self):
        """Tension-positive initial TOTAL stress (s_xx, s_yy, s_zz, s_xy)."""
        return np.array([
            -self.K0 * self.sigma_v,
            -self.sigma_v,
            -self.K0 * self.sigma_v,
            0.0,
        ])


@dataclass(frozen=True)
class Elastic:
    """No yield surface (plasticity inactive)."""

    kind: str = "elastic"


@dataclass(frozen=True)
class MohrCoulomb:
    """Perfect Mohr-Coulomb: cohesion c, friction phi, dilation psi (rad).

    Tension-positive yield on sorted principals (s1 >= s2 >= s3):
        f = 0.5*(s1 - s3) + 0.5*(s1 + s3)*sin(phi) - c*cos(phi)
    Non-associated flow through the same expression with psi (<= phi).
    """

    c: float
    phi: float
    psi: float
    kind: str = "mc"

    def theta(self):
        return {"c": self.c, "phi": self.phi, "psi": self.psi}


@dataclass(frozen=True)
class HoekBrown:
    """Generalized Hoek-Brown (perfect), tension-positive principal form.

    Compression-positive generalized HB
        sig1c = sig3c + sigma_ci*(m_b*sig3c/sigma_ci + s)**a_hb
    maps under the sign flip sig_c = -sig_t (tension-positive ordering
    s1 >= s2 >= s3, so s1 = -sig3c, s3 = -sig1c) to
        f = (s1 - s3) - sigma_ci*(s - m_b*s1/sigma_ci)**a_hb ,
    valid for s1 <= s*sigma_ci/m_b (hydrostatic tension apex).

    Flow rule (documented choice): Mohr-Coulomb-TYPE plastic potential
        g = 0.5*(s1 - s3) + 0.5*(s1 + s3)*sin(psi)
    with a constant dilation angle psi (non-associated; the constant-psi
    potential is the common engineering choice for HB media and keeps
    the out-of-plane plastic strain zero on in-plane faces, matching the
    axisymmetric benchmark construction).
    """

    sigma_ci: float
    m_b: float
    s: float
    a_hb: float
    psi: float
    kind: str = "hb"

    def theta(self):
        return {"sigma_ci": self.sigma_ci, "m_b": self.m_b,
                "s": self.s, "a_hb": self.a_hb, "psi_hb": self.psi}


@dataclass(frozen=True)
class CWFSMohrCoulomb:
    """CWFS strain-softening Mohr-Coulomb (tier B), tension-positive.

    Cohesion-weakening / friction-strengthening laws in the internal
    variable kappa (equivalent plastic shear strain, per increment
    d_kappa = deps_p_max - deps_p_min over the returned principal
    plastic-strain increments):

        c(kappa)   = c_peak + (c_res - c_peak) * min(kappa/gamma_c, 1)
        phi(kappa) = phi_peak + (phi_res - phi_peak) * min(kappa/gamma_phi, 1)

    Yield on sorted principals (s1 >= s2 >= s3), same 0.5-normalized
    form as perfect MC but with kappa-dependent strength:
        f = 0.5*(s1 - s3) + 0.5*(s1 + s3)*sin(phi(k)) - c(k)*cos(phi(k))
    Non-associated MC-type potential with CONSTANT dilation psi.

    Angles (phi_peak, phi_res, psi) in RADIANS; gamma_c, gamma_phi > 0
    (saturation strains of the piecewise-linear laws; the laws are
    nonsmooth at kappa = gamma_c / gamma_phi -- clamp subgradient,
    see material.py nonsmooth caveats).  Setting c_peak = c_res and
    phi_peak = phi_res degenerates to perfect Mohr-Coulomb.

    ``visc_H`` (stress units; default 0.0) is the OPTIONAL Perzyna
    regularization of the softening return (material.py module
    docstring, CWFS-Perzyna item; a THESIS-NEW construction -- the
    reference CWFS line carries no viscous regularization): the
    backward-Euler consistency f_j = visc_H * dlam_j at the fixed
    ratio visc_H = eta_vp/dt.  Under a lambda schedule this defines a
    FIXED regularized pseudo-time problem (viscoplastic continuation
    of the rate-independent softening problem), not a physical time
    integration.  visc_H = 0.0 is the exact rate-independent return
    bit-for-bit.  It is a solver-side REGULARIZATION SETTING, not a
    calibrated material parameter: deliberately NOT in :meth:`theta`
    (whose 7 keys are the differentiable material parameters of the
    replay/gradient contract); the FE system and ``diff.replay``
    thread it as a theta-independent constant.
    """

    c_peak: float
    c_res: float
    gamma_c: float
    phi_peak: float
    phi_res: float
    gamma_phi: float
    psi: float
    visc_H: float = 0.0
    kind: str = "cwfs"

    def theta(self):
        return {"c_peak": self.c_peak, "c_res": self.c_res,
                "gamma_c": self.gamma_c, "phi_peak": self.phi_peak,
                "phi_res": self.phi_res, "gamma_phi": self.gamma_phi,
                "psi": self.psi}


@dataclass(frozen=True)
class NishiharaMC:
    """Maxwell-extended (Burgers-type) Nishihara visco-elasto-plastic
    chain around perfect MC (tier C), tension-positive.

    Series chain (one total stress, strains add)::

        eps = C_e(E, nu)^-1 : sig  +  eps_K  +  eps_M  +  eps_vp

    * elastic spring: the problem's (E, nu) -- not stored here;
    * Kelvin element (G_K, eta_K):
          eps_K_dot = (s - 2 G_K eps_K) / (2 eta_K),   s = dev(sig)
    * Maxwell dashpot (eta_M):
          eps_M_dot = s / (2 eta_M)
    * Bingham/Perzyna viscoplastic element (eta_vp) on the perfect-MC
      surface with the tier-A non-associated MC-type potential
      (constant psi).

    NAMING (variant, not the classical model): the CLASSICAL Nishihara
    chain is instantaneous elastic + Kelvin + THRESHOLDED viscoplastic
    element in series -- it has NO free (unthresholded) Maxwell
    dashpot (the dtPINN-nishihara governing-equations convention:
    deviatoric split e = e_e + e_K + e_vp with Kelvin (G_2, eta_2) and
    a Macaulay-bracket viscoplastic element eta_3).  This container
    ADDS the free Maxwell dashpot eta_M, making the chain a Maxwell-
    extended (Burgers-type) Nishihara variant; the classical model is
    recovered exactly in the limit eta_M -> inf (the doc272 fits could
    accordingly only bound eta_M from BELOW -- consistent with that
    limit).  PARAMETER-MAPPING CAUTION: the classical/dtPINN
    convention's THRESHOLDED viscoplastic dashpot (eta_3 in that
    notation; eta_2 is its Kelvin dashpot = this container's eta_K)
    corresponds to THIS container's eta_vp, NEVER to eta_M (free
    Maxwell, unthresholded).

    DEVIATORIC-VISCOUS / ELASTIC-VOLUMETRIC split (standard engineering
    assumption, documented package choice): the Kelvin and Maxwell
    elements act on the deviatoric stress only, so the volumetric
    response stays purely elastic (bulk modulus of the E, nu spring);
    eps_K and eps_M are trace-free.  The Kelvin shear modulus is
    derived from the stored Young-type modulus by the isotropic
    conversion at the ELASTIC Poisson ratio,

        G_K = E_K / (2 (1 + nu)),

    (the Kelvin spring shares nu; its volumetric action is suppressed
    by the deviatoric-flow assumption).

    PERZYNA EQUIVALENCE: backward-Euler consistency of the overstress
    flow  eps_vp_dot = <f>/eta_vp * dg/dsig  is the rate-independent
    MC return with a linear regularization term (eta_vp/dt) * dlam
    added to every multiplier equation -- the analogue of a positive
    linear hardening modulus visc_H = eta_vp/dt (material._mc_sorted;
    integrator in material_visc).

    1D AXIAL CORRESPONDENCE used by the analytic anchors (constant
    axial sigma below the viscoplastic threshold; the doc272
    staged-creep fit identifies these 1D constants)::

        eps_axial_creep(t) = sigma/E_K_1D * (1 - exp(-t/tau_K))
                             + sigma * t / eta_M_1D

    with (derivation in tests/test_nishihara_material.py -- a purely
    deviatoric mechanism responds incompressibly in uniaxial loading,
    so its uniaxial modulus is 3x the shear value)::

        E_K_1D   = 3 G_K = 3 E_K / (2 (1 + nu))
        eta_M_1D = 3 eta_M
        tau_K    = eta_K / G_K          (frame-invariant)

    Above the threshold the axial creep rate gains
    <sigma - sigma_s_1D>/eta_vp_1D.  The face structure of that branch
    depends on the stress state, so TWO axial correspondences exist:

    * UNIAXIAL compression (the analytic gold anchors): the principal
      state (0, 0, -q) sits ON the triaxial corner, BOTH surfaces
      active (Koiter)::

          sigma_s_1D = 2 c cos(phi) / (1 - sin(phi)) = 2 c sqrt(N_phi)
          eta_vp_1D  = eta_vp / (0.5 (1 - sin phi) (1 - sin psi))

    * CONFINED axial loading at sigma_2 > sigma_3 > 0 (the doc272
      true-triaxial staged tests, sigma_2 = 10, sigma_3 = 5 MPa): the
      state just above threshold sits on ONE face only, so with
      N_phi = (1 + sin phi)/(1 - sin phi)::

          sigma_s_1D = N_phi sigma_3 + 2 c sqrt(N_phi)
          eta_vp_1D  = eta_vp / (0.25 (1 - sin phi) (1 - sin psi))

      (single-face rate is HALF the corner rate -- the same factor-2
      noted in tests/test_nishihara_material.py; the confining shift
      N_phi sigma_3 is constant within a stage and leaves the axial
      yield gradient unchanged).  The doc272 1D -> FE inversion
      (:class:`Doc272NishiharaGroup`) uses THIS correspondence.

    Angles (phi, psi) in RADIANS.  Stress/viscosity/time units are the
    caller's consistent set (e.g. MPa, MPa*h, h).  eta_K, eta_M,
    eta_vp > 0 finite; the rate-independent tier-A limit is recovered
    for eta_K, eta_M -> inf with eta_vp/dt -> 0.
    """

    c: float
    phi: float
    psi: float
    E_K: float
    eta_K: float
    eta_M: float
    eta_vp: float
    kind: str = "nishihara"

    def theta(self):
        return {"c": self.c, "phi": self.phi, "psi": self.psi,
                "E_K": self.E_K, "eta_K": self.eta_K,
                "eta_M": self.eta_M, "eta_vp": self.eta_vp}


CWFS_TIER_B = CWFSMohrCoulomb(
    c_peak=1.0, c_res=0.2, gamma_c=5.0e-3,
    phi_peak=float(np.deg2rad(30.0)), phi_res=float(np.deg2rad(40.0)),
    gamma_phi=2.5e-3, psi=float(np.deg2rad(10.0)))
"""Tier-B default CWFS set at the tier-A scale (BaseParams sigma_v=5,
E=1800, nu=0.30; c_peak/phi_peak/psi = the tier-A MC benchmark set,
c_res/c_peak = 0.2, phi_res = 40 deg).

Rationale (one line): the 2:1 saturation pair gamma_c = 5e-3,
gamma_phi = 2.5e-3 (friction strengthening saturating FIRST) is the
mild-transient choice that keeps the calibration probe's plastic front
mid-band under substantial softening -- equal-rate pairs let the
cohesion loss outrun the friction gain near the wall (low mean stress)
and push the front past 1.6a.

Probe result (K0 = 1, R_Gamma/a = 3, 8x32 condensed, uniform 5-knot
lambda schedule to 1; asserted by
tests/test_cwfs_system.py::test_tier_b_default_probe): plastic front
r_front = 1.4642a (band 1.2-1.6a), margin to Gamma = 1.5358a (no
breach), n_plastic = 384 QPs (all kappa > 1e-8), kappa_max = 1.77e-2
= 3.5 gamma_c = 7.1 gamma_phi (both laws saturated well inside the
zone), zero increment halvings.  NOTE the strongly-softening transient
makes Newton robustness schedule-sensitive (a denser 9-knot schedule
fails on this set by committing a partially-softened state and
reloading it); the probe schedule above is part of this default's
contract -- the coupled drivers' halving/rollback machinery is the
intended recovery path on other schedules.
"""


CWFS_SOFT_ROCK_DESIGN = CWFSMohrCoulomb(
    c_peak=1.0, c_res=0.2, gamma_c=1.0e-2,
    phi_peak=float(np.deg2rad(20.0)), phi_res=float(np.deg2rad(30.0)),
    gamma_phi=1.5e-2, psi=float(np.deg2rad(5.0)))
"""Convergent-core soft-rock DESIGN point of the reference CWFS line
(cwfs-inversion revision), verbatim:

* softening laws -- ``run_fe_coverage.py`` CORE family (c_peak = 1.0,
  phi 20 -> 30 deg, gamma_p_c = 0.01, gamma_p_phi = 0.015) at the
  design point c_res = 0.2;
* companion frame (NOT stored here; pass through BaseParams) --
  E = 15000 MPa, nu = 0.25, psi = 5 deg (the frozen SOFT_ROCK element
  anchor, ``e2_method_config.json`` frozen_element_params), design
  far-field P0 = 5.62 MPa at K0 = 1, tunnel radius a = 4.9 m.

Documented role: the reference line's MESH-CONVERGED softening
configuration (hydrostatic finest-pair wall residual 0.55 %, plastic
zone Rp/a ~ 2.0; localization onset sits at slope ~90-150 / P0
~8.5-12 MPa, beyond this point) -- the in-convergent-domain sanity
anchor of the CWFS x Perzyna regularization validation.  Relative
brittleness (c_peak - c_res)/gamma_c/E = 80/15000 = 0.0053 vs the
tier-B set's 160/1800 = 0.089; note CWFS_TIER_B also INVERTS the
saturation ordering (gamma_phi < gamma_c) relative to every reference
set."""


# ---------------------------------------------------------------------------
# doc272 tier-C creep calibration set (carbonaceous slate, staged
# true-triaxial creep at bedding angles beta = 0/30/90 deg).

DOC272_CONV_PHI = float(np.deg2rad(30.0))
DOC272_CONV_PSI = float(np.deg2rad(10.0))
"""Default MC angles of the doc272 1D -> FE conversion: a PLACEHOLDER
convention, NOT a doc272 measurement.

No verified friction/dilation angle for the doc272 slate exists on
disk (the source report's plastic-parameter table 3-3 remains
unverified OCR -- tier-C batch report section 7 item 3), so
``Doc272NishiharaGroup.nishihara`` defaults to the fe_dev tier-A demo
MC convention (phi = 30 deg, psi = 10 deg -- ``run_matrix_formal
.MC_TIER_A`` and the test-suite MC set).  ``c`` (inverted from
sigma_s) and ``eta_vp`` (Koiter corner factor) inherit this
convention; re-derive both before quoting them as doc272 material
properties under any other angle set.
"""

_CALIB_KINDS = ("identified", "weak", "lower_bound", "upper_bound")


@dataclass(frozen=True)
class CalibValue:
    """One calibrated constant with explicit identifiability semantics.

    ``kind``:

    * ``"identified"``  -- point estimate resolved by the fit; usable
      as a default.
    * ``"weak"``        -- point estimate the fit resolves only weakly
      (near-flat objective direction); usable as a default, flagged so
      consumers see the caveat.
    * ``"lower_bound"`` -- NO point estimate exists; the data constrain
      the quantity to ``>= value`` only.
    * ``"upper_bound"`` -- NO point estimate exists; the data constrain
      the quantity to ``<= value`` only.

    Bound kinds never default: consumers must pass an explicit scenario
    value, which is validated against ``value``.
    """

    value: float
    kind: str

    def __post_init__(self):
        if self.kind not in _CALIB_KINDS:
            raise ValueError(f"unknown identifiability kind {self.kind!r}"
                             f" (expected one of {_CALIB_KINDS})")
        if not (np.isfinite(self.value) and self.value > 0.0):
            raise ValueError("CalibValue.value must be positive finite, "
                             f"got {self.value!r}")


def _resolve_calib(key, name, fld, arg):
    """Default/scenario resolution honoring the identifiability kind.

    ``arg is None`` requests the default: allowed only for kinds
    ``"identified"``/``"weak"`` (a point estimate exists).  Bound-only
    fields and absent fields (``fld is None``) REQUIRE an explicit
    scenario value; explicit values must be positive finite and respect
    the recorded bound.
    """
    if arg is None:
        if fld is not None and fld.kind in ("identified", "weak"):
            return float(fld.value)
        if fld is None:
            raise ValueError(
                f"{key}.{name} is not identifiable from the doc272 fit "
                "(no point estimate, no bound recorded): pass an "
                "explicit scenario value")
        rel = ">=" if fld.kind == "lower_bound" else "<="
        raise ValueError(
            f"{key}.{name} carries a {fld.kind.replace('_', ' ')} only "
            f"({rel} {fld.value:g}): no point estimate exists; pass an "
            "explicit scenario value respecting the bound")
    arg = float(arg)
    if not (np.isfinite(arg) and arg > 0.0):
        raise ValueError(f"{key}.{name} scenario value must be positive "
                         f"finite, got {arg!r}")
    if fld is not None and fld.kind == "lower_bound" and arg < fld.value:
        raise ValueError(f"{key}.{name} scenario value {arg:g} violates "
                         f"the recorded lower bound {fld.value:g}")
    if fld is not None and fld.kind == "upper_bound" and arg > fld.value:
        raise ValueError(f"{key}.{name} scenario value {arg:g} violates "
                         f"the recorded upper bound {fld.value:g}")
    return arg


@dataclass(frozen=True)
class Doc272NishiharaGroup:
    """One doc272 creep-calibrated group: the fitted 1D Nishihara
    constants with their identifiability structure, plus the exact
    1D -> FE inversion producing a :class:`NishiharaMC`.

    SOURCE (single fitted-number source; values stored verbatim):
    ``results/fe_dev/doc272_creep_fit/nishihara_params.csv``
    (``fit_creep_doc272.py`` on the frozen snapshot
    ``data/doc272_creep``; research_archive doc272 section 3.3 staged
    true-triaxial creep, carbonaceous slate; batch report
    ``results/fe_dev/2026-07-11_exec7_fe_tierc_creep_fit_report.md``).
    Fields hold the UNIAXIAL (1D) frame constants exactly as fitted, in
    the CSV units (GPa / GPa*h / MPa / h, per the field suffixes).

    IDENTIFIABILITY (25 h stage window x ~0.005 % digitisation
    resolution; carried as :class:`CalibValue` kinds, never as invented
    point estimates):

    * ``sigma_s_1D_MPa`` -- identified for group A ONLY (66 MPa; a
      weak-signal point estimate -- the viscoplastic steady term sits
      just above the digitisation floor -- cross-checked by
      sigma_s/sigma_p = 0.57 inside the [0.5, 0.8] empirical band of
      doc272 table 3-13).  Groups B/C carry UPPER BOUNDS only (the
      failure-stage stress, 52 / 166 MPa; sub-resolution activation at
      lower stages cannot be excluded).
    * ``eta_M_1D_GPa_h`` -- lower bounds only for A (>= 5.1e4 GPa*h)
      and C (>= 3.2e4 GPa*h): the fitted steady term is below the
      digitisation floor, so the hold window constrains eta_M from
      below only (long-term constraint pre-declared to field
      back-analysis).  Group B resolves a weakly-constrained point
      (1.184e4 GPa*h, kind ``"weak"``).
    * ``eta_vp_1D_GPa_h`` -- identified for group A only (paired with
      its sigma_s signal); ``None`` for B/C (not identifiable; no bound
      recorded).

    ``E0_GPa`` is the loading-branch low-stress secant with inter-stage
    creep embedded -> a systematic UNDERESTIMATE, kept for magnitude
    reference only; the FE elastic tier keeps the problem's own (E, nu)
    (batch report section 3 footnote).

    The 1D -> FE inversion (:meth:`nishihara`) is the exact inverse of
    the CONFINED-axial correspondence at the recorded test confinement
    ``sigma3_MPa`` (see the :class:`NishiharaMC` docstring; viscous
    branch derivation anchored in ``tests/test_nishihara_material.py``),
    with N_phi = (1 + sin phi)/(1 - sin phi)::

        G_K    = E_K_1D / 3      (stored E_K = 2 (1 + nu) G_K)
        eta_K  = eta_K_1D / 3
        eta_M  = eta_M_1D / 3
        c      = (sigma_s_1D - N_phi sigma_3) / (2 sqrt(N_phi))
        eta_vp = eta_vp_1D * 0.25 (1 - sin phi) (1 - sin psi)

    CORRECTION (2026-07-14, review-r1 fix A1; committee finding): the
    original inversion used the UNIAXIAL corner correspondence
    (c = sigma_s_1D (1 - sin phi)/(2 cos phi), Koiter factor 0.5),
    i.e. it read the fitted stage-axial threshold as an unconfined
    uniaxial strength.  The doc272 staged creep tests are TRUE
    TRIAXIAL at sigma_2 = 10, sigma_3 = 5 MPa, so sigma_s_1D is the
    axial threshold AT sigma_3 = 5 MPa: the Mohr-Coulomb zero-
    overstress condition at the test state gives the confined form
    above (group A: c 19.05 -> 14.72 MPa), and just above threshold
    exactly ONE face is active (sigma_2 strictly intermediate), so the
    flow factor is the single-face 0.25, not the corner 0.5.  Caveat
    recorded with the derivation: under the converted set the second
    face activates at sigma_1 >= N_phi sigma_2 + 2 c sqrt(N_phi)
    (group A convention: 81 MPa), which the two highest fitted stages
    (86/101 MPa) straddle; the single-slope empirical Bingham fit does
    not resolve that doubling (weak-signal regime), and the
    correspondence maps the threshold and the at-threshold slope.

    The default conversion angles are the ``DOC272_CONV_PHI/PSI``
    placeholder convention (see its docstring), NOT doc272
    measurements.
    """

    key: str
    beta_deg: float
    E0_GPa: float
    E_K_1D_GPa: float
    eta_K_1D_GPa_h: float
    tau_K_h: float
    eta_M_1D_GPa_h: CalibValue
    sigma_s_1D_MPa: CalibValue
    eta_vp_1D_GPa_h: CalibValue | None
    r2: float
    sigma3_MPa: float = 5.0

    def nishihara(self, nu, phi=DOC272_CONV_PHI, psi=DOC272_CONV_PSI, *,
                  sigma_s_1D_MPa=None, eta_vp_1D_GPa_h=None,
                  eta_M_1D_GPa_h=None):
        """FE-frame :class:`NishiharaMC` in (MPa, MPa*h, h) units.

        ``nu`` must be the SAME elastic Poisson ratio the integrator
        will receive: the stored ``E_K = 2 (1 + nu) E_K_1D / 3``
        encodes ``G_K = E_K_1D / 3`` through the container convention
        ``G_K = E_K / (2 (1 + nu))``.  Keyword scenario values close
        the fields whose kind is a bound (or that are ``None``); they
        are validated against the recorded bounds and REQUIRED there --
        this entry never invents point estimates for non-identified
        constants (see :func:`_resolve_calib`).

        The threshold/viscosity inversion is the CONFINED-axial
        correspondence at the stored test confinement ``sigma3_MPa``
        (class docstring, 2026-07-14 correction note): the fitted
        stage-axial threshold sigma_s_1D is converted through the MC
        zero-overstress condition at (sigma_1 = sigma_s_1D,
        sigma_3 = sigma3_MPa), and eta_vp through the single-face flow
        factor.  Raises when the scenario threshold does not exceed
        N_phi * sigma3_MPa (non-positive cohesion).
        """
        sig_s = _resolve_calib(self.key, "sigma_s_1D_MPa",
                               self.sigma_s_1D_MPa, sigma_s_1D_MPa)
        eta_vp_1d = _resolve_calib(self.key, "eta_vp_1D_GPa_h",
                                   self.eta_vp_1D_GPa_h, eta_vp_1D_GPa_h)
        eta_m_1d = _resolve_calib(self.key, "eta_M_1D_GPa_h",
                                  self.eta_M_1D_GPa_h, eta_M_1D_GPa_h)
        sphi = np.sin(phi)
        spsi = np.sin(psi)
        n_phi = (1.0 + sphi) / (1.0 - sphi)
        if sig_s <= n_phi * self.sigma3_MPa:
            raise ValueError(
                f"{self.key}.sigma_s_1D_MPa = {sig_s:g} MPa does not "
                f"exceed N_phi*sigma_3 = {n_phi * self.sigma3_MPa:g} "
                "MPa at the recorded test confinement: the confined "
                "MC inversion would give a non-positive cohesion")
        G_K = self.E_K_1D_GPa * 1.0e3 / 3.0                    # MPa
        return NishiharaMC(
            c=float((sig_s - n_phi * self.sigma3_MPa)
                    / (2.0 * np.sqrt(n_phi))),
            phi=float(phi), psi=float(psi),
            E_K=float(2.0 * (1.0 + nu) * G_K),
            eta_K=float(self.eta_K_1D_GPa_h * 1.0e3 / 3.0),
            eta_M=float(eta_m_1d * 1.0e3 / 3.0),
            eta_vp=float(eta_vp_1d * 1.0e3
                         * 0.25 * (1.0 - sphi) * (1.0 - spsi)))


DOC272_GROUP_A = Doc272NishiharaGroup(
    key="A_beta0", beta_deg=0.0, E0_GPa=25.25,
    E_K_1D_GPa=36.504, eta_K_1D_GPa_h=0.9453, tau_K_h=0.0259,
    eta_M_1D_GPa_h=CalibValue(5.1e4, "lower_bound"),
    sigma_s_1D_MPa=CalibValue(66.0, "identified"),
    eta_vp_1D_GPa_h=CalibValue(1.347e4, "identified"),
    r2=0.9983, sigma3_MPa=5.0)
"""beta = 0 deg (7 fitted levels, R2 = 0.9983, RMSE 0.0024 %): the one
group with sigma_s/eta_vp identified (weak-signal point estimates);
eta_M carries its lower bound only (>= 5.1e4 GPa*h = 5.1e7 MPa*h).
sigma3_MPa = 5.0: test-plan common minimum principal stress of the
doc272 staged true-triaxial programme (source table 3-4, original-PDF
verified 2026-07-14; sigma_2 = 10 MPa for the three fitted groups)."""

DOC272_GROUP_B = Doc272NishiharaGroup(
    key="B_beta30", beta_deg=30.0, E0_GPa=7.85,
    E_K_1D_GPa=12.719, eta_K_1D_GPa_h=0.7525, tau_K_h=0.0592,
    eta_M_1D_GPa_h=CalibValue(1.184e4, "weak"),
    sigma_s_1D_MPa=CalibValue(52.0, "upper_bound"),
    eta_vp_1D_GPa_h=None,
    r2=0.9953, sigma3_MPa=5.0)
"""beta = 30 deg (4 fitted levels, R2 = 0.9953, RMSE 0.0063 %): the
weakest direction; sigma_s <= 52 MPa (failure-stage upper bound),
eta_vp not identifiable, eta_M a weakly-constrained point."""

DOC272_GROUP_C = Doc272NishiharaGroup(
    key="C_beta90", beta_deg=90.0, E0_GPa=5.87,
    E_K_1D_GPa=19.316, eta_K_1D_GPa_h=6.3177, tau_K_h=0.3271,
    eta_M_1D_GPa_h=CalibValue(3.2e4, "lower_bound"),
    sigma_s_1D_MPa=CalibValue(166.0, "upper_bound"),
    eta_vp_1D_GPa_h=None,
    r2=0.9570, sigma3_MPa=5.0)
"""beta = 90 deg (2 fitted levels, R2 = 0.9570, RMSE 0.0281 %):
sigma_s <= 166 MPa (failure-stage upper bound), eta_vp not
identifiable, eta_M >= 3.2e4 GPa*h (= 3.2e7 MPa*h) only."""

DOC272_GROUPS = (DOC272_GROUP_A, DOC272_GROUP_B, DOC272_GROUP_C)
"""doc272 calibration set in bedding-angle order (beta = 0/30/90 deg)."""
