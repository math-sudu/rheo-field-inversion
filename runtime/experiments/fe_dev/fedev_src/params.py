
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


CWFS_SOFT_ROCK_DESIGN = CWFSMohrCoulomb(
    c_peak=1.0, c_res=0.2, gamma_c=1.0e-2,
    phi_peak=float(np.deg2rad(20.0)), phi_res=float(np.deg2rad(30.0)),
    gamma_phi=1.5e-2, psi=float(np.deg2rad(5.0)))


# ---------------------------------------------------------------------------

# true-triaxial creep at bedding angles beta = 0/30/90 deg).

SLATE_CONV_PHI = float(np.deg2rad(30.0))
SLATE_CONV_PSI = float(np.deg2rad(10.0))

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
                f"{key}.{name} is not identifiable from the slate fit "
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
class SlateNishiharaGroup:

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

    def nishihara(self, nu, phi=SLATE_CONV_PHI, psi=SLATE_CONV_PSI, *,
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


SLATE_GROUP_A = SlateNishiharaGroup(
    key="A_beta0", beta_deg=0.0, E0_GPa=25.25,
    E_K_1D_GPa=36.504, eta_K_1D_GPa_h=0.9453, tau_K_h=0.0259,
    eta_M_1D_GPa_h=CalibValue(5.1e4, "lower_bound"),
    sigma_s_1D_MPa=CalibValue(66.0, "identified"),
    eta_vp_1D_GPa_h=CalibValue(1.347e4, "identified"),
    r2=0.9983, sigma3_MPa=5.0)

SLATE_GROUP_B = SlateNishiharaGroup(
    key="B_beta30", beta_deg=30.0, E0_GPa=7.85,
    E_K_1D_GPa=12.719, eta_K_1D_GPa_h=0.7525, tau_K_h=0.0592,
    eta_M_1D_GPa_h=CalibValue(1.184e4, "weak"),
    sigma_s_1D_MPa=CalibValue(52.0, "upper_bound"),
    eta_vp_1D_GPa_h=None,
    r2=0.9953, sigma3_MPa=5.0)
"""beta = 30 deg (4 fitted levels, R2 = 0.9953, RMSE 0.0063 %): the
weakest direction; sigma_s <= 52 MPa (failure-stage upper bound),
eta_vp not identifiable, eta_M a weakly-constrained point."""

SLATE_GROUP_C = SlateNishiharaGroup(
    key="C_beta90", beta_deg=90.0, E0_GPa=5.87,
    E_K_1D_GPa=19.316, eta_K_1D_GPa_h=6.3177, tau_K_h=0.3271,
    eta_M_1D_GPa_h=CalibValue(3.2e4, "lower_bound"),
    sigma_s_1D_MPa=CalibValue(166.0, "upper_bound"),
    eta_vp_1D_GPa_h=None,
    r2=0.9570, sigma3_MPa=5.0)
"""beta = 90 deg (2 fitted levels, R2 = 0.9570, RMSE 0.0281 %):
sigma_s <= 166 MPa (failure-stage upper bound), eta_vp not
identifiable, eta_M >= 3.2e4 GPa*h (= 3.2e7 MPa*h) only."""

SLATE_GROUPS = (SLATE_GROUP_A, SLATE_GROUP_B, SLATE_GROUP_C)
