"""General-Laurent-map exterior KM representation.

Generalization of the frozen elliptic module
``experiments/seed_skeleton_km/mapped_net.py`` (map omega(zeta) =
R (zeta + m / zeta)) -- exactly the extension its "narrow mapping
interface" docstring anticipates -- to the full truncated exterior
conformal map of the source family (series mapping of an arbitrarily
shaped hole exterior onto |zeta| >= 1, Eq. (9) of the anisotropic
deep-tunnel paper)::

  z = omega(zeta) = R (zeta + sum_{j=0..n_map} c_j zeta^{-j}),
  |zeta| >= 1,

with the COMPLEX coefficient vector c = (c_0, .., c_{n_map}) GIVEN
data, never solved for (c_0 is a pure translation term).  The
traction-free hole under biaxial in-situ stress (sigma_v, sigma_h =
K0 * sigma_v, compression positive) is solved with the classical
collocation least-squares channel :func:`solve_ls` -- the physics of
``mapped_net.py`` verbatim, with only the map generalized.

Representation (tension-positive internal frame, family convention)
--------------------------------------------------------------------
With zeta the exterior mapped variable (|zeta| >= 1) and the GIVEN
map z = omega(zeta) above::

  phi(zeta) = Gamma  * omega(zeta) + sum_{n=1..N} a_n zeta^{-n}
  psi(zeta) = Gamma' * omega(zeta) + sum_{n=1..N} b_n zeta^{-n}

  Gamma  = -(sigma_v + sigma_h) / 4,
  Gamma' = -(sigma_v - sigma_h) / 2.

The far-field carriers Gamma, Gamma' are ANALYTICALLY CARRIED data
(never unknowns), composed with omega so they synthesize exactly the
undisturbed in-situ field in the physical plane: Phi_carrier =
(Gamma omega') / omega' = Gamma identically -- the division cancels
algebraically, so the code carries the carriers as the constant pair
(Phi, Psi) = (Gamma, Gamma') with Phi' = 0, the closed form of the
composition, exact at any point for ANY map (the same argument as
``mapped_net``, which nowhere used the elliptic form).  The complex
coefficients (a_n, b_n) are the ONLY unknowns.

Mapping = constructor data ("change the map, the potential layer
stands still"): the coefficients (R, c) are GIVEN, never solved for.
c = [0] reproduces the circular case (omega = R zeta, junction with
``refsol/kirsch.py``), and c = [0, m] reproduces
``mapped_net.MappedLaurentKM`` exactly (same map values, same
boundary system, same solutions -- the ellipse junction test).

Narrow mapping interface: the map form lives ONLY in the helper
methods :meth:`GeneralMappedKM.omega`,
:meth:`GeneralMappedKM.omega_prime` and
:meth:`GeneralMappedKM.omega_second` (plus the non-degeneracy
readouts :meth:`GeneralMappedKM.delta_measured` and
:meth:`GeneralMappedKM.delta_lower_bound`, and the map-analysis gate
:func:`validate_univalence`, which probes the map through those same
methods).  The potential layer, the boundary system and the solve
channel are untouched by the generalization, by construction.

No log channel: this case's traction release is self-equilibrated --
the resultant of the in-situ traction over the closed hole boundary
is zero (divergence theorem on the uniform field), so per the
family's O2 discipline the log(zeta) pair must vanish; this
representation deliberately does not carry one (the explicit-switch
machinery lives in the seed's ``laurent_net.LaurentKM``, where the
obligation is asserted both ways).

Synthesis (closed-form zeta-derivatives; no autodiff)
-----------------------------------------------------
Cartesian components via the mapped Kolosov-Muskhelishvili relations
(kappa = 3 - 4 nu, G = E / (2 (1 + nu)))::

  Phi(z)  = phi'(zeta) / omega'(zeta),
  Psi(z)  = psi'(zeta) / omega'(zeta),
  Phi'(z) = [phi''(zeta) omega'(zeta) - phi'(zeta) omega''(zeta)]
            / omega'(zeta)^3,

  sigma_xx + sigma_yy                = 4 Re Phi,
  sigma_yy - sigma_xx + 2 i sigma_xy = 2 [conj(z) Phi' + Psi],
  2 G (u_x + i u_y)                  = kappa phi - z conj(Phi)
                                       - conj(psi),

with the displacement evaluated on the decaying (excavation) parts
alone -- the carriers hold the unbounded in-situ straining -- exactly
as the seed's ``refsol/elliptic.py`` and ``mapped_net.py`` do.

Frames and sign conversions (all conversions happen HERE)
---------------------------------------------------------
Internally everything lives in the tension-positive elasticity frame;
the public API returns COMPRESSION-POSITIVE Cartesian stresses and
EXCAVATION-INDUCED, globally sign-flipped Cartesian displacements,
mirroring the API shapes of ``mapped_net.py`` / ``refsol/elliptic.py``
so the fields are directly comparable.  Cartesian (not polar)
components because the mapped-polar address (rho, theta_zeta) is not
a polar frame of the physical plane for a general map; the circular
junction test rotates frames where it compares with the polar
circular reference.

Boundary system and scaling
---------------------------
Traction-free wall: the physical traction of the TOTAL field on
|zeta| = 1 vanishes, t = sigma . n = 0, with the outward unit normal
n = zeta omega'(zeta) / |zeta omega'(zeta)| (the image of the radial
direction; exactly the normal of ``refsol.elliptic`` and
``mapped_net``).  Both scalar components at M collocation angles are
R-linear in the real coefficient vector, so
:meth:`GeneralMappedKM.boundary_system` assembles G x = h ONCE --
columns by unit-coefficient synthesis through the SAME public
traction path used for evaluation, carrier (in-situ) traction on the
right-hand side -- the single physics source of truth.  Scaling: the
unknowns are scaled by ``coef_scale = sigma_v * R`` (uniform column
scaling: the least-squares / minimum-norm solution is invariant up
to the factor and the scaled unknowns are O(1), the family
convention with the mapping radius R in the role of the hole
radius).  No per-column equilibration: on the wall the potential
basis terms are trigonometric (zeta^{-n} = e^{-i n theta}), which
keeps the collocation system well-conditioned; the lstsq-reported
rank is returned for audit.

Univalence (map acceptance, not solved-field physics)
-----------------------------------------------------
A GIVEN coefficient vector c need not define a one-to-one exterior
map; two readouts and one gate quantify this:

* ``delta_measured``: min |omega'| over a dense |zeta| = 1 scan
  (map-form-agnostic, mirrors ``mapped_net.delta_measured``).
* ``delta_lower_bound``: the closed-form convexity bound (chapter
  dossier derivation D4)

    delta_0-bound = R (1 - sum_{j>=1} j |c_j|):

  the single convex constraint sum_{j>=1} j |c_j| <= 1 - delta_0
  guarantees BOTH global exterior univalence AND
  inf_{|zeta| >= 1} |omega'| >= R delta_0.  The bound is SUFFICIENT,
  NOT NECESSARY: maps violating it (the returned value may even be
  negative; it is reported as-is) can still be univalent, so a
  negative or small bound routes the map to the numerical gate below
  rather than rejecting it.
* :func:`validate_univalence`: numerical acceptance of one map.
  Rationale: for an exterior map with omega(infinity) = infinity and
  omega' nonvanishing on |zeta| >= 1, a SIMPLE boundary image implies
  exterior univalence (argument principle: the image of |zeta| = rho
  winds once around every point it encloses, so point preimages are
  unique).  The gate therefore checks (i) omega' nonvanishing on
  dense boundary and annulus scans, (ii) the analytic far guard --
  for |zeta| >= rho_scan, |sum_{j>=1} j c_j zeta^{-(j+1)}| <=
  (sum_{j>=1} j |c_j|) / rho_scan^2 < 1, so omega' cannot vanish
  beyond the scanned annulus -- and (iii) simplicity of the closed
  boundary polygon.  Scan-resolution caveat: (i) and (iii) certify
  the sampled polygon/grid, not the continuum; resolutions are
  parameters.

Pure numpy + stdlib at module level (no torch anywhere in this
module).  The frozen metric implementations ``eval_v1.rel_l2`` /
``eval_v1.rel_max`` are REUSED by import from the frozen seed tree
``experiments/seed_skeleton_km`` (a flat module directory; a small
sys.path shim below mirrors the seed's own flat-import mechanism).
"""

import os
import sys
import time

import numpy as np

# Data-freeze import shim: the frozen seed tree is a flat module
# directory (not a package); mirror its flat-import mechanism so the
# frozen ``eval_v1`` is importable from this sibling experiment.
_HERE = os.path.dirname(os.path.abspath(__file__))
_SEED = os.path.normpath(os.path.join(_HERE, os.pardir,
                                      "seed_skeleton_km"))
if _SEED not in sys.path:
    sys.path.insert(0, _SEED)

import eval_v1  # noqa: E402  (frozen rel_l2 / rel_max, reused not rewritten)

__all__ = [
    "GeneralMappedKM",
    "solve_ls",
    "validate_univalence",
    "evaluate_vs_reference",
]

# Accept rho >= 1 up to a small relative rounding slack (as the seed).
_RHO_MIN = 1.0 - 1e-9


def _uniform_wall_angles(n_collocation):
    """Uniform collocation angles theta_j = 2 pi j / M, j = 0 .. M-1."""
    if n_collocation < 1:
        raise ValueError("n_collocation must be >= 1")
    return (2.0 * np.pi / n_collocation) * np.arange(n_collocation)


class GeneralMappedKM:
    """General-Laurent-map exterior KM representation for one case.

    Holds the case data (R, c, sigma_v, K0) and the truncation order
    N; the complex coefficient arrays are passed per call, so the
    same instance serves the solve channel and the tests.  The
    mapping coefficients (R, c) are GIVEN data, never unknowns; only
    (a_n, b_n) are solved for.

    Parameters
    ----------
    R : float
        Mapping radius (> 0) of omega(zeta) = R (zeta +
        sum_{j=0..n_map} c_j zeta^{-j}); it also sets the coefficient
        scale ``coef_scale = sigma_v * R``.
    c : array_like, complex, shape (n_map + 1,)
        Map coefficients; c[j] multiplies zeta^{-j} (c[0] is a pure
        translation).  c = [0] is the circle of radius R; c = [0, m]
        is the elliptic standard form of ``mapped_net``.  No
        univalence is enforced here -- gate candidate maps through
        :func:`validate_univalence` first.
    sigma_v : float
        Vertical in-situ stress (compression positive, nonzero).
    K0 : float
        Lateral pressure coefficient; sigma_h = K0 * sigma_v.
    n_order : int
        Truncation order N >= 1 of the decaying Laurent sums.

    Attributes
    ----------
    gamma, gamma_p : float
        Carried far-field coefficients Gamma = -(sigma_v + sigma_h)/4
        and Gamma' = -(sigma_v - sigma_h)/2 (tension-positive frame).
    coef_scale : float
        Documented coefficient scale sigma_v * R.
    n_map : int
        Highest inverse power of the map (c has n_map + 1 entries).
    """

    def __init__(self, R, c, sigma_v, K0, n_order):
        if not R > 0.0:
            raise ValueError("mapping radius R must be positive")
        c = np.array(c, dtype=complex, copy=True)
        if c.ndim != 1 or c.size < 1:
            raise ValueError(
                "c must be a 1-D array (c_0, .., c_{n_map}) with at "
                f"least the translation entry c_0, got shape {c.shape}"
            )
        if sigma_v == 0.0:
            raise ValueError(
                "sigma_v must be nonzero (it sets the coefficient scale)"
            )
        n_order = int(n_order)
        if n_order < 1:
            raise ValueError("n_order must be >= 1")
        self.R = float(R)
        self.c = c
        self.n_map = c.size - 1
        self.sigma_v = float(sigma_v)
        self.K0 = float(K0)
        self.n_order = n_order
        sigma_h = self.K0 * self.sigma_v
        self.sigma_h = sigma_h
        # Analytically carried far-field coefficients (data, never
        # unknowns), composed with omega in the representation.
        self.gamma = -0.25 * (self.sigma_v + sigma_h)
        self.gamma_p = -0.5 * (self.sigma_v - sigma_h)
        self.coef_scale = self.sigma_v * self.R

    # ------------------------------------------------------------------
    # mapping interface (the ONLY place the map form lives; swapping
    # the map touches these helpers and nothing else)
    # ------------------------------------------------------------------

    def omega(self, zeta):
        """z = omega(zeta) = R (zeta + sum_{j=0..n_map} c_j zeta^{-j})."""
        zeta = np.asarray(zeta, dtype=complex)
        inv = 1.0 / zeta
        s = np.zeros(zeta.shape, dtype=complex)
        p = np.ones(zeta.shape, dtype=complex)  # zeta^{-j} at j = 0
        for j in range(self.n_map + 1):
            s = s + self.c[j] * p
            p = p * inv
        return self.R * (zeta + s)

    def omega_prime(self, zeta):
        """omega'(zeta) = R (1 - sum_{j=1..n_map} j c_j zeta^{-(j+1)})."""
        zeta = np.asarray(zeta, dtype=complex)
        inv = 1.0 / zeta
        s = np.zeros(zeta.shape, dtype=complex)
        p = inv * inv  # zeta^{-(j+1)} at j = 1
        for j in range(1, self.n_map + 1):
            s = s + (j * self.c[j]) * p
            p = p * inv
        return self.R * (1.0 - s)

    def omega_second(self, zeta):
        """omega''(zeta) = R sum_{j=1..n_map} j (j+1) c_j zeta^{-(j+2)}."""
        zeta = np.asarray(zeta, dtype=complex)
        inv = 1.0 / zeta
        s = np.zeros(zeta.shape, dtype=complex)
        p = inv * inv * inv  # zeta^{-(j+2)} at j = 1
        for j in range(1, self.n_map + 1):
            s = s + (j * (j + 1) * self.c[j]) * p
            p = p * inv
        return self.R * s

    def delta_measured(self, n_scan=4096):
        """Measured delta: min |omega'| over a dense |zeta| = 1 scan.

        Map-form-agnostic non-degeneracy readout (uniform scan zeta =
        exp(2 pi i k / n_scan)), mirroring
        ``mapped_net.delta_measured``; for c = [0, m] it reproduces
        the elliptic closed form R (1 - m) because the scan grid
        contains the minimizing angles 0 and pi exactly.
        """
        if n_scan < 1:
            raise ValueError("n_scan must be >= 1")
        theta = (2.0 * np.pi / n_scan) * np.arange(n_scan)
        return float(np.min(np.abs(self.omega_prime(np.exp(1j * theta)))))

    def delta_lower_bound(self):
        """Closed-form convexity bound R (1 - sum_{j>=1} j |c_j|).

        Chapter-dossier derivation D4: the single convex constraint
        sum_{j>=1} j |c_j| <= 1 - delta_0 guarantees global exterior
        univalence AND inf_{|zeta| >= 1} |omega'| >= R delta_0.  The
        bound is SUFFICIENT, NOT NECESSARY: it may be negative (it is
        reported as-is), and maps violating it can still be univalent
        -- gate those through :func:`validate_univalence` instead of
        rejecting them here.
        """
        j = np.arange(1, self.n_map + 1)
        return self.R * (1.0 - float(np.sum(j * np.abs(self.c[1:]))))

    # ------------------------------------------------------------------
    # coefficient bookkeeping (mirrors mapped_net.MappedLaurentKM)
    # ------------------------------------------------------------------

    def _as_coefficients(self, coef_a, coef_b):
        """Validate and cast one coefficient pair to complex (N,)."""
        out = []
        for name, c in (("coef_a", coef_a), ("coef_b", coef_b)):
            c = np.asarray(c, dtype=complex)
            if c.shape != (self.n_order,):
                raise ValueError(
                    f"{name} must have shape ({self.n_order},), "
                    f"got {c.shape}"
                )
            out.append(c)
        return out[0], out[1]

    def coefficients_to_vector(self, coef_a, coef_b):
        """Pack (coef_a, coef_b) into the real unknown vector.

        Ordering (fixed, family-shared)::

          x = [Re a_1..a_N | Im a_1..a_N | Re b_1..b_N | Im b_1..b_N]
        """
        coef_a, coef_b = self._as_coefficients(coef_a, coef_b)
        return np.concatenate(
            [coef_a.real, coef_a.imag, coef_b.real, coef_b.imag]
        )

    def vector_to_coefficients(self, x):
        """Inverse of :meth:`coefficients_to_vector`."""
        x = np.asarray(x, dtype=float)
        n = self.n_order
        if x.shape != (4 * n,):
            raise ValueError(f"x must have shape ({4 * n},), got {x.shape}")
        coef_a = x[0:n] + 1j * x[n:2 * n]
        coef_b = x[2 * n:3 * n] + 1j * x[3 * n:4 * n]
        return coef_a, coef_b

    # ------------------------------------------------------------------
    # synthesis (closed-form derivatives; tension-positive internally)
    # ------------------------------------------------------------------

    def _prepare(self, rho, theta_zeta, check_rho=True):
        """Broadcast (rho, theta_zeta) and validate rho >= 1."""
        rho, theta_zeta = np.broadcast_arrays(
            np.asarray(rho, dtype=float),
            np.asarray(theta_zeta, dtype=float),
        )
        if check_rho and np.any(rho < _RHO_MIN):
            raise ValueError(
                "mapped exterior representation is defined for rho >= 1"
                " only"
            )
        return rho, theta_zeta

    def _field_terms(self, zeta, coef_a, coef_b, carriers, decaying):
        """Phi(z), Phi'(z), Psi(z) at zeta (tension-positive frame).

        Carrier contribution (closed form of the composition): with
        phi_c = Gamma omega(zeta), Phi_c = Gamma omega' / omega' =
        Gamma EXACTLY -- the division cancels algebraically -- so the
        carriers enter as the constants (Phi, Psi) = (Gamma, Gamma')
        with Phi' = 0; they synthesize the uniform in-situ field
        exactly at any point, for ANY map.

        Decaying contribution (closed-form zeta-derivatives)::

          phi_0'(zeta)  = -sum n a_n zeta^{-(n+1)},
          phi_0''(zeta) =  sum n (n+1) a_n zeta^{-(n+2)},
          psi_0'(zeta)  = -sum n b_n zeta^{-(n+1)},

          Phi_0  = phi_0' / omega',
          Phi_0' = (phi_0'' omega' - phi_0' omega'') / omega'^3,
          Psi_0  = psi_0' / omega'.
        """
        Phi = np.zeros(zeta.shape, dtype=complex)
        Phi_p = np.zeros(zeta.shape, dtype=complex)
        Psi = np.zeros(zeta.shape, dtype=complex)
        if carriers:
            Phi += self.gamma
            Psi += self.gamma_p
        if decaying:
            inv = 1.0 / zeta
            dphi0 = np.zeros(zeta.shape, dtype=complex)
            ddphi0 = np.zeros(zeta.shape, dtype=complex)
            dpsi0 = np.zeros(zeta.shape, dtype=complex)
            p = inv * inv  # zeta^{-(n+1)} at n = 1
            for n in range(1, self.n_order + 1):
                dphi0 -= n * coef_a[n - 1] * p
                dpsi0 -= n * coef_b[n - 1] * p
                ddphi0 += n * (n + 1) * coef_a[n - 1] * p * inv
                p = p * inv
            wp = self.omega_prime(zeta)
            wpp = self.omega_second(zeta)
            Phi += dphi0 / wp
            Phi_p += (ddphi0 * wp - dphi0 * wpp) / (wp * wp * wp)
            Psi += dpsi0 / wp
        return Phi, Phi_p, Psi

    def _decaying_potentials(self, zeta, coef_a, coef_b):
        """phi_0, psi_0, phi_0' at zeta (tension-positive frame)."""
        phi0 = np.zeros(zeta.shape, dtype=complex)
        psi0 = np.zeros(zeta.shape, dtype=complex)
        dphi0 = np.zeros(zeta.shape, dtype=complex)
        inv = 1.0 / zeta
        p = inv  # zeta^{-n} at n = 1
        for n in range(1, self.n_order + 1):
            phi0 += coef_a[n - 1] * p
            psi0 += coef_b[n - 1] * p
            dphi0 -= n * coef_a[n - 1] * p * inv
            p = p * inv
        return phi0, psi0, dphi0

    def stresses_cartesian(self, rho, theta_zeta, coef_a, coef_b,
                           part="total"):
        """Cartesian stress components (COMPRESSION POSITIVE).

        Evaluated at the physical points z = omega(rho e^{i
        theta_zeta}) through the mapped KM relations of the module
        docstring (closed-form zeta-derivatives; the tension-positive
        result is negated at return -- the one stress sign conversion
        of the module).  API shape mirrors
        ``mapped_net.stresses_cartesian``.

        Parameters
        ----------
        rho, theta_zeta : array_like
            Mapped polar coordinates of the zeta plane (broadcast
            together); rho >= 1 for parts that involve the hole
            ("total", "excavation").
        coef_a, coef_b : array_like, complex, shape (N,)
            Physical-frame Laurent coefficients.
        part : str
            "total"      -> carriers + decaying terms (default),
            "excavation" -> decaying terms only,
            "insitu"     -> carriers only (exactly the uniform
                            in-situ field, any rho).

        Returns
        -------
        (sigma_xx, sigma_yy, sigma_xy) : tuple of ndarray
            ``total == insitu + excavation`` identically.
        """
        parts = {
            "total": (True, True),
            "excavation": (False, True),
            "insitu": (True, False),
        }
        if part not in parts:
            raise ValueError(
                "part must be 'total', 'excavation' or 'insitu'"
            )
        carriers, decaying = parts[part]
        coef_a, coef_b = self._as_coefficients(coef_a, coef_b)
        rho, theta_zeta = self._prepare(rho, theta_zeta,
                                        check_rho=decaying)
        zeta = rho * np.exp(1j * theta_zeta)
        Phi, Phi_p, Psi = self._field_terms(zeta, coef_a, coef_b,
                                            carriers, decaying)
        p1 = 4.0 * np.real(Phi)
        if decaying:
            z = self.omega(zeta)
            p2 = 2.0 * (np.conj(z) * Phi_p + Psi)
        else:
            # Carriers alone: Phi' = 0 identically, so the conj(z)
            # term drops and omega need not be evaluated (keeps
            # "insitu" finite for any rho, mirroring the seed).
            p2 = 2.0 * Psi
        s_xx_T = 0.5 * (p1 - np.real(p2))
        s_yy_T = 0.5 * (p1 + np.real(p2))
        s_xy_T = 0.5 * np.imag(p2)
        return -s_xx_T, -s_yy_T, -s_xy_T

    def displacements_cartesian(self, rho, theta_zeta, coef_a, coef_b,
                                E, nu):
        """Excavation-induced Cartesian displacements (plane strain).

        Evaluated from the decaying parts (phi_0, psi_0) alone (the
        carriers hold the unbounded in-situ straining), then globally
        sign-flipped: (u_x, u_y) = -(tension-positive displacement),
        exactly the convention and API shape of
        ``mapped_net.displacements_cartesian``.

        Parameters
        ----------
        rho, theta_zeta : array_like
            Mapped polar coordinates (broadcast together); rho >= 1.
        coef_a, coef_b : array_like, complex, shape (N,)
            Physical-frame Laurent coefficients.
        E, nu : float
            Young's modulus and Poisson's ratio (plane strain).

        Returns
        -------
        (u_x, u_y) : tuple of ndarray
        """
        coef_a, coef_b = self._as_coefficients(coef_a, coef_b)
        rho, theta_zeta = self._prepare(rho, theta_zeta)
        two_g = E / (1.0 + nu)  # 2 G = E / (1 + nu)
        kappa = 3.0 - 4.0 * nu
        zeta = rho * np.exp(1j * theta_zeta)
        z = self.omega(zeta)
        wp = self.omega_prime(zeta)
        phi0, psi0, dphi0 = self._decaying_potentials(zeta, coef_a,
                                                      coef_b)
        Phi0 = dphi0 / wp
        u_T = (kappa * phi0 - z * np.conj(Phi0) - np.conj(psi0)) / two_g
        return -np.real(u_T), -np.imag(u_T)

    # ------------------------------------------------------------------
    # boundary system (shared physics assembly of the solve channel)
    # ------------------------------------------------------------------

    def boundary_traction(self, theta_zeta, coef_a, coef_b,
                          part="total"):
        """Physical traction of the requested part on the hole wall.

        Components (t_x, t_y) of sigma . n formed with the returned
        (compression-positive) stress tensor at rho = 1 and the
        outward unit normal n = zeta omega'(zeta) /
        |zeta omega'(zeta)| -- exactly as
        ``mapped_net.boundary_traction``.  The global sign flip is
        immaterial for the zero-traction condition.

        Parameters
        ----------
        theta_zeta : array_like
            Mapped boundary angle(s).
        coef_a, coef_b : array_like, complex, shape (N,)
            Physical-frame Laurent coefficients.
        part : str
            As in :meth:`stresses_cartesian` (default "total").

        Returns
        -------
        (t_x, t_y) : tuple of ndarray
        """
        theta_zeta = np.asarray(theta_zeta, dtype=float)
        s_xx, s_yy, s_xy = self.stresses_cartesian(
            1.0, theta_zeta, coef_a, coef_b, part=part
        )
        s = np.exp(1j * theta_zeta)
        n = s * self.omega_prime(s)
        n = n / np.abs(n)
        n_x = np.real(n)
        n_y = np.imag(n)
        return s_xx * n_x + s_xy * n_y, s_xy * n_x + s_yy * n_y

    def boundary_system(self, theta_wall):
        """Linear system of the traction-free wall condition, G x = h.

        The TOTAL physical traction on |zeta| = 1 must vanish:
        t_x(theta_j) = 0 and t_y(theta_j) = 0.  Both components are
        R-linear in the physical coefficient vector x (ordering of
        :meth:`coefficients_to_vector`), so with the carrier
        (in-situ) contribution moved to the right-hand side::

          G x = h,   h = -(in-situ wall traction),

        rows: the t_x block (M rows) stacked over the t_y block
        (M rows).  Columns are built by unit-coefficient synthesis
        through :meth:`boundary_traction` (part="excavation"), so the
        system shares the exact synthesis code path used for
        evaluation -- the single physics source of truth.

        Parameters
        ----------
        theta_wall : array_like
            Collocation angles theta_j (flattened to 1-D).

        Returns
        -------
        (G, h) : (ndarray (2 M, 4 N), ndarray (2 M,))
        """
        theta = np.asarray(theta_wall, dtype=float).ravel()
        if theta.size == 0:
            raise ValueError("theta_wall must contain at least one angle")
        zero = np.zeros(self.n_order, dtype=complex)
        t_x_in, t_y_in = self.boundary_traction(theta, zero, zero,
                                                part="insitu")
        h = -np.concatenate([t_x_in, t_y_in])
        n_unknowns = 4 * self.n_order
        g_mat = np.empty((2 * theta.size, n_unknowns))
        for k in range(n_unknowns):
            e = np.zeros(n_unknowns)
            e[k] = 1.0
            coef_a, coef_b = self.vector_to_coefficients(e)
            t_x, t_y = self.boundary_traction(theta, coef_a, coef_b,
                                              part="excavation")
            g_mat[:, k] = np.concatenate([t_x, t_y])
        return g_mat, h


# ----------------------------------------------------------------------
# classical collocation least-squares direct solve
# ----------------------------------------------------------------------

def solve_ls(rep, n_collocation=200):
    """Classical collocation least-squares solve (pure numpy).

    Assembles the boundary system at ``n_collocation`` uniform angles
    theta_j = 2 pi j / M and solves it by ``numpy.linalg.lstsq`` in
    the SCALED unknowns (columns multiplied by ``coef_scale =
    sigma_v * R``, so the unknown vector is O(1); a uniform column
    scale leaves the least-squares/minimum-norm solution unchanged up
    to that factor).  Rank deficiency, if present, is resolved by the
    minimum-norm property of lstsq.  Mirrors ``mapped_net.solve_ls``
    verbatim (same return dict).

    Parameters
    ----------
    rep : GeneralMappedKM
    n_collocation : int
        Number M of uniform wall collocation angles.

    Returns
    -------
    dict
        ``coef_a`` / ``coef_b`` (physical, complex (N,)),
        ``coef_a_scaled`` / ``coef_b_scaled``, ``wall_s`` (assembly +
        solve wall time via time.perf_counter), ``n_params`` (0 --
        the classical channel has no trainable parameters), ``rank``
        (lstsq-reported rank of the scaled system),
        ``wall_residual_max`` (max |G x - h| / |sigma_v|: the wall
        TRACTION residual, bounded below by the Laurent truncation
        tail of the exact decaying potentials -- geometric in N with
        rate set by the singularities of the analytically continued
        solution inside the unit disk; the elliptic specialization
        has poles at +/- sqrt(m), rate sqrt(m), per the seed's
        ``refsol/elliptic.py`` docstring), and ``n_collocation``.
    """
    theta = _uniform_wall_angles(n_collocation)
    t0 = time.perf_counter()
    g_mat, h = rep.boundary_system(theta)
    g_scaled = g_mat * rep.coef_scale
    x_scaled, _, rank, _ = np.linalg.lstsq(g_scaled, h, rcond=None)
    wall_s = time.perf_counter() - t0
    x = rep.coef_scale * x_scaled
    coef_a, coef_b = rep.vector_to_coefficients(x)
    residual = g_scaled @ x_scaled - h
    return {
        "channel": "ls",
        "coef_a": coef_a,
        "coef_b": coef_b,
        "coef_a_scaled": coef_a / rep.coef_scale,
        "coef_b_scaled": coef_b / rep.coef_scale,
        "wall_s": wall_s,
        "n_params": 0,
        "rank": int(rank),
        "wall_residual_max": float(np.max(np.abs(residual))
                                   / abs(rep.sigma_v)),
        "n_collocation": int(n_collocation),
    }


# ----------------------------------------------------------------------
# univalence acceptance gate (map analysis; no solved-field physics)
# ----------------------------------------------------------------------

def _polygon_is_simple(points, chunk=256):
    """True when the closed polygon through ``points`` is simple.

    Segment k joins points[k] to points[(k+1) % K].  The strict
    orientation test detects PROPER crossings only, so adjacent
    segments -- which share an endpoint legitimately, including the
    closing pair (K-1, 0) -- never trigger; they are additionally
    excluded by index.  O(K^2) pairwise (each unordered non-adjacent
    pair tested once), evaluated in row chunks to bound memory.
    """
    pts = np.asarray(points, dtype=complex).ravel()
    K = pts.size
    if K < 4:
        return True  # a triangle or less cannot properly self-intersect
    x1 = np.real(pts)
    y1 = np.imag(pts)
    x2 = np.roll(x1, -1)
    y2 = np.roll(y1, -1)
    jj = np.arange(K)[None, :]
    cx = x1[None, :]
    cy = y1[None, :]
    dx = x2[None, :]
    dy = y2[None, :]
    for i0 in range(0, K, chunk):
        i1 = min(i0 + chunk, K)
        ii = np.arange(i0, i1)[:, None]
        ax = x1[i0:i1][:, None]
        ay = y1[i0:i1][:, None]
        bx = x2[i0:i1][:, None]
        by = y2[i0:i1][:, None]
        d1 = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
        d2 = (bx - ax) * (dy - ay) - (by - ay) * (dx - ax)
        d3 = (dx - cx) * (ay - cy) - (dy - cy) * (ax - cx)
        d4 = (dx - cx) * (by - cy) - (dy - cy) * (bx - cx)
        crossing = (d1 * d2 < 0.0) & (d3 * d4 < 0.0)
        # each unordered pair once: j >= i + 2, minus the closing
        # adjacency (0, K-1)
        nonadjacent = (jj >= ii + 2) & ~((ii == 0) & (jj == K - 1))
        if np.any(crossing & nonadjacent):
            return False
    return True


def validate_univalence(R, c, n_scan_boundary=4096, rho_scan=3.0,
                        n_rho=64, n_scan_annulus=1024):
    """Numerical univalence acceptance for one map (R, c).

    Rationale (module docstring "Univalence"): for an exterior map
    with omega(infinity) = infinity and omega' nonvanishing on
    |zeta| >= 1, a SIMPLE boundary image implies exterior univalence
    (argument principle).  Three geometric conditions are therefore
    checked -- omega' nonvanishing (dense boundary scan + annulus
    grid scan + the analytic far guard closing the unbounded rest of
    the domain), and boundary-image simplicity:

    * boundary scan: min |omega'| over ``n_scan_boundary`` uniform
      angles on |zeta| = 1;
    * annulus scan: min |omega'| over an ``n_rho x n_scan_annulus``
      grid of 1 <= rho <= ``rho_scan``;
    * far guard (analytic, no scan): for |zeta| >= rho_scan,
      |sum_{j>=1} j c_j zeta^{-(j+1)}| <= (sum_{j>=1} j |c_j|)
      / rho_scan^2, so omega' cannot vanish beyond rho_scan whenever
      sum_{j>=1} j |c_j| < rho_scan^2;
    * simplicity: the closed polygon through omega(e^{i theta_k}) at
      the ``n_scan_boundary`` angles has no proper segment
      self-intersection (adjacent segments share endpoints
      legitimately).

    Scan-resolution caveat: the scans and the polygon certify the
    sampled resolution, not the continuum; the analytic far guard
    and :meth:`GeneralMappedKM.delta_lower_bound` are the exact
    statements.

    Parameters
    ----------
    R : float
        Mapping radius (> 0).
    c : array_like, complex, shape (n_map + 1,)
        Map coefficients (c[0] = translation).
    n_scan_boundary : int
        Boundary scan / polygon resolution (>= 8).
    rho_scan : float
        Outer radius of the scanned annulus (> 1); beyond it the far
        guard takes over.
    n_rho, n_scan_annulus : int
        Annulus grid resolution (n_rho >= 2, n_scan_annulus >= 1).

    Returns
    -------
    dict
        ``min_abs_omega_prime_boundary``,
        ``min_abs_omega_prime_annulus``, ``far_guard_ok``,
        ``boundary_simple``, ``sum_j_abs_jcj``,
        ``delta_lower_bound`` (the D4 bound, as-is, possibly
        negative), and ``univalent_ok`` = both scan minima > 0 AND
        far_guard_ok AND boundary_simple.
    """
    if n_scan_boundary < 8:
        raise ValueError("n_scan_boundary must be >= 8")
    if not rho_scan > 1.0:
        raise ValueError("rho_scan must be > 1")
    if n_rho < 2:
        raise ValueError("n_rho must be >= 2")
    if n_scan_annulus < 1:
        raise ValueError("n_scan_annulus must be >= 1")
    # Map-only probe: dummy physics constants (unused); keeps the map
    # form living solely in the GeneralMappedKM mapping interface.
    probe = GeneralMappedKM(R, c, sigma_v=1.0, K0=0.0, n_order=1)
    theta_b = (2.0 * np.pi / n_scan_boundary) * np.arange(n_scan_boundary)
    s = np.exp(1j * theta_b)
    min_boundary = float(np.min(np.abs(probe.omega_prime(s))))
    rho_g = np.linspace(1.0, rho_scan, n_rho)[:, None]
    theta_a = (2.0 * np.pi / n_scan_annulus) \
        * np.arange(n_scan_annulus)[None, :]
    min_annulus = float(np.min(np.abs(
        probe.omega_prime(rho_g * np.exp(1j * theta_a))
    )))
    j = np.arange(1, probe.n_map + 1)
    sum_j_abs_jcj = float(np.sum(j * np.abs(probe.c[1:])))
    far_guard_ok = bool(sum_j_abs_jcj < rho_scan * rho_scan)
    boundary_simple = bool(_polygon_is_simple(probe.omega(s)))
    univalent_ok = bool(
        min_boundary > 0.0 and min_annulus > 0.0
        and far_guard_ok and boundary_simple
    )
    return {
        "min_abs_omega_prime_boundary": min_boundary,
        "min_abs_omega_prime_annulus": min_annulus,
        "far_guard_ok": far_guard_ok,
        "boundary_simple": boundary_simple,
        "sum_j_abs_jcj": sum_j_abs_jcj,
        "delta_lower_bound": probe.delta_lower_bound(),
        "univalent_ok": univalent_ok,
    }


# ----------------------------------------------------------------------
# self-convergence evaluation (low-N solution vs higher-N reference)
# ----------------------------------------------------------------------

def evaluate_vs_reference(rep, coef_a, coef_b, rep_ref, coef_a_ref,
                          coef_b_ref, E, nu, n_rho=101, n_theta=256):
    """Mapped-annulus metrics of a solution against a same-map reference.

    General maps have no closed-form reference solution, so the
    truncation-convergence measurement scores a low-N solution
    against a HIGHER-N solution of the SAME map (identical R and c
    required; comparing fields across different maps on this grid
    would compare different physical domains and is refused).

    Measurement convention: exactly the
    ``mapped_net.evaluate_vs_elliptic`` grid -- rho in
    ``linspace(1, 5, n_rho)`` x ``n_theta`` uniform zeta angles,
    quadrature weights for the PHYSICAL area measure

      dA_z = |omega'(zeta)|^2 * rho * drho * dtheta

    (Jacobian of z = omega(zeta); trapezoidal rule in rho -- interior
    weight drho, endpoint weight drho/2 -- times the uniform weight
    2 pi / n_theta per angle).  Synthesizes the TOTAL Cartesian
    stresses and the excavation-induced Cartesian displacements of
    both solutions and scores them with the frozen, solver-agnostic
    ``eval_v1.rel_l2`` / ``eval_v1.rel_max`` (3 stress components
    stacked; 2 displacement components).

    Parameters
    ----------
    rep : GeneralMappedKM
        Representation of the low-N solution.
    coef_a, coef_b : array_like, complex, shape (rep.n_order,)
        Physical-frame Laurent coefficients of the low-N solution.
    rep_ref : GeneralMappedKM
        Representation of the reference solution; must share the
        identical map (R, c) with ``rep``.
    coef_a_ref, coef_b_ref : array_like, complex,
    shape (rep_ref.n_order,)
        Physical-frame Laurent coefficients of the reference.
    E, nu : float
        Elastic constants for the displacement metrics.
    n_rho, n_theta : int
        Grid resolution (defaults 101 x 256, the family convention).

    Returns
    -------
    dict
        ``rel_l2`` / ``rel_max`` (stresses), ``disp_rel_l2`` /
        ``disp_rel_max`` (displacements).
    """
    if rep.R != rep_ref.R or rep.c.shape != rep_ref.c.shape \
            or not np.array_equal(rep.c, rep_ref.c):
        raise ValueError(
            "rep and rep_ref must share the identical map (R, c); "
            "fields of different maps live on different physical "
            "domains and are not comparable on this grid"
        )
    if n_rho < 2:
        raise ValueError("n_rho must be >= 2 (trapezoidal rule in rho)")
    if n_theta < 1:
        raise ValueError("n_theta must be >= 1")
    rho_1d = np.linspace(1.0, 5.0, n_rho)
    drho = 4.0 / (n_rho - 1)
    w_rho = np.full(n_rho, drho)
    w_rho[0] = 0.5 * drho
    w_rho[-1] = 0.5 * drho
    theta_1d = (2.0 * np.pi / n_theta) * np.arange(n_theta)
    rho, theta = np.meshgrid(rho_1d, theta_1d, indexing="ij")
    wp = rep.omega_prime(rho * np.exp(1j * theta))
    w = (np.abs(wp) ** 2) * rho * (w_rho[:, None] * (2.0 * np.pi / n_theta))
    s_pred = rep.stresses_cartesian(rho, theta, coef_a, coef_b,
                                    part="total")
    s_ref = rep_ref.stresses_cartesian(rho, theta, coef_a_ref,
                                       coef_b_ref, part="total")
    u_pred = rep.displacements_cartesian(rho, theta, coef_a, coef_b,
                                         E, nu)
    u_ref = rep_ref.displacements_cartesian(rho, theta, coef_a_ref,
                                            coef_b_ref, E, nu)
    return {
        "rel_l2": eval_v1.rel_l2(s_pred, s_ref, w),
        "rel_max": eval_v1.rel_max(s_pred, s_ref),
        "disp_rel_l2": eval_v1.rel_l2(u_pred, u_ref, w),
        "disp_rel_max": eval_v1.rel_max(u_pred, u_ref),
    }
