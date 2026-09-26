"""Elliptic-hole analytical reference solution (conformal mapping).

Stage-2 reference of the seed experiment ``seed_skeleton_km``
(alongside ``kirsch.py`` and ``ring_composite.py``): trusted
closed-form plane-strain elasticity solution for a deep tunnel of
ELLIPTIC cross-section -- the family's first non-circular geometry
and the anchor for the stage-2 conformal-mapping
Kolosov-Muskhelishvili networks and their Laurent
truncation-convergence study.  Pure numpy; no other dependencies.

Problem
-------
An elliptic hole (the tunnel) is excavated in an infinite,
homogeneous, isotropic, linear-elastic medium (Young's modulus ``E``,
Poisson's ratio ``nu``) under plane-strain conditions; quasi-static,
no body force.  The exterior of the hole is parameterized by the
exterior conformal map

  z = omega(zeta) = R (zeta + m / zeta),   |zeta| >= 1,

with R > 0 and 0 <= m < 1: the unit circle |zeta| = 1 maps onto the
ellipse with semi-axes

  a_ell = R (1 + m)  along x,     b_ell = R (1 - m)  along y,

and m = 0 degenerates to the circle of radius R.  The pre-excavation
(in-situ) stress field is uniform:

* ``sigma_v``              -- vertical principal in-situ stress,
* ``sigma_h = K0*sigma_v`` -- horizontal principal in-situ stress.

Excavation removes the core (|zeta| < 1) and leaves the elliptic
wall fully traction free -- single-step full excavation release,
exactly the Kirsch-problem semantics of ``kirsch.py`` generalized to
the ellipse.  This module evaluates, at mapped-polar points
(rho, theta_zeta) with zeta = rho * exp(i theta_zeta) and rho >= 1,
vectorized over numpy arrays:

* the TOTAL Cartesian stress field (in-situ field + excavation-
  induced perturbation), with the same ``part`` split as
  ``kirsch.py``,
* the EXCAVATION-INDUCED Cartesian displacement field (displacement
  release relative to the pre-excavation state; decays to zero as
  rho -> infinity), and
* the physical traction of the total field on the hole wall
  (:func:`boundary_traction`; identically zero for this solution).

Coordinates and angle convention
--------------------------------
x horizontal (direction of sigma_h), y vertical (direction of
sigma_v), exactly as in ``kirsch.py``.  Points are addressed by the
mapped polar coordinates (rho, theta_zeta) of the zeta plane;
theta_zeta is measured counter-clockwise from the positive x axis:

* theta_zeta = 0, pi         -> springline (tips of a_ell),
* theta_zeta = pi/2, 3*pi/2  -> crown / invert (tips of b_ell).

For m > 0 theta_zeta is NOT the physical polar angle of
z = omega(zeta) (they coincide at m = 0 and on the symmetry axes).
Returned field components are therefore CARTESIAN (x, y), which is
frame-unambiguous at every point; the circular-family modules return
polar components instead, and the verification tests rotate between
the two frames where a comparison is made.

Sign convention (geomechanics; compression positive)
----------------------------------------------------
Identical to ``kirsch.py``.  Stresses are COMPRESSION POSITIVE: the
inputs sigma_v and sigma_h = K0*sigma_v are compressive magnitudes,
and the returned Cartesian components (sigma_xx, sigma_yy, sigma_xy)
form the negative of the tension-positive elasticity stress tensor.
Displacements are reported in the SAME globally sign-flipped system:

  (u_x, u_y) = -(tension-positive elasticity displacement),

so the radial projection u_x cos t + u_y sin t (t = physical polar
angle of the point) is positive when the point moves TOWARD the
tunnel axis, matching the convergence-positive u_r of ``kirsch.py``.
Because every governing relation (equilibrium, strain-displacement,
Hooke's law) is linear and homogeneous, the global flip is
self-consistent: the returned stresses satisfy the Cartesian
equilibrium equations verbatim, and the standard plane-strain Hooke
law maps the strains computed from the returned excavation
displacements exactly onto the returned excavation stresses.  The
verification tests exploit this.

Kolosov-Muskhelishvili potentials (exact; textbook material)
------------------------------------------------------------
In the TENSION-POSITIVE frame (far field S_x = -sigma_h,
S_y = -sigma_v) define the far-field carriers of the family

  Gamma  = -(sigma_v + sigma_h) / 4,
  Gamma' = -(sigma_v - sigma_h) / 2.

The exact potentials of the traction-free elliptic exterior, written
as functions of zeta, are

  phi(zeta) = Gamma R zeta + A / zeta,

  psi(zeta) = Gamma' R zeta - Gamma R / zeta
              - (B zeta^2 - A) / (zeta (zeta^2 - m)),

with the REAL constants

  A = -R (Gamma' + m Gamma),
  B =  R (Gamma (1 + 2 m^2) + m Gamma').

Derivation sketch (boundary-condition solve)
--------------------------------------------
Zero traction on the whole wall and zero resultant force admit
potentials phi = Gamma R zeta + phi_0, psi = Gamma' R zeta + psi_0
with phi_0, psi_0 holomorphic in |zeta| > 1 and vanishing at
infinity (no logarithmic terms).  The traction-free condition on the
unit circle s = exp(i theta), far-field carriers included and the
zero-traction constant gauged to zero, reads

  phi(s) + [omega(s) / conj(omega'(s))] conj(phi'(s))
         + conj(psi(s)) = 0.

Conjugating and continuing analytically off |s| = 1 expresses psi
through phi ("bar" = conjugated coefficients; real here):

  psi(zeta) = -bar_phi(1/zeta)
              - [bar_omega(1/zeta) / omega'(zeta)] phi'(zeta),

  bar_omega(1/zeta) / omega'(zeta)
      = zeta (1 + m zeta^2) / (zeta^2 - m).

For this psi to be holomorphic in |zeta| > 1 with the prescribed
linear far field, bar_phi_0(1/zeta) -- a series in POSITIVE powers
of zeta -- must reduce to a polynomial of degree <= 1: hence phi_0
is the single term A / zeta, and matching the coefficient of zeta in
the expansion at infinity (-A - m Gamma R = Gamma' R) fixes
A = -R (Gamma' + m Gamma).  Substituting phi and dividing out the
polynomial part yields the closed rational psi quoted above.  All
coefficients are real because the biaxial principal directions are
aligned with the ellipse axes.

Pole structure and truncation reference
---------------------------------------
The decaying part of psi contains the rational term

  T(zeta) = (B zeta^2 - A) / (zeta (zeta^2 - m)),

whose poles sit at zeta = +/- sqrt(m) (and zeta = 0), strictly
INSIDE the unit disk -- the preimages of the ellipse foci
z = +/- 2 R sqrt(m), with the +/- sqrt(m) pair inherited from
bar_omega(1/zeta) / omega'(zeta) in the boundary-condition solve.
Its Laurent expansion valid on |zeta| > 1,

  T(zeta) = B / zeta
            + sum_{j >= 1} m^{j-1} (B m - A) zeta^{-(2 j + 1)},

converges only down to |zeta| = sqrt(m): the coefficient of
zeta^{-n} (odd n) decays like m^{n/2}.  This module is therefore the
reference for the stage-2 Laurent truncation-convergence study, with
the mapping non-degeneracy measure

  delta = min_{|zeta| = 1} |omega'(zeta)| = R (1 - m) > 0

(attained at the springline tips zeta = +/- 1).

Kirsch degeneration (m = 0, R = a)
----------------------------------
A = -a Gamma', B = a Gamma, omega(zeta) = a zeta; with z = a zeta
the potentials collapse exactly onto the circular-hole potentials of
``kirsch.py``::

  phi(z) = Gamma z - Gamma' a^2 / z
         = -(P/2) z + D a^2 / z,
  psi(z) = Gamma' z - 2 Gamma a^2 / z - Gamma' a^4 / z^3
         = -D z + P a^2 / z + D a^4 / z^3,

with P = (sigma_v + sigma_h)/2 and D = (sigma_v - sigma_h)/2 -- the
docstring potentials of ``kirsch.py`` verbatim.

Field synthesis (implemented relations)
---------------------------------------
With the closed-form zeta-derivatives of phi, psi and omega (no
autodiff, no numerical differentiation) and

  Phi(z)  = phi'(zeta) / omega'(zeta),
  Psi(z)  = psi'(zeta) / omega'(zeta),
  Phi'(z) = [phi''(zeta) omega'(zeta) - phi'(zeta) omega''(zeta)]
            / omega'(zeta)^3,

the tension-positive fields are

  sigma_xx + sigma_yy                = 4 Re Phi(z),
  sigma_yy - sigma_xx + 2 i sigma_xy = 2 [conj(z) Phi'(z) + Psi(z)],
  2 G (u_x + i u_y)                  = kappa phi - z conj(Phi)
                                       - conj(psi),

with kappa = 3 - 4 nu and G = E / (2 (1 + nu)), and the module
returns the globally flipped quantities.  Part split (same meaning
as ``kirsch.py``), implemented by coefficient selection rather than
subtraction:

* "insitu"     -- the uniform carriers (Gamma z, Gamma' z):
                  sigma_xx = sigma_h, sigma_yy = sigma_v,
                  sigma_xy = 0 (compression positive);
* "excavation" -- the physically decaying remainder, potentials

                    phi_exc(zeta) = (A - m Gamma R) / zeta,
                    psi_exc(zeta) = -(Gamma R + m Gamma' R) / zeta
                                    - T(zeta),

                  i.e. the total potentials minus Gamma omega(zeta)
                  resp. Gamma' omega(zeta) (in the zeta variable the
                  1/zeta pieces of the carriers move into the
                  decaying part);
* "total"      -- insitu + excavation, identically.

The excavation-induced displacement evaluates the displacement
formula on (phi_exc, psi_exc) alone; it decays to zero at infinity,
and the mirror symmetries of the loading exclude rigid-body motion,
so no gauge term is needed.

Scope notes: only the in-plane components are returned (the
out-of-plane plane-strain reaction sigma_zz is not modelled); the
solution is valid for rho >= 1 and 0 <= m < 1.
"""

import numpy as np

from .kirsch import shear_modulus

__all__ = [
    "omega",
    "omega_prime",
    "stresses_cartesian",
    "displacements_cartesian",
    "boundary_traction",
]

# Accept rho >= 1 up to a small relative rounding slack.
_RHO_MIN = 1.0 - 1e-9

_PARTS = ("total", "excavation", "insitu")


def _validate_map(R, m):
    """Validate the conformal-map parameters."""
    if not R > 0.0:
        raise ValueError("mapping radius R must be positive")
    if not (0.0 <= m < 1.0):
        raise ValueError("mapping parameter m must satisfy 0 <= m < 1")


def omega(zeta, R, m):
    """Exterior conformal map z = omega(zeta) = R (zeta + m / zeta).

    Maps |zeta| >= 1 onto the exterior of the ellipse with semi-axes
    a_ell = R (1 + m) along x and b_ell = R (1 - m) along y.

    Parameters
    ----------
    zeta : array_like (complex)
        Points of the zeta plane (intended domain |zeta| >= 1).
    R : float
        Mapping radius (> 0).
    m : float
        Mapping parameter (0 <= m < 1); m = 0 gives z = R zeta.

    Returns
    -------
    ndarray (complex)
        Physical points z = omega(zeta).
    """
    _validate_map(R, m)
    zeta = np.asarray(zeta, dtype=complex)
    return R * (zeta + m / zeta)


def omega_prime(zeta, R, m):
    """Derivative of the map: omega'(zeta) = R (1 - m / zeta^2).

    Nonzero on |zeta| >= 1 for 0 <= m < 1, with
    min_{|zeta| = 1} |omega'| = R (1 - m) (the mapping
    non-degeneracy measure delta of the module docstring).

    Parameters as in :func:`omega`; complex in, complex out.
    """
    _validate_map(R, m)
    zeta = np.asarray(zeta, dtype=complex)
    return R * (1.0 - m / (zeta * zeta))


def _prepare(rho, theta_zeta, R, m, check_rho=True):
    """Broadcast (rho, theta_zeta) to floats and validate rho >= 1."""
    _validate_map(R, m)
    rho, theta_zeta = np.broadcast_arrays(
        np.asarray(rho, dtype=float), np.asarray(theta_zeta, dtype=float)
    )
    if check_rho and np.any(rho < _RHO_MIN):
        raise ValueError("elliptic solution is defined for rho >= 1 only")
    return rho, theta_zeta


def _far_field(sigma_v, K0):
    """Tension-positive far-field carriers (Gamma, Gamma')."""
    sigma_h = K0 * sigma_v
    return -0.25 * (sigma_v + sigma_h), -0.5 * (sigma_v - sigma_h)


def _km_constants(R, m, sigma_v, K0):
    """Constants (g1, g2, A, B) of the docstring potentials.

    g1 = Gamma R and g2 = Gamma' R are the linear (far-field carrier)
    coefficients; A and B are the decaying-part constants
    A = -R (Gamma' + m Gamma), B = R (Gamma (1 + 2 m^2) + m Gamma').
    """
    Gamma, Gamma_p = _far_field(sigma_v, K0)
    g1 = Gamma * R
    g2 = Gamma_p * R
    A = -(g2 + m * g1)
    B = g1 * (1.0 + m * m) - m * A
    return g1, g2, A, B


def _potentials(zeta, R, m, sigma_v, K0, part):
    """phi, phi', phi'', psi, psi' of the requested part at zeta.

    Tension-positive frame, closed-form zeta-derivatives.  The parts
    are selected by coefficient, not by subtraction:

    * "total":       phi = g1 zeta + A / zeta,
                     psi = g2 zeta - g1 / zeta - T(zeta),
    * "excavation":  phi = (A - m g1) / zeta,
                     psi = -(g1 + m g2) / zeta - T(zeta),

    with T(zeta) = (B zeta^2 - A) / (zeta (zeta^2 - m)) shared by
    both parts (T is entirely excavation-induced).
    """
    g1, g2, A, B = _km_constants(R, m, sigma_v, K0)
    if part == "total":
        p_lin, p_inv = g1, A
        q_lin, q_inv = g2, -g1
    else:  # "excavation": total minus the carriers Gamma/Gamma' omega
        p_lin, p_inv = 0.0, A - m * g1
        q_lin, q_inv = 0.0, -(g1 + m * g2)
    inv = 1.0 / zeta
    inv2 = inv * inv
    inv3 = inv2 * inv
    phi = p_lin * zeta + p_inv * inv
    dphi = p_lin - p_inv * inv2
    ddphi = 2.0 * p_inv * inv3
    zeta2 = zeta * zeta
    den = zeta * (zeta2 - m)
    T = (B * zeta2 - A) / den
    dT = (-B * zeta2 * zeta2 + (3.0 * A - B * m) * zeta2 - A * m) \
        / (den * den)
    psi = q_lin * zeta + q_inv * inv - T
    dpsi = q_lin - q_inv * inv2 - dT
    return phi, dphi, ddphi, psi, dpsi


def stresses_cartesian(rho, theta_zeta, R, m, sigma_v, K0, part="total"):
    """Cartesian stress components (compression positive).

    Evaluated at the physical points z = omega(rho exp(i theta_zeta))
    through the Kolosov-Muskhelishvili relations of the module
    docstring (closed-form zeta-derivatives, global sign flip).

    Parameters
    ----------
    rho, theta_zeta : array_like
        Mapped polar coordinates of the zeta plane (broadcast
        together); rho >= 1 for parts that involve the hole
        ("total", "excavation").
    R : float
        Mapping radius (> 0).
    m : float
        Mapping parameter (0 <= m < 1).
    sigma_v : float
        Vertical in-situ stress (compression positive).
    K0 : float
        Lateral pressure coefficient; sigma_h = K0 * sigma_v.
    part : str
        "total"      -> in-situ + excavation-induced field (default),
        "excavation" -> perturbation caused by the excavation only,
        "insitu"     -> undisturbed in-situ field (uniform).

    Returns
    -------
    (sigma_xx, sigma_yy, sigma_xy) : tuple of ndarray
        Cartesian components at the broadcast shape of
        (rho, theta_zeta).  ``total == insitu + excavation``
        identically.
    """
    if part not in _PARTS:
        raise ValueError("part must be 'total', 'excavation' or 'insitu'")
    rho, theta_zeta = _prepare(rho, theta_zeta, R, m,
                               check_rho=(part != "insitu"))
    if part == "insitu":
        shape = rho.shape
        s_xx = np.full(shape, K0 * sigma_v, dtype=float)
        s_yy = np.full(shape, sigma_v, dtype=float)
        s_xy = np.zeros(shape)
        return s_xx, s_yy, s_xy
    zeta = rho * np.exp(1j * theta_zeta)
    z = omega(zeta, R, m)
    wp = omega_prime(zeta, R, m)
    wpp = 2.0 * R * m / (zeta * zeta * zeta)
    _, dphi, ddphi, _, dpsi = _potentials(zeta, R, m, sigma_v, K0, part)
    Phi = dphi / wp
    Phi_p = (ddphi * wp - dphi * wpp) / (wp * wp * wp)
    Psi = dpsi / wp
    p1 = 4.0 * np.real(Phi)
    p2 = 2.0 * (np.conj(z) * Phi_p + Psi)
    s_xx_T = 0.5 * (p1 - np.real(p2))
    s_yy_T = 0.5 * (p1 + np.real(p2))
    s_xy_T = 0.5 * np.imag(p2)
    return -s_xx_T, -s_yy_T, -s_xy_T


def displacements_cartesian(rho, theta_zeta, R, m, sigma_v, K0, E, nu):
    """Excavation-induced Cartesian displacements (plane strain).

    The displacement release caused by removing the core, evaluated
    from the decaying potential parts (phi_exc, psi_exc) alone; it
    decays to zero as rho -> infinity.  Sign convention (module
    docstring): (u_x, u_y) = -(tension-positive elasticity
    displacement), so the radial projection is positive toward the
    tunnel axis (convergence), as in ``kirsch.py``.

    Parameters
    ----------
    rho, theta_zeta : array_like
        Mapped polar coordinates (broadcast together); rho >= 1.
    R, m : float
        Conformal-map parameters as in :func:`stresses_cartesian`.
    sigma_v, K0 : float
        In-situ field as in :func:`stresses_cartesian`.
    E, nu : float
        Young's modulus and Poisson's ratio (plane strain).

    Returns
    -------
    (u_x, u_y) : tuple of ndarray
        Components at the broadcast shape of (rho, theta_zeta).
    """
    rho, theta_zeta = _prepare(rho, theta_zeta, R, m)
    G = shear_modulus(E, nu)
    kap = 3.0 - 4.0 * nu
    zeta = rho * np.exp(1j * theta_zeta)
    z = omega(zeta, R, m)
    wp = omega_prime(zeta, R, m)
    phi, dphi, _, psi, _ = _potentials(zeta, R, m, sigma_v, K0,
                                       "excavation")
    Phi = dphi / wp
    disp_T = (kap * phi - z * np.conj(Phi) - np.conj(psi)) / (2.0 * G)
    return -np.real(disp_T), -np.imag(disp_T)


def boundary_traction(theta_zeta, R, m, sigma_v, K0):
    """Physical traction of the TOTAL field on the hole wall.

    Components (t_x, t_y) of sigma . n formed with the returned
    (compression-positive) total stress tensor at rho = 1 and the
    outward unit normal of the ellipse (pointing from the hole into
    the medium, direction zeta omega'(zeta) / |zeta omega'(zeta)|).
    Identically zero for this traction-free solution; used by the
    verification tests.

    Parameters
    ----------
    theta_zeta : array_like
        Mapped boundary angle(s).
    R, m, sigma_v, K0 : float
        As in :func:`stresses_cartesian`.

    Returns
    -------
    (t_x, t_y) : tuple of ndarray
        Traction components at the shape of theta_zeta.
    """
    theta_zeta = np.asarray(theta_zeta, dtype=float)
    s_xx, s_yy, s_xy = stresses_cartesian(1.0, theta_zeta, R, m,
                                          sigma_v, K0, part="total")
    s = np.exp(1j * theta_zeta)
    n = s * omega_prime(s, R, m)
    n = n / np.abs(n)
    n_x = np.real(n)
    n_y = np.imag(n)
    return s_xx * n_x + s_xy * n_y, s_xy * n_x + s_yy * n_y
