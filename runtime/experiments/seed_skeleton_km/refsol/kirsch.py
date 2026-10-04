"""Kirsch analytical reference solution for a deep circular tunnel.

Stage 0 of the seed experiment ``seed_skeleton_km``: trusted analytical
reference (plus verification tests) for the exterior plane-strain
elasticity problem that the Kolosov-Muskhelishvili potential network
will later be trained on.  Pure numpy; no other dependencies.

Problem
-------
A circular hole of radius ``a`` (the tunnel) is excavated in an
infinite, homogeneous, isotropic, linear-elastic medium (Young's
modulus ``E``, Poisson's ratio ``nu``) under plane-strain conditions.
The pre-excavation (in-situ) stress field is uniform:

* ``sigma_v``              -- vertical principal in-situ stress,
* ``sigma_h = K0*sigma_v`` -- horizontal principal in-situ stress.

Excavation removes the core ``r < a`` and leaves the wall ``r = a``
traction free.  This module evaluates, at arbitrary polar points
(r, theta) with r >= a, vectorized over numpy arrays:

* the TOTAL stress field (in-situ field + excavation-induced
  perturbation), and
* the EXCAVATION-INDUCED displacement field (the displacement release
  caused by removing the core, i.e. displacement measured relative to
  the pre-excavation state; it decays to zero as r -> infinity).

Coordinates and theta convention
--------------------------------
x horizontal (direction of sigma_h), y vertical (direction of
sigma_v).  Polar coordinates (r, theta) are centred on the tunnel
axis; theta is measured counter-clockwise from the positive x axis:

* theta = 0, pi          -> springline (horizontal diameter),
* theta = pi/2, 3*pi/2   -> crown / invert (vertical diameter).

Sign convention (geomechanics; compression positive)
----------------------------------------------------
Stresses are COMPRESSION POSITIVE.  The inputs sigma_v and
sigma_h = K0*sigma_v are compressive magnitudes, and the returned
polar components (sigma_rr, sigma_tt, sigma_rt) form the negative of
the tension-positive elasticity stress tensor.

Displacements are reported in the SAME globally sign-flipped system:

* u_r     > 0 : the point moves TOWARD the tunnel axis (convergence),
* u_theta > 0 : the point moves in the direction of DECREASING theta,

i.e. (u_r, u_theta) = -(tension-positive elasticity displacement
components).  Because every governing relation (equilibrium,
strain-displacement, Hooke's law) is linear and homogeneous, this
global flip is self-consistent: the returned stresses satisfy the
standard polar equilibrium equations verbatim, and the standard
plane-strain Hooke's law maps the strains computed from the returned
displacements exactly onto the returned excavation-induced
(perturbation) stresses.  The verification tests exploit this.

Closed-form fields
------------------
With m = (sigma_v + sigma_h)/2, d = (sigma_v - sigma_h)/2,
rho = a/r and G = E / (2 (1 + nu)):

total stresses (compression positive)::

  sigma_rr = m (1 - rho^2) - d (1 - 4 rho^2 + 3 rho^4) cos(2 theta)
  sigma_tt = m (1 + rho^2) + d (1 + 3 rho^4) cos(2 theta)
  sigma_rt = d (1 + 2 rho^2 - 3 rho^4) sin(2 theta)

in-situ stresses (the rho -> 0 limit)::

  sigma_rr = m - d cos(2 theta)
  sigma_tt = m + d cos(2 theta)
  sigma_rt = d sin(2 theta)

excavation-induced displacements (plane strain)::

  u_r     = (a^2 / (2 G r)) [ m - d (4 (1 - nu) - rho^2) cos(2 theta) ]
  u_theta = (a^2 / (2 G r)) [ d (2 (1 - 2 nu) + rho^2) sin(2 theta) ]

Kolosov-Muskhelishvili potentials (reference for later stages)
--------------------------------------------------------------
In the TENSION-POSITIVE frame (far field S_x = -sigma_h,
S_y = -sigma_v) the exact potentials of this solution are::

  phi(z) = -(m/2) z + d a^2 / z
  psi(z) = -d z + m a^2 / z + d a^4 / z^3

with the usual plane-strain relations (kappa = 3 - 4 nu)::

  sigma_rr + sigma_tt                = 4 Re phi'(z)
  sigma_tt - sigma_rr + 2 i sigma_rt = 2 e^{2 i theta}
                                       (conj(z) phi''(z) + psi'(z))
  2 G (u_x + i u_y)                  = kappa phi(z) - z conj(phi'(z))
                                       - conj(psi(z))

The excavation-induced displacement corresponds to the decaying parts
phi_0 = d a^2 / z and psi_0 = m a^2 / z + d a^4 / z^3 alone (the
linear far-field terms carry the unbounded in-situ straining).  These
potentials are cross-checked numerically against the polar formulas
implemented below in ``tests/test_kirsch.py``.

Scope notes: only the in-plane components are returned (the
out-of-plane plane-strain reaction sigma_zz is not modelled); the
solution is valid for r >= a.
"""

import numpy as np

__all__ = ["shear_modulus", "stresses", "displacements"]

# Accept r >= a up to a small relative rounding slack.
_R_MIN_FACTOR = 1.0 - 1e-9


def shear_modulus(E, nu):
    """Shear modulus G = E / (2 (1 + nu))."""
    return E / (2.0 * (1.0 + nu))


def _prepare(r, theta, a, check_r=True):
    """Broadcast (r, theta) to a common float shape and validate r >= a."""
    if not a > 0.0:
        raise ValueError("hole radius a must be positive")
    r, theta = np.broadcast_arrays(
        np.asarray(r, dtype=float), np.asarray(theta, dtype=float)
    )
    if check_r and np.any(r < a * _R_MIN_FACTOR):
        raise ValueError("Kirsch solution is defined for r >= a only")
    return r, theta


def _mean_deviator(sigma_v, K0):
    """Mean and half-difference of the in-situ field (compression +)."""
    sigma_h = K0 * sigma_v
    return 0.5 * (sigma_v + sigma_h), 0.5 * (sigma_v - sigma_h)


def stresses(r, theta, a, sigma_v, K0, part="total"):
    """Polar stress components around the tunnel (compression positive).

    Parameters
    ----------
    r, theta : array_like
        Polar coordinates (broadcast together); r >= a for parts that
        involve the hole ("total", "excavation").
    a : float
        Tunnel radius (> 0).
    sigma_v : float
        Vertical in-situ stress (compression positive).
    K0 : float
        Lateral pressure coefficient; sigma_h = K0 * sigma_v.
    part : str
        "total"      -> in-situ + excavation-induced field (default),
        "excavation" -> perturbation caused by the excavation only,
        "insitu"     -> undisturbed in-situ field (r-independent).

    Returns
    -------
    (sigma_rr, sigma_tt, sigma_rt) : tuple of ndarray
        Radial, hoop and shear components at the broadcast shape of
        (r, theta).  ``total == insitu + excavation`` identically.
    """
    r, theta = _prepare(r, theta, a, check_r=(part != "insitu"))
    m, d = _mean_deviator(sigma_v, K0)
    c2 = np.cos(2.0 * theta)
    s2 = np.sin(2.0 * theta)
    if part == "insitu":
        return m - d * c2, m + d * c2, d * s2
    rho2 = (a / r) ** 2
    rho4 = rho2 * rho2
    if part == "excavation":
        s_rr = -m * rho2 + d * (4.0 * rho2 - 3.0 * rho4) * c2
        s_tt = m * rho2 + 3.0 * d * rho4 * c2
        s_rt = d * (2.0 * rho2 - 3.0 * rho4) * s2
        return s_rr, s_tt, s_rt
    if part == "total":
        s_rr = m * (1.0 - rho2) - d * (1.0 - 4.0 * rho2 + 3.0 * rho4) * c2
        s_tt = m * (1.0 + rho2) + d * (1.0 + 3.0 * rho4) * c2
        s_rt = d * (1.0 + 2.0 * rho2 - 3.0 * rho4) * s2
        return s_rr, s_tt, s_rt
    raise ValueError("part must be 'total', 'excavation' or 'insitu'")


def displacements(r, theta, a, sigma_v, K0, E, nu):
    """Excavation-induced displacements (plane strain).

    The displacement release caused by removing the core, i.e. the
    displacement measured relative to the pre-excavation state; it
    decays to zero as r -> infinity.  Sign convention (module
    docstring): u_r > 0 toward the tunnel axis (convergence),
    u_theta > 0 toward decreasing theta.

    Parameters
    ----------
    r, theta : array_like
        Polar coordinates (broadcast together); r >= a.
    a : float
        Tunnel radius (> 0).
    sigma_v, K0 : float
        In-situ field as in :func:`stresses`.
    E, nu : float
        Young's modulus and Poisson's ratio (plane strain).

    Returns
    -------
    (u_r, u_theta) : tuple of ndarray
        Components at the broadcast shape of (r, theta).
    """
    r, theta = _prepare(r, theta, a)
    m, d = _mean_deviator(sigma_v, K0)
    G = shear_modulus(E, nu)
    rho2 = (a / r) ** 2
    pref = a * a / (2.0 * G * r)
    u_r = pref * (m - d * (4.0 * (1.0 - nu) - rho2) * np.cos(2.0 * theta))
    u_t = pref * d * (2.0 * (1.0 - 2.0 * nu) + rho2) * np.sin(2.0 * theta)
    return u_r, u_t
