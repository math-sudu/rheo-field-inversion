"""Two-zone composite (liner ring + infinite ground) reference solution.

Stage 0 companion of the seed experiment ``seed_skeleton_km``
(alongside ``kirsch.py``): trusted closed-form plane-strain
elasticity reference for a lined deep circular tunnel, against which
the Kolosov-Muskhelishvili potential networks of the later stages
will be verified.  Pure numpy; no other dependencies.

Problem
-------
A circular elastic liner ring (inner radius ``R_i``, outer radius
``R_o``, Young's modulus ``E_l``, Poisson's ratio ``nu_l``) is
perfectly bonded at ``r = R_o`` to an infinite, homogeneous,
isotropic, linear-elastic ground (``E_g``, ``nu_g``); plane strain,
quasi-static, no body force.  Loading configuration: WISHED IN
PLACE -- the bonded composite is initially stress free and is loaded
in one simultaneous step by

* a uniform far-field (in-situ) stress state: ``sigma_v`` vertical
  (y direction), ``sigma_h = K0 * sigma_v`` horizontal (x
  direction), ``K0 != 1`` allowed, and
* a uniform normal pressure ``p_i`` on the inner wall ``r = R_i``
  (compression positive, zero wall shear).

There is NO excavation-release staging (no partial-relief factor
lambda): this module models the single-step loading of the composite
only.  Perfect bond: both displacement components and both traction
components (sigma_rr, sigma_rt) are continuous across ``r = R_o``.
The module evaluates, vectorized over numpy arrays, the TOTAL stress
field and the TOTAL displacement field (reference state below) at
arbitrary polar points (r, theta) in BOTH zones (``zone="liner"``:
R_i <= r <= R_o; ``zone="ground"``: r >= R_o).

Coordinates and theta convention
--------------------------------
x horizontal (direction of sigma_h), y vertical (direction of
sigma_v).  Polar coordinates (r, theta) are centred on the tunnel
axis; theta is measured counter-clockwise from the positive x axis:

* theta = 0, pi          -> springline (horizontal diameter),
* theta = pi/2, 3*pi/2   -> crown / invert (vertical diameter).

Sign convention (geomechanics; compression positive)
----------------------------------------------------
Identical to ``kirsch.py``.  Stresses are COMPRESSION POSITIVE: the
inputs sigma_v, sigma_h = K0*sigma_v and p_i are compressive
magnitudes, and the returned polar components (sigma_rr, sigma_tt,
sigma_rt) form the negative of the tension-positive elasticity
stress tensor (so ``sigma_rr = +p_i`` at ``r = R_i``).
Displacements are reported in the SAME globally sign-flipped system:

* u_r     > 0 : the point moves TOWARD the tunnel axis (convergence),
* u_theta > 0 : the point moves in the direction of DECREASING theta,

i.e. (u_r, u_theta) = -(tension-positive elasticity displacement
components).  Because every governing relation (equilibrium,
strain-displacement, Hooke's law) is linear and homogeneous, the
global flip is self-consistent: the returned stresses satisfy the
standard polar equilibrium equations verbatim, and in EACH zone the
standard plane-strain Hooke law (with that zone's constants) maps
the strains computed from the returned displacements exactly onto
that zone's returned total stresses.

Displacement quantity and reference state
-----------------------------------------
The returned displacement is the TOTAL displacement of the single
wished-in-place loading step: position of each material point in the
loaded equilibrium state minus its position in the initial,
stress-free, unloaded state of the bonded composite.  No reference
field is subtracted.  The rigid-body gauge is fixed by the two
mirror symmetries of the loading (the x and y axes are symmetry
planes of geometry and load), which exclude rigid translation and
rotation; equivalently, the displacement is exactly the standard
Kolosov-Muskhelishvili displacement formula evaluated with the
potentials quoted below and no added rigid-body term.

Because the far-field stress strains the unbounded ground
homogeneously, the total displacement grows linearly in r as
r -> infinity (u ~ eps_inf . x, where eps_inf is the plane-strain
strain of the in-situ stress state under the GROUND constants); it
is finite at every finite point but is NOT a decaying perturbation.

A "perturbation" split relative to the homogeneous far-field
straining is deliberately NOT used for the returned quantities:
inside the liner the far-field reference strain field is not a Hooke
pair under the liner's constants (material mismatch), so subtracting
it would break the exact strain <-> stress correspondence in the
ring zone, whereas the total field keeps the correspondence exact in
both zones and keeps the returned displacements continuous across
the bonded interface.

Kolosov-Muskhelishvili potentials (exact, per zone)
---------------------------------------------------
In the TENSION-POSITIVE frame (far field S_x = -sigma_h,
S_y = -sigma_v; inner wall sigma_rr = -p_i) the potentials of the
implemented solution are Laurent forms with REAL coefficients::

  liner ring (R_i <= r <= R_o):
    phi_l(z) = a1 z + a3 z^3 + am1 / z
    psi_l(z) = b1 z + bm1 / z + bm3 / z^3

  ground (r >= R_o), with Gamma  = -(sigma_v + sigma_h) / 4
                     and  Gamma' = -(sigma_v - sigma_h) / 2:
    phi_g(z) = Gamma z + cm1 / z
    psi_g(z) = Gamma' z + dm1 / z + dm3 / z^3

This is the full annulus basis for the angular harmonics n = 0 and
n = 2 excited by the loading.  Logarithmic terms, admissible in an
annulus / punctured plane in general, vanish identically here: the
resultant force of the tractions on the inner wall (uniform normal
pressure, zero shear) is zero and the displacements are
single-valued, which forces both ring log coefficients to zero; for
the ground, the force transmitted across any circle enclosing the
tunnel is zero (force balance), so its log term is zero as well.
The linear far-field terms Gamma z and Gamma' z carry the in-situ
state (rho -> 0 limit of the Kirsch potentials in ``kirsch.py``).

The nine free coefficients (a1, a3, am1, b1, bm1, bm3) and
(cm1, dm1, dm3), returned by :func:`km_coefficients`, solve the nine
boundary/interface conditions ([.] = jump across r = R_o)

* n = 0:  sigma_rr(R_i) = -p_i;   [sigma_rr] = [u_r] = 0,
* n = 2:  sigma_rr(R_i) = sigma_rt(R_i) = 0;
          [sigma_rr] = [sigma_rt] = [u_r] = [u_theta] = 0,

via two small row-equilibrated linear systems (3x3 and 6x6, exact
closed-form conditions; the displacement rows are scaled by
G_l / G_g so the vanishing-liner limit stays well conditioned).  The
usual plane-strain relations (kappa = 3 - 4 nu, zone-wise constants)

::

  sigma_rr + sigma_tt                = 4 Re phi'(z)
  sigma_tt - sigma_rr + 2 i sigma_rt = 2 e^{2 i theta}
                                       (conj(z) phi''(z) + psi'(z))
  2 G (u_x + i u_y)                  = kappa phi(z) - z conj(phi'(z))
                                       - conj(psi(z))

are cross-checked numerically in ``tests/test_ring_composite.py``.

Closed-form fields per zone
---------------------------
Both zones share one evaluation rule.  With zone coefficients
(A, B, C, D, F, H) meaning phi(z) = A z + B z^3 + C / z and
psi(z) = D z + F / z + H / z^3 -- liner: (a1, a3, am1, b1, bm1,
bm3); ground: (Gamma, 0, cm1, Gamma', dm1, dm3) -- the
tension-positive fields are::

  sigma_rr^T = 2 A + F / r^2 + (-4 C / r^2 - D + 3 H / r^4) cos(2 theta)
  sigma_tt^T = 2 A - F / r^2 + (12 B r^2 + D - 3 H / r^4) cos(2 theta)
  sigma_rt^T = (6 B r^2 - 2 C / r^2 + D + 3 H / r^4) sin(2 theta)

  2 G u_r^T     = (kappa - 1) A r - F / r
                  + ((kappa - 3) B r^3 + (kappa + 1) C / r
                     - D r - H / r^3) cos(2 theta)
  2 G u_theta^T = ((kappa + 3) B r^3 + (1 - kappa) C / r
                   + D r - H / r^3) sin(2 theta)

and the module returns the globally flipped quantities
(sigma = -sigma^T, u = -u^T).

Scope notes: only the in-plane components are returned (the
out-of-plane plane-strain reaction sigma_zz is not modelled); the
liner zone is valid for R_i <= r <= R_o and the ground zone for
r >= R_o (evaluating both zones at r = R_o is allowed and is how the
interface jump/continuity is inspected).
"""

import numpy as np

from .kirsch import shear_modulus

__all__ = [
    "shear_modulus",
    "kappa",
    "km_coefficients",
    "stresses",
    "displacements",
]

# Accept r at the zone edges up to a small relative rounding slack.
_R_TOL = 1e-9

_ZONES = ("liner", "ground")


def kappa(nu):
    """Kolosov constant for plane strain: kappa = 3 - 4 nu."""
    return 3.0 - 4.0 * nu


def _validate_geometry(R_i, R_o):
    if not (R_i > 0.0 and R_o > R_i):
        raise ValueError("geometry requires 0 < R_i < R_o")


def _equilibrated_solve(M, b):
    """Solve M x = b after scaling each row by its max-abs entry."""
    scale = np.max(np.abs(M), axis=1)
    return np.linalg.solve(M / scale[:, None], b / scale)


def km_coefficients(R_i, R_o, sigma_v, K0, p_i, E_l, nu_l, E_g, nu_g):
    """Kolosov-Muskhelishvili coefficients of the two-zone solution.

    Solves the nine boundary/interface conditions (module docstring)
    and returns the REAL coefficients of the tension-positive-frame
    potentials

    * liner:  phi_l = a1 z + a3 z^3 + am1 / z,
              psi_l = b1 z + bm1 / z + bm3 / z^3,
    * ground: phi_g = Gamma z + cm1 / z,
              psi_g = Gamma' z + dm1 / z + dm3 / z^3,

    as the dict keys ``a1, a3, am1, b1, bm1, bm3`` (liner) and
    ``Gamma, cm1, Gamma_prime, dm1, dm3`` (ground; the far-field
    carrying values Gamma = -(sigma_v + sigma_h)/4 and
    Gamma' = -(sigma_v - sigma_h)/2 are included for completeness).

    Parameters
    ----------
    R_i, R_o : float
        Liner inner/outer radius, 0 < R_i < R_o.
    sigma_v : float
        Vertical in-situ stress (compression positive).
    K0 : float
        Lateral pressure coefficient; sigma_h = K0 * sigma_v.
    p_i : float
        Uniform normal pressure on the inner wall (compression
        positive, zero wall shear).
    E_l, nu_l : float
        Liner Young's modulus and Poisson's ratio (plane strain).
    E_g, nu_g : float
        Ground Young's modulus and Poisson's ratio (plane strain).
    """
    _validate_geometry(R_i, R_o)
    sigma_h = K0 * sigma_v
    m = 0.5 * (sigma_v + sigma_h)
    d = 0.5 * (sigma_v - sigma_h)
    Gamma = -0.5 * m
    Gamma_p = -d
    G_l = shear_modulus(E_l, nu_l)
    G_g = shear_modulus(E_g, nu_g)
    k_l = kappa(nu_l)
    k_g = kappa(nu_g)
    g = G_l / G_g  # displacement rows scaled by 2 G_l (soft-liner safe)
    rho = R_i / R_o
    rho2 = rho * rho
    rho4 = rho2 * rho2

    # n = 0 block; unknowns (a1, bm1 / R_o^2, dm1 / R_o^2):
    #   sigma_rr(R_i) = -p_i, [sigma_rr](R_o) = 0, [u_r](R_o) = 0.
    M0 = np.array([
        [2.0, 1.0 / rho2, 0.0],
        [2.0, 1.0, -1.0],
        [k_l - 1.0, -1.0, g],
    ])
    b0 = np.array([-p_i, 2.0 * Gamma, g * (k_g - 1.0) * Gamma])
    a1, bm1_h, dm1_h = _equilibrated_solve(M0, b0)

    # n = 2 block; unknowns
    # (a3 R_o^2, am1 / R_o^2, b1, bm3 / R_o^4, cm1 / R_o^2, dm3 / R_o^4):
    #   sigma_rr(R_i) = sigma_rt(R_i) = 0,
    #   [sigma_rr](R_o) = [sigma_rt](R_o) = [u_r](R_o) = [u_theta](R_o) = 0.
    M2 = np.array([
        [0.0, -4.0 / rho2, -1.0, 3.0 / rho4, 0.0, 0.0],
        [6.0 * rho2, -2.0 / rho2, 1.0, 3.0 / rho4, 0.0, 0.0],
        [0.0, -4.0, -1.0, 3.0, 4.0, -3.0],
        [6.0, -2.0, 1.0, 3.0, 2.0, -3.0],
        [k_l - 3.0, k_l + 1.0, -1.0, -1.0, -g * (k_g + 1.0), g],
        [k_l + 3.0, 1.0 - k_l, 1.0, -1.0, -g * (1.0 - k_g), g],
    ])
    b2 = np.array([0.0, 0.0, -Gamma_p, Gamma_p, -g * Gamma_p, g * Gamma_p])
    a3_h, am1_h, b1, bm3_h, cm1_h, dm3_h = _equilibrated_solve(M2, b2)

    Ro2 = R_o * R_o
    Ro4 = Ro2 * Ro2
    return {
        "a1": float(a1),
        "a3": float(a3_h) / Ro2,
        "am1": float(am1_h) * Ro2,
        "b1": float(b1),
        "bm1": float(bm1_h) * Ro2,
        "bm3": float(bm3_h) * Ro4,
        "Gamma": float(Gamma),
        "cm1": float(cm1_h) * Ro2,
        "Gamma_prime": float(Gamma_p),
        "dm1": float(dm1_h) * Ro2,
        "dm3": float(dm3_h) * Ro4,
    }


def _zone_terms(coef, zone):
    """Generic (A, B, C, D, F, H) coefficients of the requested zone."""
    if zone == "liner":
        return (coef["a1"], coef["a3"], coef["am1"],
                coef["b1"], coef["bm1"], coef["bm3"])
    return (coef["Gamma"], 0.0, coef["cm1"],
            coef["Gamma_prime"], coef["dm1"], coef["dm3"])


def _prepare(r, theta, R_i, R_o, zone):
    """Broadcast (r, theta) to floats and validate r against the zone."""
    if zone not in _ZONES:
        raise ValueError("zone must be 'liner' or 'ground'")
    r, theta = np.broadcast_arrays(
        np.asarray(r, dtype=float), np.asarray(theta, dtype=float)
    )
    if zone == "liner":
        if (np.any(r < R_i * (1.0 - _R_TOL))
                or np.any(r > R_o * (1.0 + _R_TOL))):
            raise ValueError("liner zone is defined for R_i <= r <= R_o")
    elif np.any(r < R_o * (1.0 - _R_TOL)):
        raise ValueError("ground zone is defined for r >= R_o")
    return r, theta


def stresses(r, theta, R_i, R_o, sigma_v, K0, p_i, E_l, nu_l, E_g, nu_g,
             zone):
    """Total polar stress components (compression positive).

    Parameters
    ----------
    r, theta : array_like
        Polar coordinates (broadcast together); R_i <= r <= R_o for
        ``zone="liner"``, r >= R_o for ``zone="ground"``.
    R_i, R_o, sigma_v, K0, p_i, E_l, nu_l, E_g, nu_g : float
        Geometry, loading and elastic constants as in
        :func:`km_coefficients`.
    zone : str
        "liner" or "ground".

    Returns
    -------
    (sigma_rr, sigma_tt, sigma_rt) : tuple of ndarray
        Radial, hoop and shear components at the broadcast shape of
        (r, theta); ``sigma_rr = p_i`` and ``sigma_rt = 0`` at
        ``r = R_i``, and the far field is recovered as r -> infinity.
    """
    coef = km_coefficients(R_i, R_o, sigma_v, K0, p_i, E_l, nu_l, E_g, nu_g)
    r, theta = _prepare(r, theta, R_i, R_o, zone)
    A, B, C, D, F, H = _zone_terms(coef, zone)
    c2 = np.cos(2.0 * theta)
    s2 = np.sin(2.0 * theta)
    inv2 = 1.0 / (r * r)
    inv4 = inv2 * inv2
    r2 = r * r
    s_rr = 2.0 * A + F * inv2 + (-4.0 * C * inv2 - D + 3.0 * H * inv4) * c2
    s_tt = 2.0 * A - F * inv2 + (12.0 * B * r2 + D - 3.0 * H * inv4) * c2
    s_rt = (6.0 * B * r2 - 2.0 * C * inv2 + D + 3.0 * H * inv4) * s2
    return -s_rr, -s_tt, -s_rt


def displacements(r, theta, R_i, R_o, sigma_v, K0, p_i, E_l, nu_l, E_g, nu_g,
                  zone):
    """Total displacements of the wished-in-place loading (plane strain).

    Reference state and rigid-body gauge per the module docstring:
    displacement from the stress-free unloaded composite to the
    loaded equilibrium state, with no rigid-body motion (mirror
    symmetries); it grows like the homogeneous far-field straining as
    r -> infinity.  Sign convention: u_r > 0 toward the tunnel axis
    (convergence), u_theta > 0 toward decreasing theta.

    Parameters as in :func:`stresses`.

    Returns
    -------
    (u_r, u_theta) : tuple of ndarray
        Components at the broadcast shape of (r, theta).
    """
    coef = km_coefficients(R_i, R_o, sigma_v, K0, p_i, E_l, nu_l, E_g, nu_g)
    r, theta = _prepare(r, theta, R_i, R_o, zone)
    A, B, C, D, F, H = _zone_terms(coef, zone)
    if zone == "liner":
        G, k = shear_modulus(E_l, nu_l), kappa(nu_l)
    else:
        G, k = shear_modulus(E_g, nu_g), kappa(nu_g)
    c2 = np.cos(2.0 * theta)
    s2 = np.sin(2.0 * theta)
    inv1 = 1.0 / r
    inv3 = inv1 / (r * r)
    r3 = r * r * r
    pref = 1.0 / (2.0 * G)
    u_r = pref * ((k - 1.0) * A * r - F * inv1
                  + ((k - 3.0) * B * r3 + (k + 1.0) * C * inv1
                     - D * r - H * inv3) * c2)
    u_t = pref * (((k + 3.0) * B * r3 + (1.0 - k) * C * inv1
                   + D * r - H * inv3) * s2)
    return -u_r, -u_t
