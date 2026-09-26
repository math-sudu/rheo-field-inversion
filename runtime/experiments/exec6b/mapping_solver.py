"""Classical mapping-coefficient solver (map acquisition).

Problem
-------
Given an ordered CLOSED boundary point set of a tunnel cross-section
(counterclockwise, complex points z_k, arbitrary spacing), determine
the constants (R, c) of the truncated exterior conformal map of the
family convention (source paper Eq. 9, anchor
author reference archive: pid_3bd1b31e158b/full.md line 235)::

  z = omega(zeta) = R (zeta + sum_{j=0..n_map} c_j zeta^{-j}),
  |zeta| >= 1,

with R > 0 real and c complex (c_0 a pure translation).  This is the
"map acquisition" step of the classical complex-variable pipeline:
``general_map.GeneralMappedKM`` CONSUMES (R, c) as given data; this
module PRODUCES them from boundary geometry.

Flow provenance (verified by the driver)
----------------------------------------
The control-point flow -- representative boundary control points
selected counterclockwise, mapping equation as objective, solve for
the unknown constants including the correspondence angles
(author reference archive: pid_afcb9bac3a70/full.md lines 81 and
95); least-squares solving of the resulting overdetermined systems is
standard in this pipeline
(author reference archive: pid_f29ccd500323/full.md line 294).

Gauge (well-posedness)
----------------------
For a Jordan boundary the exterior Riemann map with omega(inf)=inf
and omega'(inf)=R>0 is UNIQUE, so (R, c) recovery from exact in-class
data is well-posed (no residual rotation freedom).  Concretely: the
reparameterization omega(e^{i alpha} zeta) traces the same boundary
but has leading coefficient R e^{i alpha}, which leaves the model
class unless alpha = 0 -- pinning the leading coefficient real
positive is exactly what removes the rotation gauge, and the
correspondence angles theta_k absorb the parameter shift instead.

Unknowns and objective
----------------------
Unknowns: R > 0, complex B_j := R c_j (j = 0..n_coeffs), and one
correspondence angle theta_k per input point.  Objective (the
"mapping equation as objective" of the flow above)::

  minimize sum_k | z_k - [R e^{i theta_k}
                          + sum_j B_j e^{-i j theta_k}] |^2 .

The model is LINEAR in (R, Re B, Im B) at fixed theta and each
residual depends on ONE theta_k -- a separable nonlinear
least-squares problem.

Algorithm (chosen; two stages, pure numpy)
------------------------------------------
Data are first normalized: z_n = (z - mean z) / s with s =
max_k |z_k - mean z| (the section half-diameter); the affine change
maps the model class onto itself exactly (R -> R/s, B_0 ->
(B_0 - mean z)/s, B_j -> B_j/s), so the solution is transformed back
exactly at the end and all tolerances below live in the normalized
frame (unit half-diameter).

1. Alternating warm-up (block coordinate descent): theta initialized
   as arg(z_n) (angles about the centroid); then alternate (a) the
   LINEAR least squares for (R, Re B, Im B) at fixed theta
   (``numpy.linalg.lstsq`` on the 2M x (2 n_coeffs + 3) real system)
   with (b) per-point Newton projection sweeps updating each theta_k
   toward the nearest-point preimage (guarded: Gauss-Newton fallback
   step Re(conj(w_t) r)/|w_t|^2 when the Newton curvature is weak or
   negative, per-sweep step clamp).  Monotone decrease, globally
   stable, linear rate -- used only to enter the joint basin.
2. Joint Gauss-Newton polish on ALL unknowns (R, Re B, Im B, theta)
   with backtracking line search: the dense Jacobian is the linear
   design block next to the diagonal theta block; the step solves
   ``lstsq(J, -res)``.  For exact in-class data the problem is a
   ZERO-RESIDUAL nonlinear LS, where Gauss-Newton converges locally
   QUADRATICALLY -- machine-precision recovery of (R, c) in a
   handful of iterations; near small-residual minima (out-of-class
   boundaries, noisy data) the local rate is linear with contraction
   proportional to the residual-curvature product (small here), and
   the line search keeps every accepted step a strict decrease.

Termination / ``converged`` semantics (normalized frame): converged
is True when (i) max_k |r_k| <= tol_fit (residual floor: exact-fit
class), or (ii) the joint gradient satisfies ||J^T res||_inf <=
tol_grad (stationary point of the LS objective: the expected exit
for out-of-class or noisy data), or (iii) the decrease has hit the
numerical floor (backtracking cannot decrease, or two consecutive
relative decreases <= 1e-15) while the gradient is already small
(<= 1e-6).  Exhausting max_iter_gn while still improving returns
converged=False, as does a nonpositive recovered R.

Acceptance metrics (plainly)
----------------------------
* Recovery metric (in-class ground-truth (R*, c*) tests): the solve
  is accepted when |R_rec - R*| <= 1e-8 * max(R*, 1) and, ELEMENTWISE,
  |c_rec_j - c*_j| <= 1e-8 * max(||c*||_inf, 1) -- an absolute
  tolerance anchored to the largest true coefficient (documented
  exact metric of tests b1/b2).
* Geometric misfit: ``fit_max_dist`` = max_k |z_k -
  omega(e^{i theta_k})| (original units) and ``fit_rel_max_dist`` =
  the same normalized by the section half-diameter
  max_k |z_k - mean z| (identical to the normalization scale s, so
  it equals the normalized-frame max residual).
* Map acceptance: every solve calls
  ``general_map.validate_univalence(R, c)`` on the result and
  returns the full gate dict under key ``univalence``.

``wall_s`` measures the optimization span (validation + warm-up +
Gauss-Newton); the univalence acceptance scan is timed separately by
callers if needed.  Pure numpy + stdlib (no torch).
"""

import os
import sys
import time

import numpy as np

# Flat-module import shim (mirrors general_map.py / the frozen seed):
# this experiment directory is a flat module dir, not a package.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import general_map  # noqa: E402  (committed, read-only dependency)

__all__ = [
    "solve_mapping",
    "section_three_centered_arch",
    "sample_map_boundary",
]


# ----------------------------------------------------------------------
# model evaluation helpers (boundary restriction of the map)
# ----------------------------------------------------------------------

def _boundary_values(R, B, theta):
    """w, w_theta, w_thetatheta of w(theta) = R e^{i theta} + sum_j
    B_j e^{-i j theta} (the boundary restriction of omega and its
    theta-derivatives; each e^{-i j theta} derivative multiplies by
    -i j)."""
    theta = np.asarray(theta, dtype=float)
    n = B.size - 1
    jj = np.arange(n + 1)
    E = np.exp(-1j * np.outer(theta, jj))       # (M, n+1)
    e1 = np.exp(1j * theta)
    w = R * e1 + E @ B
    wt = 1j * (R * e1) - 1j * (E @ (jj * B))
    wtt = -(R * e1) - E @ (jj * jj * B)
    return w, wt, wtt


def _design_matrix(theta, n):
    """Real 2M x (2n+3) design matrix A with A u = [Re w; Im w] for
    u = [R, Re B_0..B_n, Im B_0..B_n] (model linear in u at fixed
    theta): B_j e^{-i j theta} = [p cos + q sin] + i [q cos - p sin]
    for B_j = p + i q."""
    theta = np.asarray(theta, dtype=float)
    m = theta.size
    jj = np.arange(n + 1)
    E = np.exp(-1j * np.outer(theta, jj))  # E.real = cos, E.imag = -sin
    a = np.empty((2 * m, 2 * n + 3))
    a[:m, 0] = np.cos(theta)
    a[m:, 0] = np.sin(theta)
    a[:m, 1:n + 2] = E.real
    a[m:, 1:n + 2] = E.imag
    a[:m, n + 2:] = -E.imag
    a[m:, n + 2:] = E.real
    return a


def _linear_ls(zn, theta, n):
    """Linear least squares for (R, B) at fixed theta."""
    a = _design_matrix(theta, n)
    rhs = np.concatenate([zn.real, zn.imag])
    u, _, _, _ = np.linalg.lstsq(a, rhs, rcond=None)
    return float(u[0]), u[1:n + 2] + 1j * u[n + 2:]


def _project_theta(R, B, theta, zn, sweeps, clamp=0.3):
    """Per-point correspondence update: Newton sweeps on f(theta) =
    |z_k - w(theta)|^2 with Gauss-Newton fallback (dtheta =
    Re(conj(w_t) r) / |w_t|^2) when the Newton curvature f''/2 =
    |w_t|^2 - Re(conj(r) w_tt) is below 0.25 |w_t|^2, and a per-sweep
    step clamp (vectorized over all points)."""
    for _ in range(sweeps):
        w, wt, wtt = _boundary_values(R, B, theta)
        r = zn - w
        num = np.real(np.conj(r) * wt)
        wt2 = np.abs(wt) ** 2
        den = wt2 - np.real(np.conj(r) * wtt)
        den = np.where(den > 0.25 * wt2, den, wt2)
        step = np.clip(num / den, -clamp, clamp)
        theta = theta + step
    return theta


# ----------------------------------------------------------------------
# input validation (Jordan-boundary gate)
# ----------------------------------------------------------------------

def _validate_boundary(points, n_coeffs):
    """Cast/validate the input point set; return (z, center, scale).

    Raises ValueError on: non-1-D or nonfinite input; too few points
    (M < max(2 n_coeffs + 3, 8): the count where the free-theta
    system 2M >= (2 n_coeffs + 3) + M stops being overdetermined,
    floored at 8); zero half-diameter; near-zero enclosed area
    (degenerate input, e.g. all collinear -- the shoelace area test);
    clockwise orientation (counterclockwise required); self-
    intersecting input polygon (not a Jordan boundary -- reuses the
    committed proper-crossing test ``general_map._polygon_is_simple``,
    the family's single definition of polygon simplicity).  An exact
    closing duplicate z[-1] == z[0] is dropped silently.
    """
    z = np.asarray(points, dtype=complex)
    if z.ndim != 1:
        raise ValueError(
            f"boundary_points must be 1-D complex, got shape {z.shape}"
        )
    if not np.all(np.isfinite(z.real) & np.isfinite(z.imag)):
        raise ValueError("boundary_points must be finite")
    if z.size >= 2 and z[-1] == z[0]:
        z = z[:-1]
    m_min = max(2 * n_coeffs + 3, 8)
    if z.size < m_min:
        raise ValueError(
            f"too few points: got {z.size}, need >= {m_min} for "
            f"n_coeffs = {n_coeffs} (free-theta system must stay "
            "overdetermined)"
        )
    center = complex(np.mean(z))
    scale = float(np.max(np.abs(z - center)))
    if scale <= 0.0:
        raise ValueError("degenerate input: all points coincide")
    area = 0.5 * float(np.sum(np.imag(np.conj(z) * np.roll(z, -1))))
    if abs(area) <= 1e-9 * scale * scale:
        raise ValueError(
            "degenerate input: near-zero enclosed area (open or "
            "collinear point set is not a Jordan boundary)"
        )
    if area < 0.0:
        raise ValueError(
            "boundary_points are ordered clockwise; counterclockwise "
            "ordering is required"
        )
    if not general_map._polygon_is_simple(z):
        raise ValueError(
            "boundary_points self-intersect; a Jordan (simple closed) "
            "boundary is required"
        )
    return z, center, scale


# ----------------------------------------------------------------------
# public solver
# ----------------------------------------------------------------------

def solve_mapping(boundary_points, n_coeffs, max_iter_alternating=30,
                  max_iter_gn=100, projection_sweeps=3, tol_fit=1e-14,
                  tol_grad=1e-11):
    """Solve the map-acquisition problem for one boundary point set.

    Parameters
    ----------
    boundary_points : array_like, complex, shape (M,)
        Ordered counterclockwise points of a closed Jordan boundary
        (arbitrary spacing; an exact closing duplicate is dropped).
    n_coeffs : int
        Highest inverse power n_map of the map; c has n_coeffs + 1
        entries (c_0 = translation).
    max_iter_alternating : int
        Warm-up rounds of (linear LS <-> theta projection).
    max_iter_gn : int
        Joint Gauss-Newton iteration cap.
    projection_sweeps : int
        Newton sweeps per warm-up round.
    tol_fit : float
        Residual-floor exit: max normalized point misfit (fraction of
        the section half-diameter) declared an exact fit.
    tol_grad : float
        Stationarity exit on ||J^T res||_inf (normalized frame).

    Returns
    -------
    dict
        ``R`` (float, > 0 on success), ``c`` (complex ndarray,
        n_coeffs + 1), ``theta`` (correspondence angles, one per
        input point, wrapped to [0, 2 pi)), ``fit_max_dist`` /
        ``fit_rel_max_dist`` (geometric misfit, module docstring),
        ``n_iter`` (warm-up rounds + GN iterations), ``wall_s``
        (optimization span), ``converged`` (bool, semantics in the
        module docstring), ``termination`` (exit label:
        residual_floor / stationary / decrease_floor /
        linesearch_floor / max_iter / nonpositive_R), and
        ``univalence`` = ``general_map.validate_univalence(R, c)``
        on the result (``univalent_ok`` forced False on a
        nonpositive-R failure, where the map is not evaluable).

    Raises
    ------
    ValueError
        On non-Jordan input per :func:`_validate_boundary` (too few
        points, degenerate/collinear, clockwise, self-intersecting).
    """
    n_coeffs = int(n_coeffs)
    if n_coeffs < 0:
        raise ValueError("n_coeffs must be >= 0")
    z, center, scale = _validate_boundary(boundary_points, n_coeffs)
    t0 = time.perf_counter()
    zn = (z - center) / scale
    m = zn.size
    n = n_coeffs
    n_lin = 2 * n + 3

    # ---- stage 1: alternating warm-up --------------------------------
    theta = np.angle(zn)
    n_iter = 0
    R, B = 1.0, np.zeros(n + 1, dtype=complex)
    for _ in range(max_iter_alternating):
        R, B = _linear_ls(zn, theta, n)
        theta = _project_theta(R, B, theta, zn, projection_sweeps)
        n_iter += 1
        w, _, _ = _boundary_values(R, B, theta)
        if float(np.max(np.abs(zn - w))) <= tol_fit:
            break

    # ---- stage 2: joint Gauss-Newton polish ---------------------------
    def residual(R_, B_, theta_):
        w_, _, _ = _boundary_values(R_, B_, theta_)
        r_ = zn - w_
        return np.concatenate([r_.real, r_.imag])

    res = residual(R, B, theta)
    f = float(res @ res)
    termination = "max_iter"
    converged = False
    stall = 0
    for _ in range(max_iter_gn):
        w, wt, _ = _boundary_values(R, B, theta)
        r = zn - w
        res = np.concatenate([r.real, r.imag])
        f = float(res @ res)
        if float(np.max(np.abs(r))) <= tol_fit:
            termination, converged = "residual_floor", True
            break
        jac = np.zeros((2 * m, n_lin + m))
        jac[:, :n_lin] = -_design_matrix(theta, n)
        rows = np.arange(m)
        jac[rows, n_lin + rows] = -wt.real
        jac[m + rows, n_lin + rows] = -wt.imag
        grad_inf = float(np.max(np.abs(jac.T @ res)))
        if grad_inf <= tol_grad:
            termination, converged = "stationary", True
            break
        dp, _, _, _ = np.linalg.lstsq(jac, -res, rcond=None)
        alpha, accepted = 1.0, False
        for _ in range(30):
            R2 = R + alpha * dp[0]
            B2 = B + alpha * (dp[1:n + 2] + 1j * dp[n + 2:n_lin])
            th2 = theta + alpha * dp[n_lin:]
            res2 = residual(R2, B2, th2)
            f2 = float(res2 @ res2)
            if f2 < f:
                R, B, theta = R2, B2, th2
                accepted = True
                break
            alpha *= 0.5
        n_iter += 1
        if not accepted:
            # Backtracking cannot decrease: numerical floor of the
            # objective; a stationary point in exact arithmetic when
            # the gradient is already small.
            converged = grad_inf <= 1e-6
            termination = ("linesearch_floor" if converged
                           else "linesearch_failed")
            break
        if f - f2 <= 1e-15 * max(f, 1e-300):
            stall += 1
            if stall >= 2:
                converged = grad_inf <= 1e-6
                termination = ("decrease_floor" if converged
                               else "decrease_stalled")
                break
        else:
            stall = 0

    # ---- un-normalize and assemble ------------------------------------
    w, _, _ = _boundary_values(R, B, theta)
    fit_rel = float(np.max(np.abs(zn - w)))
    wall_s = time.perf_counter() - t0
    R_out = scale * R
    B_out = scale * B
    B_out = B_out.copy()
    B_out[0] += center
    if R_out <= 0.0:
        converged = False
        termination = "nonpositive_R"
        c_out = np.full(n + 1, np.nan, dtype=complex)
        univalence = {"univalent_ok": False,
                      "reason": "nonpositive recovered R"}
    else:
        c_out = B_out / R_out
        univalence = general_map.validate_univalence(R_out, c_out)
    return {
        "R": R_out,
        "c": c_out,
        "theta": np.mod(theta, 2.0 * np.pi),
        "fit_max_dist": fit_rel * scale,
        "fit_rel_max_dist": fit_rel,
        "n_iter": n_iter,
        "wall_s": wall_s,
        "converged": bool(converged),
        "termination": termination,
        "univalence": univalence,
    }


# ----------------------------------------------------------------------
# deterministic in-class boundary sampler (test/runner fixture helper)
# ----------------------------------------------------------------------

def sample_map_boundary(R, c, n_points, warp_amplitude=0.25,
                        warp_frequency=3, warp_phase=0.7):
    """Boundary samples of a GIVEN map at deterministic NON-uniform
    angles.

    theta*_k = 2 pi k / M + warp_amplitude * sin(warp_frequency *
    2 pi k / M + warp_phase): a smooth injective warp of the uniform
    grid (monotone whenever warp_amplitude * warp_frequency < 1),
    counterclockwise, no two points coincide.  Returns (points,
    theta_true).
    """
    if warp_amplitude * warp_frequency >= 1.0:
        raise ValueError(
            "warp_amplitude * warp_frequency must be < 1 to keep the "
            "sample angles monotone (ordered counterclockwise)"
        )
    tt = (2.0 * np.pi / n_points) * np.arange(n_points)
    theta = tt + warp_amplitude * np.sin(warp_frequency * tt
                                         + warp_phase)
    c = np.asarray(c, dtype=complex)
    probe = general_map.GeneralMappedKM(R, c, sigma_v=1.0, K0=0.0,
                                        n_order=1)
    return probe.omega(np.exp(1j * theta)), theta


# ----------------------------------------------------------------------
# three-centered-arch section generator (C1 arc chain)
# ----------------------------------------------------------------------

def section_three_centered_arch(width=10.0, height=8.0, n_points=360,
                                knee_radius_ratio=0.18,
                                haunch_radius_ratio=0.35,
                                breakpoints_deg=(25.0, 60.0, 105.0,
                                                 145.0)):
    """Smooth curved-wall horseshoe profile: three-centered arch roof
    (crown arc + two haunch arcs = three centers) over curved walls,
    knee fillets and a curved invert -- the standard curved-wall
    tunnel section built ENTIRELY from circular arcs with C1
    (tangent-continuous) joins.

    Geometry (turning-angle construction).  A convex C1 closed curve
    traversed counterclockwise is fixed by its radius of curvature
    rho(psi) as a function of the tangent direction angle psi in
    [0, 2 pi) via dz/dpsi = rho(psi) e^{i psi}; PIECEWISE-CONSTANT
    rho makes every piece an exact circular arc and every join
    tangent-continuous BY CONSTRUCTION (psi is continuous).  psi = 0
    at the invert bottom, pi/2 at the right springline, pi at the
    crown apex; the profile is mirror-symmetric (rho even in psi).
    Right-half segments between ``breakpoints_deg`` = (psi_1..psi_4)::

      [0,     psi_1]  invert   rho_v  (solved)
      [psi_1, psi_2]  knee     rho_k = knee_radius_ratio * width
      [psi_2, psi_3]  wall     rho_w  (solved)
      [psi_3, psi_4]  haunch   rho_h = haunch_radius_ratio * width
      [psi_4, pi   ]  crown    rho_c  (solved)

    The three free radii (rho_v, rho_w, rho_c) solve the 3 x 3 LINEAR
    system (all integrals of rho cos psi / rho sin psi are linear in
    the segment radii):

    * closure  int_0^pi rho cos psi dpsi = 0  (right half starts and
      ends on the symmetry axis; the mirrored curve then closes
      exactly),
    * height   int_0^pi rho sin psi dpsi = height,
    * width    2 int_0^{pi/2} rho cos psi dpsi = width.

    Solved radii must all be positive, else the requested aspect is
    outside this parameterization's range and ValueError is raised.
    Defaults (width 10, height 8) give rho_v/rho_w/rho_c ~
    8.52 / 4.50 / 6.05: a realistic shallow invert, curved walls and
    a crown flatter than the haunches (three-centered arch).  The
    returned M points are ordered counterclockwise, uniformly spaced
    in ARC LENGTH along the closed curve, starting at the invert
    bottom (placed at the origin, apex at (0, height)); the curve is
    closed and C1 (no corner discontinuities) by construction.
    """
    if width <= 0.0 or height <= 0.0:
        raise ValueError("width and height must be positive")
    if n_points < 3:
        raise ValueError("n_points must be >= 3")
    psi_b = np.deg2rad(np.asarray(breakpoints_deg, dtype=float))
    if psi_b.shape != (4,) or not np.all(np.diff(psi_b) > 0.0) \
            or psi_b[0] <= 0.0 or psi_b[-1] >= np.pi:
        raise ValueError(
            "breakpoints_deg must be 4 strictly increasing angles in "
            "(0, 180)"
        )
    rho_k = knee_radius_ratio * width
    rho_h = haunch_radius_ratio * width
    if rho_k <= 0.0 or rho_h <= 0.0:
        raise ValueError("radius ratios must be positive")
    edges = np.concatenate([[0.0], psi_b, [np.pi]])  # 5 segments

    def i_cos(lo, hi):
        return np.sin(hi) - np.sin(lo)

    def i_sin(lo, hi):
        return np.cos(lo) - np.cos(hi)

    def i_cos_half(lo, hi):  # clipped to [0, pi/2] for the width row
        lo, hi = min(lo, np.pi / 2.0), min(hi, np.pi / 2.0)
        return np.sin(hi) - np.sin(lo)

    # unknown order: (rho_v, rho_w, rho_c) = segments 0, 2, 4
    seg_lo, seg_hi = edges[:-1], edges[1:]
    a_mat = np.zeros((3, 3))
    b_vec = np.zeros(3)
    known = {1: rho_k, 3: rho_h}
    col = {0: 0, 2: 1, 4: 2}
    rows = (
        (i_cos, 0.0),
        (i_sin, height),
        (lambda lo, hi: 2.0 * i_cos_half(lo, hi), width),
    )
    for ri, (integ, target) in enumerate(rows):
        b_vec[ri] = target
        for si in range(5):
            v = integ(seg_lo[si], seg_hi[si])
            if si in col:
                a_mat[ri, col[si]] = v
            else:
                b_vec[ri] -= known[si] * v
    rho_v, rho_w, rho_c = np.linalg.solve(a_mat, b_vec)
    if min(rho_v, rho_w, rho_c) <= 0.0:
        raise ValueError(
            "requested width/height aspect is outside this "
            f"parameterization's range (solved radii {rho_v:.4g}, "
            f"{rho_w:.4g}, {rho_c:.4g} must all be positive)"
        )
    # full-circle segment table (right half then mirrored left half)
    rho_half = np.array([rho_v, rho_k, rho_w, rho_h, rho_c])
    rho_all = np.concatenate([rho_half, rho_half[::-1]])
    edges_all = np.concatenate([edges, 2.0 * np.pi - edges[-2::-1]])
    # breakpoint positions z_i by the closed-form arc primitive
    # z(psi) = z_i - i rho (e^{i psi} - e^{i psi_i})
    z_break = np.zeros(11, dtype=complex)
    for si in range(10):
        z_break[si + 1] = z_break[si] - 1j * rho_all[si] * (
            np.exp(1j * edges_all[si + 1]) - np.exp(1j * edges_all[si])
        )
    seg_len = rho_all * np.diff(edges_all)
    s_cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    s_samples = (s_cum[-1] / n_points) * np.arange(n_points)
    si = np.clip(np.searchsorted(s_cum, s_samples, side="right") - 1,
                 0, 9)
    psi = edges_all[si] + (s_samples - s_cum[si]) / rho_all[si]
    return z_break[si] - 1j * rho_all[si] * (
        np.exp(1j * psi) - np.exp(1j * edges_all[si])
    )
