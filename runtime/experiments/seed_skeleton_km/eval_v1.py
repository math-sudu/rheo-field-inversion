
import math

import numpy as np

from refsol import kirsch

__all__ = [
    "annulus_grid",
    "rel_l2",
    "rel_max",
    "stress_error_vs_kirsch",
    "far_field_deviation",
    "disp_error_vs_kirsch",
]


R_OUTER_FACTOR = 5.0


def annulus_grid(a, n_r=101, n_theta=256):
    if not a > 0.0:
        raise ValueError("hole radius a must be positive")
    if n_r < 2:
        raise ValueError("n_r must be >= 2 (trapezoidal rule in r)")
    if n_theta < 1:
        raise ValueError("n_theta must be >= 1")
    r_1d = np.linspace(a, R_OUTER_FACTOR * a, n_r)
    dr = (R_OUTER_FACTOR * a - a) / (n_r - 1)
    w_r = np.full(n_r, dr)
    w_r[0] = 0.5 * dr
    w_r[-1] = 0.5 * dr
    theta_1d = (2.0 * np.pi / n_theta) * np.arange(n_theta)
    r, theta = np.meshgrid(r_1d, theta_1d, indexing="ij")
    w = r * (w_r[:, None] * (2.0 * np.pi / n_theta))
    return r, theta, w


def _components(fields, name):
    """Convert a sequence of field components to float ndarrays."""
    comps = [np.asarray(f, dtype=float) for f in fields]
    if not comps:
        raise ValueError(f"{name} must contain at least one component")
    return comps


def rel_l2(fields, ref_fields, w):
    """Relative weighted L2 error of a stacked vector field.

    Parameters
    ----------
    fields, ref_fields : sequence of array_like
        Same-length sequences of same-shape component arrays (e.g.
        the 3 stress components), treated as ONE stacked vector
        field: predicted and reference respectively.
    w : array_like
        Quadrature weights, same shape as every component (e.g. the
        ``w`` returned by :func:`annulus_grid`).

    Returns
    -------
    float
        sqrt(sum over components and nodes of w |f - g|^2) divided by
        the same expression with the reference g alone.

    Raises
    ------
    ValueError
        On component-count or shape mismatch, or when the reference
        has zero weighted norm (relative error undefined).
    """
    f_comps = _components(fields, "fields")
    g_comps = _components(ref_fields, "ref_fields")
    if len(f_comps) != len(g_comps):
        raise ValueError(
            "fields and ref_fields must have the same number of components"
        )
    w = np.asarray(w, dtype=float)
    num = 0.0
    den = 0.0
    for f, g in zip(f_comps, g_comps):
        if f.shape != w.shape or g.shape != w.shape:
            raise ValueError(
                "every field component must have the same shape as w"
            )
        d = f - g
        num += float(np.sum(w * d * d))
        den += float(np.sum(w * g * g))
    if den == 0.0:
        raise ValueError("reference field has zero weighted L2 norm")
    return math.sqrt(num) / math.sqrt(den)


def rel_max(fields, ref_fields):
    """Relative max (worst-node) error of a stacked vector field.

    Parameters
    ----------
    fields, ref_fields : sequence of array_like
        Same-length sequences of same-shape component arrays,
        predicted and reference respectively.

    Returns
    -------
    float
        max over all components and nodes of |f - g|, divided by max
        over all components and nodes of |g|.

    Raises
    ------
    ValueError
        On component-count or shape mismatch, or when the reference
        is identically zero (relative error undefined).
    """
    f_comps = _components(fields, "fields")
    g_comps = _components(ref_fields, "ref_fields")
    if len(f_comps) != len(g_comps):
        raise ValueError(
            "fields and ref_fields must have the same number of components"
        )
    num = 0.0
    den = 0.0
    for f, g in zip(f_comps, g_comps):
        if f.shape != g.shape:
            raise ValueError(
                "paired field components must have the same shape"
            )
        # NaN must POISON the metric, not be swallowed: Python's
        # max(0.0, nan) keeps 0.0 (all nan comparisons are False), so
        # a diverged all-NaN field would otherwise report 0.0 error.
        comp_num = float(np.max(np.abs(f - g)))
        comp_den = float(np.max(np.abs(g)))
        if np.isnan(comp_num) or np.isnan(num):
            num = float("nan")
        else:
            num = max(num, comp_num)
        if np.isnan(comp_den) or np.isnan(den):
            den = float("nan")
        else:
            den = max(den, comp_den)
    if den == 0.0:
        raise ValueError("reference field has zero max norm")
    return num / den


def stress_error_vs_kirsch(sigma_rr, sigma_tt, sigma_rt, a, sigma_v, K0,
                           n_r=101, n_theta=256):
    r, theta, w = annulus_grid(a, n_r=n_r, n_theta=n_theta)
    pred = _components((sigma_rr, sigma_tt, sigma_rt), "fields")
    for c in pred:
        if c.shape != w.shape:
            raise ValueError(
                "predicted components must be evaluated on "
                f"annulus_grid(a, n_r={n_r}, n_theta={n_theta}) nodes "
                f"(expected shape {w.shape}, got {c.shape})"
            )
    ref = kirsch.stresses(r, theta, a, sigma_v, K0, part="total")
    return {"rel_l2": rel_l2(pred, ref, w), "rel_max": rel_max(pred, ref)}


def far_field_deviation(stress_fn, a, sigma_v, K0, radii=(20.0, 50.0),
                        n_theta=256):
    """Deviation of predicted TOTAL stresses from in-situ, far away.

    For each radius multiple ``m`` in ``radii`` the callable is
    evaluated on the full circle ``r = m a`` (angles ``theta_k = 2 pi
    k / n_theta``) and compared with the undisturbed in-situ field
    ``refsol.kirsch.stresses(..., part="insitu")``.  Pure
    measurement -- no pass/fail logic.

    Calibration note: for the EXACT Kirsch total field the deviation
    at ``r = m a`` is the true excavation perturbation, of order
    ``(1/m)^2`` (about 4.4e-3 at m = 20 and 7.0e-4 at m = 50 for
    K0 = 0.5), so thresholds applied to the returned values must be
    radius-aware.

    Parameters
    ----------
    stress_fn : callable
        ``stress_fn(r, theta) -> (sigma_rr, sigma_tt, sigma_rt)``,
        TOTAL compression-positive stresses, vectorized over
        same-shape arrays ``r`` and ``theta``.
    a : float
        Tunnel radius (> 0).
    sigma_v : float
        Vertical in-situ stress (compression positive).
    K0 : float
        Lateral pressure coefficient; sigma_h = K0 * sigma_v.
    radii : sequence of float
        Radius multiples m; each circle is r = m * a.
    n_theta : int
        Number of uniform angular nodes per circle (>= 1).

    Returns
    -------
    dict
        ``{m: max over theta and components of |pred - insitu| /
        max(|sigma_v|, |K0 * sigma_v|)}``, keyed by the elements of
        ``radii`` exactly as given.
    """
    if not a > 0.0:
        raise ValueError("hole radius a must be positive")
    if n_theta < 1:
        raise ValueError("n_theta must be >= 1")
    scale = max(abs(sigma_v), abs(K0 * sigma_v))
    if scale == 0.0:
        raise ValueError(
            "in-situ stress magnitude is zero; deviation is undefined"
        )
    theta = (2.0 * np.pi / n_theta) * np.arange(n_theta)
    out = {}
    for mult in radii:
        r = np.full(theta.shape, float(mult) * a)
        pred = _components(stress_fn(r, theta), "stress_fn output")
        if len(pred) != 3:
            raise ValueError(
                "stress_fn must return (sigma_rr, sigma_tt, sigma_rt)"
            )
        insitu = kirsch.stresses(r, theta, a, sigma_v, K0, part="insitu")
        dev = 0.0
        for p, i in zip(pred, insitu):
            if p.shape != theta.shape:
                raise ValueError(
                    "stress_fn components must match the theta shape "
                    f"(expected {theta.shape}, got {p.shape})"
                )
            dev = max(dev, float(np.max(np.abs(p - i))))
        out[mult] = dev / scale
    return out


def disp_error_vs_kirsch(u_r, u_theta, a, sigma_v, K0, E, nu,
                         n_r=101, n_theta=256):
    r, theta, w = annulus_grid(a, n_r=n_r, n_theta=n_theta)
    pred = _components((u_r, u_theta), "fields")
    for c in pred:
        if c.shape != w.shape:
            raise ValueError(
                "predicted components must be evaluated on "
                f"annulus_grid(a, n_r={n_r}, n_theta={n_theta}) nodes "
                f"(expected shape {w.shape}, got {c.shape})"
            )
    ref = kirsch.displacements(r, theta, a, sigma_v, K0, E, nu)
    return {"rel_l2": rel_l2(pred, ref, w), "rel_max": rel_max(pred, ref)}
