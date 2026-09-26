"""Versioned interpolation of complete committed FE observation histories.

PCHIP is C1 in query time and preserves monotonicity between samples. Its
sample-value derivatives are piecewise smooth because the shape-preserving
slope limiter can change branch. It does not smooth the constitutive law,
alter the FE path, move observations, or establish a construction zero.
"""
from __future__ import annotations

import hashlib
import inspect
from pathlib import Path

import numpy as np
import scipy
from scipy.interpolate import PchipInterpolator
import torch


METHODS = ("linear", "pchip")
VERSIONS = {"linear": "committed-history-linear-v1",
            "pchip": "committed-history-shape-preserving-c1-v1"}


def interpolation_identity(method):
    """Record both local AD implementation and the installed NumPy backend."""
    if method not in METHODS:
        raise ValueError("Time interpolation must be explicitly linear or pchip")
    return {"method": method, "version": VERSIONS[method],
            "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "numpy_backend": "scipy.interpolate.PchipInterpolator" if method == "pchip" else "linear",
            "scipy_version": scipy.__version__,
            "scipy_cubic_code_sha256": hashlib.sha256(
                Path(inspect.getfile(PchipInterpolator)).read_bytes()).hexdigest(),
            "extrapolation": "forbidden", "samples": "all committed FE increments",
            "smoothness": "C1 in query time; piecewise smooth in sample values" if method == "pchip"
                          else "continuous in query time; derivative jumps at knots"}


def validate_samples(query, knots, values):
    """Validate ordinary numeric arrays before entering an AD transform."""
    query, knots, values = [np.asarray(v, dtype=float) for v in (query, knots, values)]
    if (knots.ndim != 1 or len(knots) < 2 or values.ndim < 1
            or values.shape[0] != len(knots) or not np.isfinite(knots).all()
            or not np.isfinite(values).all() or np.any(np.diff(knots) <= 0)):
        raise ValueError("Interpolation requires finite samples on strictly increasing knots")
    if not np.isfinite(query).all():
        raise ValueError("Interpolation query must be finite")
    if np.any(query < knots[0]) or np.any(query > knots[-1]):
        raise ValueError("Observation query is outside the committed time axis; extrapolation is forbidden")
    return query, knots, values


def interpolate_numpy(query, knots, values, *, method):
    """Interpolate along the first value axis, with no endpoint clamping."""
    if method not in METHODS:
        raise ValueError("Time interpolation must be explicitly linear or pchip")
    query, knots, values = validate_samples(query, knots, values)
    if method == "pchip":
        return PchipInterpolator(knots, values, axis=0, extrapolate=False)(query)
    indices = np.clip(np.searchsorted(knots, query, side="right"), 1, len(knots)-1)
    fraction = (query-knots[indices-1]) / (knots[indices]-knots[indices-1])
    fraction = fraction.reshape(query.shape + (1,) * (values.ndim-1))
    return values[indices-1] + fraction * (values[indices]-values[indices-1])


def _edge_slope(h0, h1, m0, m1):
    derivative = ((2*h0+h1)*m0 - h0*m1) / (h0+h1)
    opposite = torch.sign(derivative) != torch.sign(m0)
    limit = (torch.sign(m0) != torch.sign(m1)) & (torch.abs(derivative) > 3*torch.abs(m0))
    return torch.where(opposite, torch.zeros_like(derivative),
                       torch.where(limit, 3*m0, derivative))


def _pchip_slopes(knots, values):
    """SciPy 1.16 PCHIP rules, preserving forward and reverse AD graphs."""
    widths = torch.diff(knots).reshape((-1,) + (1,) * (values.ndim-1))
    secants = torch.diff(values, dim=0) / widths
    if len(knots) == 2:
        return torch.cat((secants, secants), dim=0)
    previous, following = secants[:-1], secants[1:]
    same_sign = (torch.sign(previous) == torch.sign(following)) & (previous != 0) & (following != 0)
    # Safe inactive branches avoid NaN AD contributions from a zero secant.
    safe_previous = torch.where(same_sign, previous, torch.ones_like(previous))
    safe_following = torch.where(same_sign, following, torch.ones_like(following))
    w1, w2 = 2*widths[1:]+widths[:-1], widths[1:]+2*widths[:-1]
    harmonic = (w1+w2) / (w1/safe_previous+w2/safe_following)
    interior = torch.where(same_sign, harmonic, torch.zeros_like(harmonic))
    first = _edge_slope(widths[0], widths[1], secants[0], secants[1])
    last = _edge_slope(widths[-1], widths[-2], secants[-1], secants[-2])
    return torch.cat((first.unsqueeze(0), interior, last.unsqueeze(0)), dim=0)


def interpolate_torch(query, knots, values, *, method, validate=True):
    """Torch interpolation with differentiable values and query coordinates.

    Set validate=False only inside an AD transform whose ordinary numeric
    history/query was validated immediately before the transform. Shapes,
    method and dtype are still checked; no query is clamped or extrapolated.
    """
    if method not in METHODS:
        raise ValueError("Time interpolation must be explicitly linear or pchip")
    if (not all(torch.is_tensor(v) and v.is_floating_point() for v in (query, knots, values))
            or len({v.dtype for v in (query, knots, values)}) != 1
            or len({v.device for v in (query, knots, values)}) != 1):
        raise TypeError("Query, knots and values must be floating tensors with the same dtype and device")
    if knots.ndim != 1 or len(knots) < 2 or values.ndim < 1 or len(values) != len(knots):
        raise ValueError("Interpolation requires a one-dimensional knot axis and matching value samples")
    if validate:
        validate_samples(*[v.detach().cpu().numpy() for v in (query, knots, values)])
    original_shape = query.shape
    flat_query = query.reshape(-1)
    indices = torch.searchsorted(knots, flat_query.detach().contiguous(), right=True).clamp(1, len(knots)-1)
    left, right = indices-1, indices
    width = knots[right]-knots[left]
    fraction = ((flat_query-knots[left]) / width).reshape((-1,) + (1,) * (values.ndim-1))
    if method == "linear" or len(knots) == 2:
        result = values[left] + fraction * (values[right]-values[left])
    else:
        slopes = _pchip_slopes(knots, values)
        width = width.reshape((-1,) + (1,) * (values.ndim-1))
        t2, t3 = fraction.square(), fraction.pow(3)
        result = ((2*t3-3*t2+1)*values[left] + (t3-2*t2+fraction)*width*slopes[left]
                  + (-2*t3+3*t2)*values[right] + (t3-t2)*width*slopes[right])
    return result.reshape(original_shape + values.shape[1:])
