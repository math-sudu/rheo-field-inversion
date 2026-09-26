"""Versioned exact-arc source; engineering identity is separate from a model.

Author confirmation, 2026-09-05: these are MAIN-TUNNEL initial-support
outlines. One drawing unit is one metre despite DXF $INSUNITS=4. Neither
outline supplies a shaft survey frame, an excavation line, or a second lining.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

SOURCE_VERSION = "main_tunnel_tangent_v1"
SOURCE_SHA256 = "53a477fcda15c53d2e2bb09bce487c040edc1fd5c209e9e00ed824bf3901e557"
ARCS_SHA256 = "cea3b2f1b975dd2172b57647b2bff8654d7a0d59811dd0a542f4f5ad6c934a95"
TANGENT_DXF_SHA256 = "cbddcc41e1973c3f5ad8e5d424b439be9bf38eb1e57cc578d159b3f193e35c15"
ORIGINAL_ARCS_SHA256 = "c6cc69a963b02089b6f58b48cd34361e2157aff15c5ab3f9a3affbffd979468b"
SOURCE_DIR = (Path(__file__).resolve().parents[3] / "results" /
              "geometry_correction" / "source" / SOURCE_VERSION)


class GeometryBindingError(ValueError):
    """A geometry/observation model has no authorised engineering binding."""


def require_shaft_dimensions():
    raise GeometryBindingError(
        "The author confirms the same shape for the shaft, with different size. "
        "Supply one shaft similarity scale and its dimensional source; the "
        "7.5 x 7.5 m clearance does not specify the initial-support outline. "
        "No shaft computation is disabled: build_geometry(scale=..., "
        "size_source=...) selects it; reference=True selects the main tunnel.")


def load_source():
    for name, expected in (("source.dxf", SOURCE_SHA256),
                           ("circular_arcs.json", ORIGINAL_ARCS_SHA256),
                           ("tangent_arcs.json", ARCS_SHA256),
                           ("tangent_section.dxf", TANGENT_DXF_SHA256)):
        if hashlib.sha256((SOURCE_DIR / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"CAD source checksum mismatch: {name}")
    data = json.loads((SOURCE_DIR / "tangent_arcs.json").read_text("utf-8"))
    assert data["confirmed_metres_per_drawing_unit"] == 1.0
    assert data["source_sha256"] == SOURCE_SHA256
    assert data["geometry_id"] == SOURCE_VERSION
    assert data["source_arcs_sha256"] == ORIGINAL_ARCS_SHA256
    return data


def boundary(role="initial_support_outer"):
    return next(b for b in load_source()["boundaries"] if b["role"] == role)


def arc_point(arc, angle):
    return complex(*arc["center"]) + arc["radius"] * np.exp(1j * angle)


def sample_boundary(role="initial_support_outer", n=720, scale=1.0,
                    translation=0j):
    """Sample authorised tangent arcs by length, retaining every joint.

    The single positive scale acts about the CAD origin, then translation is
    applied. Counts are minimum total density, not an independent x/y scaling.
    """
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Similarity scale must be one positive finite scalar")
    b = boundary(role)
    points = []
    for arc in b["circular_arcs"]:
        count = max(2, int(np.ceil(n * arc["radius"] * abs(arc["sweep_rad"])
                                   / b["perimeter"])))
        angle = arc["start_rad"] + arc["sweep_rad"] * np.arange(count) / count
        points.extend(arc_point(arc, angle))
    return scale * np.asarray(points) + translation


def on_arc(angle, arc, tol=1e-10):
    delta = (angle - arc["start_rad"]) % (2 * np.pi)
    return delta <= arc["sweep_rad"] + tol or 2 * np.pi - delta < tol


def horizontal_intersections(y, role="initial_support_outer", scale=1.0,
                             translation=0j):
    """Exact circle-line intersections restricted to the source arc spans."""
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Invalid similarity scale")
    y0 = (y - translation.imag) / scale
    hits = []
    for arc in boundary(role)["circular_arcs"]:
        cx, cy = arc["center"]
        q = (y0 - cy) / arc["radius"]
        if abs(q) > 1 + 1e-12:
            continue
        alpha = np.arcsin(np.clip(q, -1, 1))
        for a in (alpha, np.pi - alpha):
            if on_arc(a, arc):
                z = scale * arc_point(arc, a) + translation
                if all(abs(z - prev) > 1e-8 for prev in hits):
                    hits.append(z)
    hits.sort(key=lambda z: z.real)
    if len(hits) != 2:
        raise GeometryBindingError(
            f"Horizontal line y={y:.9g} m has {len(hits)} wall intersections; "
            "point relocation, clipping and anisotropic scaling are forbidden")
    return np.array(hits)


def distance_to_boundary(points, role="initial_support_outer", scale=1.0,
                         translation=0j):
    """Exact nearest distance to the union of the adopted circular arcs."""
    zz = (np.asarray(points) - translation) / scale
    best = np.full(zz.shape, np.inf)
    for arc in boundary(role)["circular_arcs"]:
        c = complex(*arc["center"])
        v = zz - c
        angle = np.angle(v)
        delta = np.mod(angle - arc["start_rad"], 2 * np.pi)
        inside = (delta <= arc["sweep_rad"] + 1e-12) | (delta > 2*np.pi-1e-12)
        ends = arc_point(arc, np.array([arc["start_rad"],
                                        arc["start_rad"] + arc["sweep_rad"]]))
        d = np.minimum(abs(zz - ends[0]), abs(zz - ends[1]))
        d = np.where(inside, np.abs(abs(v) - arc["radius"]), d)
        best = np.minimum(best, d)
    return scale * best
