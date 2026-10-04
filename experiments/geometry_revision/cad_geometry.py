"""Exact CAD measurement locations and their distinct rock-boundary images.

No survey datum is inferred here. A caller must supply the depth datum and
measured surface explicitly. The normal transfer is a declared kinematic
approximation; it does not resolve deformation across the support thickness.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
from pathlib import Path

import numpy as np

from .baseline import load_modules, digest

CHANNELS = ("GD", "SL01-SL02", "SL03-SL04", "SL05-SL06")
POINT_NAMES = ("SL01", "SL03", "SL05", "SL02", "SL04", "SL06", "GD")
DEPTHS_M = (2.5, 6.0, 9.5)
INNER = "initial_support_inner"
OUTER = "initial_support_outer"
OPERATOR_VERSION = "cad-normal-pair-physical-chord-v2"


def pairs(values):
    return [[float(z.real), float(z.imag)] for z in values]


def project_normal(points, *, from_role, to_role=OUTER, scale=1.0,
                   translation=0j):
    """Pair the same arc/angle on the two author-verified parallel outlines."""
    cad = load_modules().cad
    source = cad.boundary(from_role)["circular_arcs"]
    target = {arc["arc_id"]: arc for arc in cad.boundary(to_role)["circular_arcs"]}
    projected, records = [], []
    for point in np.asarray(points, dtype=complex):
        original = (point - translation) / scale
        hits = []
        for arc in source:
            center = complex(*arc["center"])
            angle = np.angle(original - center)
            if abs(abs(original - center) - arc["radius"]) < 1e-8 and cad.on_arc(angle, arc):
                other = target[arc["arc_id"]]
                if not np.allclose(arc["center"], other["center"], atol=1e-12, rtol=0):
                    raise ValueError("Normal pairing requires identical source arc centres")
                if not cad.on_arc(angle, other):
                    raise ValueError("Corresponding angle is outside the target arc")
                candidate = scale * cad.arc_point(other, angle) + translation
                hits.append((candidate, arc["arc_id"], float(angle)))
        if not hits:
            raise ValueError(f"Measurement point {point!r} is not on the declared CAD surface")
        if any(abs(item[0] - hits[0][0]) > 1e-8 for item in hits):
            raise ValueError("Ambiguous normal image at an arc junction")
        projected.append(hits[0][0])
        records.append({"arc_id": hits[0][1], "angle_rad": hits[0][2],
                        "normal_distance_m": float(abs(hits[0][0] - point))})
    return np.asarray(projected), records


def physical_projection(points):
    """Rows in CHANNELS order; input [ux(7),uy(7)], metres to millimetres."""
    points = np.asarray(points, dtype=complex)
    if points.shape != (7,) or not np.isfinite(points).all():
        raise ValueError("Seven finite physical measurement points are required")
    matrix = np.zeros((4, 14))
    matrix[0, 13] = -1000.0
    for j in range(3):
        difference = points[j + 3] - points[j]
        if abs(difference) <= 0:
            raise ValueError("Zero-length physical chord")
        direction = difference / abs(difference)
        matrix[j + 1, j] = 1000 * direction.real
        matrix[j + 1, j + 3] = -1000 * direction.real
        matrix[j + 1, j + 7] = 1000 * direction.imag
        matrix[j + 1, j + 10] = -1000 * direction.imag
    return matrix


@dataclass(frozen=True)
class CadLayout:
    geometry: object
    physical_points: np.ndarray
    boundary_points: np.ndarray
    physical_chord_lengths: np.ndarray
    record: dict
    fingerprint: str


def build_layout(*, datum_role, measured_surface, scale=1.0, translation=0j,
                 depths_m=DEPTHS_M, n_map=32, size_source,
                 observation_source, engineering_status):
    """Construct a single-similarity CAD model without moving measurements.

    GD is the crown of the measured surface. Chord heights are defined by
    the explicitly selected crown datum, which may be a different surface.
    Neither dimensionless source scaling nor translation rescales the supplied
    physical depths in metres.
    """
    if datum_role not in (INNER, OUTER) or measured_surface not in (INNER, OUTER):
        raise ValueError("Explicit inner/outer crown datum and measurement surface required")
    if not np.isfinite(scale) or scale <= 0 or not size_source or not observation_source:
        raise ValueError("One positive scale and explicit dimensional/observation sources required")
    if len(depths_m) != 3 or not np.isfinite(depths_m).all() or min(depths_m) <= 0:
        raise ValueError("Three positive physical measurement depths are required")
    modules = load_modules()
    cad = modules.cad
    datum_y = scale * cad.boundary(datum_role)["bounds_max"][1] + translation.imag
    intersections = [cad.horizontal_intersections(datum_y - d, measured_surface,
                                                  scale, translation) for d in depths_m]
    surface = cad.boundary(measured_surface)
    crown = complex(scale * surface["circular_arcs"][0]["center"][0] + translation.real,
                    scale * surface["bounds_max"][1] + translation.imag)
    physical = np.array([p[0] for p in intersections] + [p[1] for p in intersections] + [crown])
    boundary, normal_records = project_normal(physical, from_role=measured_surface,
                                              scale=scale, translation=translation)
    geometry = modules.geometry.build_geometry(scale=scale, size_source=size_source,
        boundary_role=OUTER, translation=translation, n_map=n_map, points=boundary,
        observation_source="Computational normal images; physical locations registered separately: " + observation_source)
    record = {"geometry_id": cad.SOURCE_VERSION, "operator_version": OPERATOR_VERSION,
        "source_arcs_sha256": cad.ARCS_SHA256, "scale": float(scale),
        "translation_m": [float(translation.real), float(translation.imag)],
        "size_source": size_source, "engineering_status": engineering_status,
        "depth_datum_role": datum_role, "depth_datum_y_m": float(datum_y),
        "depths_below_datum_m": list(depths_m), "measured_surface": measured_surface,
        "observation_source": observation_source, "point_names": POINT_NAMES,
        "channels": CHANNELS, "physical_points_m": pairs(physical),
        "computational_boundary_points_m": pairs(boundary), "normal_pairs": normal_records,
        "physical_chord_lengths_m": abs(physical[3:6] - physical[:3]).tolist(),
        "computational_chord_lengths_m": abs(boundary[3:6] - boundary[:3]).tolist(),
        "physical_projection_mm_per_m": physical_projection(physical).tolist(),
        "cavity_role": OUTER, "support_thickness_m": 0.30 * float(scale),
        "transfer": "u_measured(point) = u_rock(normal_image(point)); support through-thickness strain is unresolved",
        "boundary_model": "Incremental excavation stress release on outer initial-support outline; total traction (1-lambda)*sigma0.n; no explicit support elements or stiffness",
        "axis_convention": "+x right, +y up; crown settlement and chord shortening positive",
        "mapping_fingerprint": geometry.fingerprint,
        "operator_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    fingerprint = digest(record)
    geometry = replace(geometry, has_survey_coordinates=False,
        provenance={**geometry.provenance, "registered_layout": record,
                    "observation_points_role": "computational images, not physical target coordinates",
                    "cavity_assumption": record["boundary_model"]},
        fingerprint=digest({"mapping_fingerprint": geometry.fingerprint,
                            "layout_fingerprint": fingerprint}))
    for array in (physical, boundary):
        array.setflags(write=False)
    return CadLayout(geometry, physical, boundary,
                     abs(physical[3:6] - physical[:3]), record, fingerprint)


def channel_matrix(mesh, layout):
    """Periodic cubic interpolation on the rock boundary, physical projection.

    The spatial interpolator is linear in nodal displacements. Geometry
    inversion supplies the normal-image angles; the chord projection retains
    the direction of the actual measured segment.
    """
    from scipy.interpolate import CubicSpline
    theta = np.asarray(mesh.theta_nodes)
    # theta_nodes stores n_t periodic nodes, excluding the duplicate endpoint.
    if theta.ndim != 1:
        raise ValueError("Expected the periodic angular mesh coordinate")
    basis = CubicSpline(np.r_[theta, theta[0] + 2*np.pi],
                        np.vstack((np.eye(len(theta)), np.eye(len(theta))[0])),
                        axis=0, bc_type="periodic")(
                            np.mod(layout.geometry.theta_points, 2*np.pi))
    interpolation = np.zeros((14, mesh.n_dof))
    inner = np.asarray(mesh.ring_dofs("inner"))
    if len(inner) != 2*len(theta):
        raise ValueError("Unexpected inner-ring displacement storage")
    interpolation[:7, inner[0::2]] = basis
    interpolation[7:, inner[1::2]] = basis
    return physical_projection(layout.physical_points) @ interpolation
