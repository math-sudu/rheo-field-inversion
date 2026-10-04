
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import minimize_scalar

from . import cad_source as cad

_EXEC6B = Path(__file__).resolve().parents[2] / "exec6b"
if str(_EXEC6B) not in sys.path:
    sys.path.insert(0, str(_EXEC6B))
import general_map
import mapping_solver

MODEL_VERSION = "tangent-cad-observation-v1"
# Both variants use the sourced contour. The second is a discretization
# check, not an invented alternative engineering dimension.
PRIMARY_CASE = "primary"
MAP_CHECK_CASE = "map48"
N_PROFILE = 720
N_MAP = 32
CHORD_DEPTHS = (2.5, 6.0, 9.5)
CHANNELS = ("SL01-SL02", "SL03-SL04", "SL05-SL06", "GD")
N_ORDER = 40
N_COLLOCATION = 320


@dataclass(frozen=True)
class FrozenGeometry:
    W: float
    H: float
    R: float
    c: np.ndarray
    theta_points: np.ndarray
    z_points: np.ndarray
    chord_lengths: np.ndarray
    fit_max_dist_m: float
    univalent: bool
    provenance: dict
    fingerprint: str
    mapped_z_points: np.ndarray
    has_survey_coordinates: bool = False


def _theta_of(R, c, z_target, n_grid=8192, n_refine=40):
    def zb(t):
        return R * (np.exp(1j * t) + sum(
            cj * np.exp(-1j * j * t) for j, cj in enumerate(c)))
    th = np.arange(n_grid) * 2*np.pi/n_grid
    k = int(np.argmin(abs(zb(th) - z_target)))
    step = 2*np.pi/n_grid
    fit = minimize_scalar(lambda t: abs(zb(t)-z_target)**2,
                          bounds=(th[k]-step, th[k]+step), method="bounded",
                          options={"xatol": 1e-14, "maxiter": 100})
    return float(fit.x % (2*np.pi))


def build_geometry(WH=None, *, reference=False, scale=None, size_source=None,
                   boundary_role="initial_support_outer", translation=0j,
                   n_map=N_MAP, n_profile=N_PROFILE, points=None,
                   observation_source=None) -> FrozenGeometry:
    """Reference or dimensioned shaft model without altering source arcs.

    One scale acts about the CAD origin followed by translation. Optional
    seven survey points are MODEL coordinates [L1,L2,L3,R1,R2,R3,GD]. Without
    them, schematic depths define model observations, not surveyed positions.
    """
    if isinstance(WH, str) or (WH is None and not reference and scale is None and size_source is None):
        from .campaign import geometry_options
        return build_geometry(**geometry_options(WH or PRIMARY_CASE))
    if WH is not None:
        raise cad.GeometryBindingError(
            "Width/height-only selection is retired. Use one scale and "
            "size_source, or reference=True; do not reuse the old WH cache.")
    if reference:
        if scale is not None or size_source is not None:
            raise ValueError("Reference uses source dimensions; select shaft for a size change")
        scale, size_source = 1.0, "reference-section coordinates in metres"
    elif scale is None or not size_source:
        cad.require_shaft_dimensions()
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("One positive finite scalar is required")
    source = cad.boundary(boundary_role)
    profile = cad.sample_boundary(boundary_role, n_profile, scale, translation)
    ymin = scale * source["bounds_min"][1] + translation.imag
    ymax = scale * source["bounds_max"][1] + translation.imag
    if points is None:
        chords = [cad.horizontal_intersections(ymax-d, boundary_role, scale,
                                               translation) for d in CHORD_DEPTHS]
        crown_x = scale*source["circular_arcs"][0]["center"][0] + translation.real
        z_all = np.array([q[0] for q in chords] + [q[1] for q in chords]
                         + [complex(crown_x, ymax)])
        observation_source = (
            "reference layout: monitoring-target depths 2.5/6.0/9.5 m; "
            "transplanted to model, not surveyed CAD registration")
    else:
        if not observation_source:
            raise ValueError("Explicit observation coordinates require their source")
        z_all = np.asarray(points, dtype=complex)
        if z_all.shape != (7,) or not np.isfinite(z_all).all():
            raise ValueError("Seven finite observation points required")
        gap = cad.distance_to_boundary(z_all, boundary_role, scale, translation)
        if np.max(gap) > 0.005:
            raise cad.GeometryBindingError(f"Survey points miss selected boundary by {max(gap):.6g} m")
    fit = mapping_solver.solve_mapping(profile, int(n_map))
    R, c = float(fit["R"]), np.asarray(fit["c"], dtype=complex)
    uni = bool(fit["univalence"]["univalent_ok"] and
               fit["univalence"]["boundary_simple"])
    if not uni or not fit["converged"]:
        raise ValueError(f"Mapping quality failure: {fit['termination']}, univalent={uni}")
    theta = np.array([_theta_of(R, c, z) for z in z_all])
    mapped = R*(np.exp(1j*theta) + sum(cj*np.exp(-1j*j*theta)
                                     for j, cj in enumerate(c)))
    provenance = {
        "model_version": MODEL_VERSION,
        "case_id": "main_tunnel_reference" if reference else "shaft_same_shape",
        "source_version": cad.SOURCE_VERSION, "source_sha256": cad.SOURCE_SHA256,
        "arcs_sha256": cad.ARCS_SHA256, "boundary_role": boundary_role,
        "tangent_dxf_sha256": cad.TANGENT_DXF_SHA256,
        "metres_per_drawing_unit": 1.0, "scale": float(scale),
        "size_source": size_source, "translation_m": [translation.real, translation.imag],
        "coordinate_transform": "z_model = scale * z_CAD + translation; +x right, +y up",
        "cavity_assumption": "selected support outline used as traction-free reference cavity; excavation coincidence not asserted",
        "second_lining": None, "observation_source": observation_source,
        "points_m": [[z.real, z.imag] for z in z_all],
        "observation_version": "chord-vector-projection-v1",
        "displacement_convention": "physical +x right/+y up; KM return = minus physical",
        "regularization": "finite Laurent least-squares fit to author-authorized tangent arcs; no additional source repair",
        "n_map": int(n_map), "n_profile_actual": len(profile),
        "fit_max_source_sample_m": float(fit["fit_max_dist"]),
        "fit_termination": fit["termination"],
        "point_projection_max_m": float(max(abs(mapped-z_all))),
        "bounds_y_m": [ymin, ymax],
    }
    identity = {"provenance": provenance, "R": R,
                "c": [[z.real, z.imag] for z in c]}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return FrozenGeometry(W=scale*source["width"], H=ymax-ymin, R=R, c=c,
                          theta_points=theta, z_points=z_all,
                          chord_lengths=abs(z_all[3:6]-z_all[:3]),
                          fit_max_dist_m=float(fit["fit_max_dist"]), univalent=uni,
                          provenance=provenance, fingerprint=fingerprint,
                          mapped_z_points=mapped,
                          has_survey_coordinates=points is not None)


def observation_matrix(geom, *, returned_frame="physical"):
    """Four rows mapping [ux(7),uy(7)] to closing/down-positive metres.

    Chords measure -(u_R-u_L).e_LR. Directions use the sourced target points,
    never relocated survey points. Crown settlement is vertical downward.
    """
    W = np.zeros((4, 14))
    for j in range(3):
        v = geom.z_points[j+3] - geom.z_points[j]
        if abs(v) <= 0:
            raise ValueError("Zero-length observation chord")
        e = v/abs(v)
        W[j, j], W[j, j+3] = e.real, -e.real
        W[j, j+7], W[j, j+10] = e.imag, -e.imag
    W[3, 13] = -1.0
    if returned_frame == "km":
        W *= -1
    elif returned_frame != "physical":
        raise ValueError("Unknown displacement frame")
    return W


def make_rep(geom, sigma_v=1.0, K0=0.0, n_order=N_ORDER):
    return general_map.GeneralMappedKM(geom.R, geom.c, sigma_v=sigma_v,
                                       K0=K0, n_order=n_order)
