"""Selected numerical functions exported from the paper implementation."""
import inspect
from types import SimpleNamespace
import numpy as np
from experiments.fe_pinc.core import CoordinateMap, fixed_active_set_tangent, projected_gradient_mapping
from experiments.geometry_revision.rheo_pinc_network import train_continuation

def local_terms(theta, values, physical_jacobian, problem, profile_coordinate):
    """Half-SSE derivatives and a regularized, fixed-active Gauss--Newton slope.

The slope solves the local normal equations. It is a profile-derivative
training target only when this point belongs to a smooth stationary profile
branch; material-side witnesses alone do not establish that property.
    """
    names = list(problem["free"])
    if profile_coordinate not in names:
        raise ValueError("The profile coordinate must be present in the physical Jacobian")
    values, physical = np.asarray(values, float), np.asarray(physical_jacobian, float)
    y, sigma = np.asarray(problem["y_mm"], float), np.asarray(problem["sigma_mm"], float)
    if (values.shape != y.shape or sigma.shape != y.shape
            or physical.shape != (len(y), len(names)) or np.any(sigma <= 0)
            or not all(np.isfinite(a).all() for a in (values, y, sigma, physical))):
        raise ValueError("Local physics requires finite compatible observations and physical derivatives")
    mapping = CoordinateMap(problem["box"])
    q_index = names.index(profile_coordinate)
    nuisance = [i for i in range(len(names)) if i != q_index]
    free = [names[i] for i in nuisance]
    chain = np.array([mapping.dphysical_dunit(n, theta[n]) for n in names])
    jacobian = physical / sigma[:, None] * chain
    residual = (values-y)/sigma
    jq, jz = jacobian[:, q_index], jacobian[:, nuisance]
    gradient, hessian, cross = jz.T@residual, jz.T@jz, jz.T@jq
    unit = np.array([mapping.to_unit(n, theta[n]) for n in free])
    tangent, active = fixed_active_set_tangent(hessian, cross, unit, gradient)
    scale = max(float(np.linalg.norm(hessian)), 1.)
    projected = projected_gradient_mapping(unit, gradient, scale=scale)
    return {"profile_coordinate": profile_coordinate, "free_names": free,
        "x": 2*mapping.to_unit(profile_coordinate, theta[profile_coordinate])-1,
        "reference_unit": unit.tolist(), "gradient": gradient.tolist(),
        "hessian": hessian.tolist(), "cross": cross.tolist(),
        "local_GN_slope_dx": (.5*tangent).tolist(), "active_mask": active.tolist(),
        "projected_stationarity_norm": float(np.linalg.norm(projected)),
        "sse": float(residual@residual)}

def training_arrays(anchors):
    if not anchors or any(row["role"] != "smooth_profile_endpoint" for row in anchors):
        raise ValueError("Training requires smooth profile endpoints; switch and side probes are local inputs only")
    first = anchors[0]["terms"]
    if any((row["terms"]["free_names"], row["terms"]["profile_coordinate"])
           != (first["free_names"], first["profile_coordinate"]) for row in anchors):
        raise ValueError("A continuation model requires one coordinate and one nuisance ordering")
    return {"x": np.array([row["terms"]["x"] for row in anchors]),
        "branch": np.array([row["branch"] for row in anchors], dtype=int),
        "target": np.array([row["terms"]["reference_unit"] for row in anchors]),
        "correction": np.array([row["correction"] for row in anchors]),
        "gradient": np.array([row["terms"]["gradient"] for row in anchors]),
        "hessian": np.array([row["terms"]["hessian"] for row in anchors]),
        "tangent": np.array([row["terms"]["local_GN_slope_dx"] for row in anchors])}

pinc = SimpleNamespace(training_arrays=training_arrays)

def prepare_profile(bundle):
    """Use every saved value, with separate rows for the local GN losses.

Targets are the already corrected FE centers. The zero additional correction
keeps the local expansion and its target at that same center. No q-side
secant or nuisance predictor error is a correction at the center's fixed q.
    """
    anchors, selected, retained, values = [], [], [], []
    for i, row in enumerate(bundle["rows"]):
        if row["branch"] not in (0, 1) or row["branch"] != row["sample"]["branch"]:
            raise ValueError("A profile row lost its saved numerical-track label")
        entries = [(0, row["terms"])] + [
            (side["direction"], side["terms"]) for side in row["one_sided_inputs"]
            if side["terms"] is not None]
        for direction, terms in entries:
            if (terms["free_names"], terms["profile_coordinate"]) != (
                    bundle["rows"][0]["terms"]["free_names"],
                    bundle["rows"][0]["terms"]["profile_coordinate"]):
                raise ValueError("Value supervision changed its nuisance ordering or coordinate")
            values.append({"row_index": i, "direction": direction,
                "branch": row["branch"], "terms": terms, "weight": 1/len(entries)})
        if row["role"] == "profile_input_with_unresolved_side_or_switch":
            retained.append(i)
            continue
        if row["role"] != "locally_resolved_profile_input":
            raise ValueError("The trainer cannot consume this profile-input role")
        sides = row["one_sided_inputs"]
        if not row["sample"]["converged"] or not sides or any(
            side["terms"] is None or side["continuation"] is None
            or not side["continuation"]["both_numerically_resolved"]
            or not side["continuation"]["same_increment_schedule"]
            or side["continuation"]["material_changes"] != []
            or not side["continuation"]["same_active_mask"] for side in sides
        ):
            raise ValueError("A local GN training row requires resolved unchanged q sides")
        # This is the existing trainer's local smooth-row convention, not an
        # assertion of an exact profile derivative between the checked points.
        anchors.append({"role": "smooth_profile_endpoint", "branch": row["branch"],
            "terms": row["terms"], "correction": np.zeros(len(row["terms"]["free_names"]))})
        selected.append(i)
    arrays = pinc.training_arrays(anchors)
    n, p = arrays["target"].shape
    shapes = {"x": (n,), "branch": (n,), "target": (n, p), "correction": (n, p),
              "gradient": (n, p), "hessian": (n, p, p), "tangent": (n, p)}
    if any(a.shape != shapes[name] or not np.isfinite(a).all()
           for name, a in arrays.items()):
        raise ValueError("The profile tensors are incompatible with the continuation trainer")
    arrays.update(value_x=np.array([v["terms"]["x"] for v in values]),
        value_branch=np.array([v["branch"] for v in values], dtype=int),
        value_target=np.array([v["terms"]["reference_unit"] for v in values]),
        value_weight=np.array([v["weight"] for v in values]))
    if not all(np.isfinite(arrays[k]).all() for k in
               ("value_x", "value_target", "value_weight")):
        raise ValueError("Saved-value supervision requires finite coordinates and targets")
    if arrays["value_target"].shape != (len(values), p):
        raise ValueError("Saved-value supervision changed its nuisance dimension")
    inspect.signature(train_continuation).bind(**arrays, seed=11)
    return {"arrays": arrays, "value_samples": values,
        "training_row_indices": list(range(len(bundle["rows"]))),
        "physics_row_indices": selected, "value_only_row_indices": retained,
        "profile_inputs": bundle,
        "free_names": anchors[0]["terms"]["free_names"],
        "value_meaning": "saved candidate parameter values; equal total weight per center and its q sides; both numerical labels retain their own targets, including where tracks nearly coincide",
        "tangent_meaning": "regularized fixed-active local GN slope with respect to x=2*u_q-1",
        "correction_meaning": "zero additional correction at the saved corrected FE center"}

def pool_reading(candidates):
    eligible = [i for i, row in enumerate(candidates) if (row.get("terminal") or {}).get("eligible", False)]
    reading = {"selected_index": None, "best_eligible_index": None, "lower_unresolved_indices": [], "status": "blocked_numerical"}
    if not eligible:
        return reading
    best = min(eligible, key=lambda i: candidates[i]["terminal"]["exact_objective"])
    value = float(candidates[best]["terminal"]["exact_objective"])
    lower = [i for i, row in enumerate(candidates) if i not in eligible and row.get("candidate_sse") is not None
             and float(row["candidate_sse"]) < value-1e-10*max(abs(value), 1.)]
    reading.update(best_eligible_index=best, lower_unresolved_indices=lower)
    if not lower:
        reading.update(selected_index=best, status="completed")
    return reading
