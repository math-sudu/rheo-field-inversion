"""CAD field forward models with explicit common material/stress anchors.

The CAD layout specifies physical targets and their separate rock-boundary
images. The caller supplies dimensional anchors and angle conventions; none
are inherited from the old field scenario. Physical conversion and replay
share one instance-local graph, so differently anchored cases may coexist.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np
import torch

from .baseline import digest, load_modules
from .cad_geometry import CHANNELS, CadLayout, channel_matrix
from .newton import install_newton
from .temporal import METHODS, interpolate_numpy, interpolate_torch, interpolation_identity


RHEO_COORDINATES = ("s_star", "K0", "r_K", "tau_K", "tau_M", "r_s", "tau_vp")
CWFS_COORDINATES = ("s_star", "K0", "r_c", "chi_c", "gamma_c", "dphi", "gamma_phi")


@dataclass(frozen=True)
class _Scenario:
    family: str
    parameters: dict
    sigma_v: float
    nu: float
    phi: float
    psi: float

    def __getattr__(self, name):
        parameters = object.__getattribute__(self, "parameters")
        if name in parameters:
            return parameters[name]
        raise AttributeError(name)

    @property
    def G0(self):
        return self.sigma_v / (2. * self.parameters["s_star"])

    @property
    def E(self):
        return 2. * (1. + self.nu) * self.G0

    def base(self, *, a):
        from fedev_src.params import BaseParams
        return BaseParams(a=a, sigma_v=self.sigma_v, K0=self.parameters["K0"], E=self.E, nu=self.nu)

    def material(self):
        from fedev_src.params import CWFSMohrCoulomb, NishiharaMC
        parameters = {name: float(value) for name, value in _physical_graph(self)["mat"].items()}
        return NishiharaMC(**parameters) if self.family == "rheo" else CWFSMohrCoulomb(**parameters)


def _physical_graph(scenario, overrides=None):
    """Explicit quotient-to-physical graph, used by both FE and its replay."""
    from fedev_src.meshing import DTYPE

    values = {**scenario.parameters, **(overrides or {})}

    def tensor(value):
        return value if torch.is_tensor(value) else torch.tensor(float(value), dtype=DTYPE)

    leaves = {name: None if value is None else tensor(value) for name, value in values.items()}
    sv, nu = tensor(scenario.sigma_v), tensor(scenario.nu)
    shear = sv / (2. * leaves["s_star"])
    young = 2. * (1. + nu) * shear
    phi, psi = tensor(scenario.phi), tensor(scenario.psi)
    if scenario.family == "rheo":
        kelvin_shear = leaves["r_K"] * shear
        if leaves["r_s"] is None:
            cohesion, eta_vp = tensor(1e9), tensor(1e12)
        else:
            n_phi = (1. + np.sin(scenario.phi)) / (1. - np.sin(scenario.phi))
            cohesion = leaves["r_s"] * sv / (2. * np.sqrt(n_phi))
            eta_vp = (leaves["tau_vp"] * 2. * shear * .5
                      * (1. - np.sin(scenario.phi)) * (1. - np.sin(scenario.psi)))
        material = {"c": cohesion, "phi": phi, "psi": psi,
                    "E_K": 2. * (1. + nu) * kelvin_shear,
                    "eta_K": leaves["tau_K"] * kelvin_shear,
                    "eta_M": leaves["tau_M"] * 2. * shear, "eta_vp": eta_vp}
    else:
        peak = leaves["r_c"] * sv
        material = {"c_peak": peak, "c_res": leaves["chi_c"] * peak,
                    "gamma_c": leaves["gamma_c"], "phi_peak": phi,
                    "phi_res": phi + leaves["dphi"], "gamma_phi": leaves["gamma_phi"],
                    "psi": psi, "visc_H": tensor(0.)}
    return {"E": young, "nu": nu, "sigma_v": sv, "K0": leaves["K0"], "mat": material}


class _TemporalObservation:
    """Keep every public observation entry on the selected temporal method."""

    def __init__(self, observation, method):
        self._observation, self.method = observation, method

    def __getattr__(self, name):
        return getattr(self._observation, name)

    def model(self, times, series, phase, *, validate=True):
        from exec8_src import schedule
        offset = schedule.T_REL * torch.log((1.-schedule.LAM_F)/(1.-phase)).pow(1./schedule.BETA)
        output = {}
        for column, channel in enumerate(CHANNELS):
            query = offset + torch.tensor(self.stamps.stamps[channel]-self.stamps.stamp0,
                                          dtype=series.dtype, device=series.device)
            values = interpolate_torch(query, times, series[:, column], method=self.method, validate=validate)
            output[channel] = values-values[0]
        return output


class CadFieldEvaluator:
    """EXEC-19 evaluator interface with explicit CAD and physical anchors."""

    def __init__(self, layout, *, family, sigma_v, nu, phi_deg, psi_deg,
                 parameter_source, section, time_interpolation, n_r=8, n_t=32, stamps=None,
                 axis=None, axis_kw=None, sigma_section=None, max_iter=80,
                 min_substep=1/64):
        if not isinstance(layout, CadLayout):
            raise TypeError("A registered CadLayout is required")
        if family not in ("rheo", "cwfs"):
            raise ValueError("Select rheo or cwfs explicitly")
        if time_interpolation not in METHODS:
            raise ValueError("Time interpolation must be explicitly linear or pchip")
        anchors = np.array([sigma_v, nu, phi_deg, psi_deg], dtype=float)
        if (not np.isfinite(anchors).all() or sigma_v <= 0 or not -1 < nu < .5
                or not 0 <= phi_deg < 90 or not -90 < psi_deg <= phi_deg):
            raise ValueError("Invalid explicit stress, Poisson ratio, friction or dilation angle")
        if not parameter_source or not section:
            raise ValueError("Parameter source and section/data handle must be explicit")
        if max_iter < 1 or not 0 < min_substep <= 1:
            raise ValueError("Invalid Newton iteration or minimum-substep controls")
        if axis is not None and axis_kw:
            raise ValueError("Specify an axis or axis-generation options, not both")

        modules = load_modules()
        from fedev_src import coupling
        from fedev_src.params import BaseParams

        self.modules, self.layout, self.geom = modules, layout, layout.geometry
        self.family = family
        self.time_interpolation = time_interpolation
        self.sigma_v, self.nu = float(sigma_v), float(nu)
        self.phi, self.psi = float(np.deg2rad(phi_deg)), float(np.deg2rad(psi_deg))
        self.max_iter, self.min_substep = int(max_iter), float(min_substep)
        self.n_r, self.n_t = int(n_r), int(n_t)
        if self.n_r < 1 or self.n_t < 8:
            raise ValueError("A positive radial mesh and at least eight angular elements are required")
        self.newton_policy = install_newton()
        original = modules.field.FieldStamps.load(section) if stamps is None else stamps
        if tuple(original.stamps) != CHANNELS and set(original.stamps) != set(CHANNELS):
            raise ValueError("Observation channels differ from the registered CAD operator")
        own_stamps, own_values = {}, {}
        for channel in CHANNELS:
            times = np.array(original.stamps[channel], dtype=float, copy=True)
            values = np.array(original.y_mm[channel], dtype=float, copy=True)
            if (times.ndim != 1 or not len(times) or values.shape != times.shape
                    or not np.isfinite(times).all() or not np.isfinite(values).all()
                    or np.any(np.diff(times) < 0)):
                raise ValueError(f"Invalid raw observation series: {channel}")
            # Distinct raw readings can share a digitized timestamp. Keep
            # both ordinates; strict ordering applies to FE knots, not data.
            times.setflags(write=False)
            values.setflags(write=False)
            own_stamps[channel], own_values[channel] = times, values
        first_stamp = min(float(times[0]) for times in own_stamps.values())
        if not np.isfinite(original.stamp0) or abs(float(original.stamp0) - first_stamp) > 1e-9:
            raise ValueError("Section stamp0 must equal the first original channel timestamp")
        self.stamps = modules.field.FieldStamps(section=original.section, stamps=own_stamps,
            y_mm=own_values, stamp0=float(original.stamp0))
        self.mesh = modules.coupled.build_mesh(self.geom, n_r=self.n_r, n_t=self.n_t,
                                               r_level=modules.coupled.R_LEVEL)
        self.W = channel_matrix(self.mesh, layout)
        self.W.setflags(write=False)
        self.obs = _TemporalObservation(
            modules.field.FieldObservation(self.stamps, self.W, sigma_section=sigma_section), time_interpolation)
        self.sigma = modules.identify.sigma_vector(self.obs)
        if not np.isfinite(self.sigma).all() or np.any(self.sigma <= 0):
            raise ValueError("Channel noise scales must be finite and positive")
        if axis is None:
            # Preserve a common full-section axis when a window supplies a
            # truncated observation object. Synthetic objects supply an axis.
            span = modules.field.FieldStamps.load(section).span() if stamps is not None else self.stamps.span()
            axis = modules.field.field_knot_axis(span, **(axis_kw or {}))
        self.lams, self.times = [np.array(v, dtype=float, copy=True) for v in axis]
        if (self.times.ndim != 1 or len(self.times) < 2 or self.lams.shape != self.times.shape
                or not np.isfinite(self.times).all() or not np.isfinite(self.lams).all()
                or self.times[0] != 0 or self.lams[0] != 0
                or np.any(np.diff(self.times) <= 0) or np.any(np.diff(self.lams) < 0)
                or np.any(self.lams < 0) or np.any(self.lams > 1)):
            raise ValueError("A finite increasing time axis and monotone release axis from zero are required")
        self.times.setflags(write=False)
        self.lams.setflags(write=False)
        # E=1 MPa is a computational scaling anchor only. The exact condensed
        # boundary is recomputed using THIS case's nu, never the old scenario.
        self._reference_base = BaseParams(a=self.geom.R, sigma_v=self.sigma_v,
                                          K0=1., E=1., nu=self.nu)
        self._E_ref = self._reference_base.E
        self._K_ref = coupling.condensed_boundary_block(self.mesh, self._reference_base)
        self._K_ref.setflags(write=False)
        self.n_forward = self.n_jac = 0
        self._configuration = (self.family, self.sigma_v, self.nu, self.phi, self.psi,
                               self.max_iter, self.min_substep, self.n_r, self.n_t, self.time_interpolation)
        self.identity = {"version": "explicit-anchors-cad-field-v3", "family": family,
            "layout_fingerprint": layout.fingerprint, "geometry_fingerprint": self.geom.fingerprint,
            "layout": layout.record, "snapshot_fingerprint": modules.snapshot["snapshot_fingerprint"],
            "parameter_source": parameter_source,
            "anchors": {"sigma_v_MPa": self.sigma_v, "nu": self.nu,
                        "phi_deg": float(phi_deg), "psi_deg": float(psi_deg)},
            "angle_scope": "instance-local forward and replay graphs; no global convention changes",
            "mesh": [self.n_r, self.n_t], "sigma_mm": self.sigma.tolist(),
            "section": self.stamps.section,
            "section_stamp0_day": self.stamps.stamp0,
            "stamps": {c: self.stamps.stamps[c].tolist() for c in CHANNELS},
            "raw_observations_mm": {c: self.stamps.y_mm[c].tolist() for c in CHANNELS},
            "observation_version": "raw-own-clock-increments; optional channel datum profiling is separate",
            "time_interpolation": interpolation_identity(time_interpolation),
            "times": self.times.tolist(), "lambdas": self.lams.tolist(),
            "release_basis": "prescribed scalar release law; not an independently observed construction history",
            "newton": self.newton_policy, "max_iter": self.max_iter, "newton_tol": 1e-11,
            "min_substep": self.min_substep, "visc_H": 0. if family == "cwfs" else None,
            "channel_operator_sha256": hashlib.sha256(self.W.tobytes()).hexdigest(),
            "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        # A fully bound new-CAD mechanical history can be observed by both
        # temporal operators for a controlled convergence comparison. Final
        # result identities still bind the selected operator and raw data.
        mechanical_keys = ("family", "layout_fingerprint", "geometry_fingerprint", "snapshot_fingerprint",
                           "anchors", "mesh", "times", "lambdas", "newton", "max_iter", "newton_tol",
                           "min_substep", "visc_H", "code_sha256")
        self.mechanical_identity = {key: self.identity[key] for key in mechanical_keys}
        self.mechanical_fingerprint = digest(self.mechanical_identity)
        self.identity["mechanical_fingerprint"] = self.mechanical_fingerprint
        self.fingerprint = digest(self.identity)

    def scenario_of(self, theta):
        names = RHEO_COORDINATES if self.family == "rheo" else CWFS_COORDINATES
        unknown = set(theta) - set(names) - {"lam0"}
        if unknown:
            raise ValueError(f"Unexpected or case-fixed coordinates: {sorted(unknown)}")
        required = names[:5] if self.family == "rheo" else names
        if any(name not in theta for name in (*required, "lam0")):
            raise ValueError("Every required quotient coordinate and lam0 must be explicit")
        parameters = {name: None if theta.get(name) is None else float(theta[name]) for name in names}
        if any(not np.isfinite(value) for value in parameters.values() if value is not None):
            raise ValueError("Quotient coordinates must be finite")
        positive = set(names) - {"dphi", "r_s", "tau_vp"}
        if any(parameters[name] <= 0 for name in positive):
            raise ValueError("Modulus, stress ratio, time and strain coordinates must be positive")
        if self.family == "rheo":
            if (parameters["r_s"] is None) != (parameters["tau_vp"] is None):
                raise ValueError("Supply both r_s and tau_vp, or omit both to disable the viscoplastic leg")
            if parameters["r_s"] is not None and min(parameters["r_s"], parameters["tau_vp"]) <= 0:
                raise ValueError("Viscoplastic threshold and time must be positive")
        elif (parameters["chi_c"] > 1 or parameters["dphi"] < 0
              or self.phi + parameters["dphi"] >= np.pi/2):
            raise ValueError("Invalid residual cohesion ratio or friction strengthening")
        from exec8_src import schedule
        phase = float(theta["lam0"])
        if not np.isfinite(phase) or not schedule.LAM_F < phase < 1:
            raise ValueError("lam0 must lie strictly between initial release and full release")
        return _Scenario(self.family, parameters, self.sigma_v, self.nu, self.phi, self.psi), phase

    def _system(self, scenario):
        from fedev_src.noncirc import NonCircFESystem
        return NonCircFESystem(self.mesh, scenario.base(a=self.geom.R), scenario.material())

    def forward(self, theta):
        from fedev_src import coupling
        self._check_configuration()
        scenario, _ = self.scenario_of(theta)
        system = self._system(scenario)
        boundary = (scenario.E / self._E_ref) * self._K_ref
        kwargs = {"times": self.times.copy()} if self.family == "rheo" else {}
        # Run arrays are owned by the FE history; the immutable case axis stays
        # separate from torch.as_tensor's shared-memory replay inputs.
        run, _ = coupling.run_condensed(system, self.lams.copy(), K_bnd=boundary,
            newton_tol=1e-11, max_iter=self.max_iter, min_substep=self.min_substep, **kwargs)
        run.field_evaluator_fingerprint = self.fingerprint
        run.field_mechanical_fingerprint = self.mechanical_fingerprint
        run.field_theta = dict(theta)
        self.n_forward += 1
        return run

    def _check_configuration(self):
        current = (self.family, self.sigma_v, self.nu, self.phi, self.psi,
                   self.max_iter, self.min_substep, self.n_r, self.n_t, self.time_interpolation)
        if (current != self._configuration or digest(self.identity) != self.fingerprint
                or digest(self.mechanical_identity) != self.mechanical_fingerprint):
            raise ValueError("An evaluator configuration changed; construct a new versioned case")

    def _check_history(self, run, theta=None):
        self._check_configuration()
        if getattr(run, "field_mechanical_fingerprint", None) != self.mechanical_fingerprint:
            raise ValueError("FE history belongs to a different geometry or mechanical parameter case")
        physical_theta = lambda values: {key: value for key, value in values.items() if key != "lam0"}
        if theta is not None and physical_theta(run.field_theta) != physical_theta(theta):
            raise ValueError("Exact Jacobian requires the coordinates of the executed FE history")

    def _commit_times(self, run):
        if self.family == "rheo":
            return np.array([ex.trial.t for ex in run.executed])
        return np.asarray(self.modules.cwfs.commit_times(run, self.times))

    def model_np(self, run, lam0):
        self._check_history(run)
        times = self._commit_times(run)
        offset = float(self.modules.field.lam_inv(lam0))
        series = np.stack([self.W @ np.asarray(increment.trial.u) for increment in run.executed])
        output = []
        for column, channel in enumerate(CHANNELS):
            query = offset + (self.stamps.stamps[channel]-self.stamps.stamp0)
            values = interpolate_numpy(query, times, series[:, column], method=self.time_interpolation)
            output.append(values-values[0])
        return np.concatenate(output)

    def _observe_torch(self, times, series, phase):
        return self.obs.flat(self.obs.model(times, series, phase, validate=False))

    def model(self, theta):
        run = self.forward(theta)
        return self.model_np(run, float(theta["lam0"])), run

    def jacobian(self, run, theta, names, *, mode=None):
        self._check_history(run, theta)
        scenario, phase = self.scenario_of(theta)
        # Validate the exact numeric history and phase before the AD transform.
        # Tensor-value validation inside jacfwd would break its dual tensors.
        self.model_np(run, phase)
        names = tuple(names)
        if len(set(names)) != len(names) or any(name not in theta for name in names):
            raise ValueError("Derivative coordinates must be unique supplied physical quotient coordinates")
        if any(theta[name] is None for name in names):
            raise ValueError("A disabled coordinate has no free derivative")
        mode = ("forward" if self.family == "rheo" else "reverse") if mode is None else mode
        from exec19_src.directional import jacobian_columns
        from exec19_src.replay19 import replay_visc
        from fedev_src.meshing import DTYPE

        def evaluate(inputs):
            physical = _physical_graph(scenario, {n: value for n, value in inputs.items() if n != "lam0"})
            if self.family == "rheo":
                commit_times, displacements, _ = replay_visc(run, physical)
            else:
                displacements, _ = self.modules.cwfs.replay_cwfs(run, physical)
                commit_times = self._commit_times(run)
            phase_tensor = inputs.get("lam0", torch.tensor(phase, dtype=DTYPE))
            return self._observe_torch(torch.tensor(commit_times, dtype=DTYPE),
                                       self.obs.channel_series(displacements), phase_tensor)

        if mode == "forward" or not names:
            values, jacobian = jacobian_columns(evaluate, {n: theta[n] for n in names})
        elif mode == "reverse":
            inputs = {n: torch.tensor(float(theta[n]), dtype=DTYPE, requires_grad=True) for n in names}
            values_tensor = evaluate(inputs)
            rows = []
            for value in values_tensor:
                gradient = torch.autograd.grad(value, tuple(inputs.values()), retain_graph=True, allow_unused=True)
                rows.append([0. if derivative is None else float(derivative) for derivative in gradient])
            values, jacobian = values_tensor.detach().numpy(), np.asarray(rows)
        else:
            raise ValueError("Select forward or reverse differentiation")
        self.n_jac += 1
        return values, jacobian

    def split_by_channel(self, flat):
        return self.modules.invert.FieldEvaluator.split_by_channel(self, flat)


def create_evaluator(layout, *, profile_datums=False, **kwargs):
    """Optionally profile additive channel offsets against unchanged raw data."""
    evaluator = CadFieldEvaluator(layout, **kwargs)
    if not profile_datums:
        return evaluator
    from .datums import ProfiledDatumEvaluator
    wrapped = ProfiledDatumEvaluator(evaluator, CHANNELS)
    wrapped.identity = {**evaluator.identity, "observation_version": wrapped.observation_version,
        "datum_profile_code_sha256": hashlib.sha256(Path(__file__).with_name("datums.py").read_bytes()).hexdigest()}
    wrapped.fingerprint = digest(wrapped.identity)
    return wrapped
