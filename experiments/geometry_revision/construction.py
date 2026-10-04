"""Dated boundary release and stress-free, staged frame support.

The annular rock mesh is retained. Unexcavated lower-boundary segments carry
their original confinement until bench passage. This is an equivalent boundary
model, not explicit removal of an interior bench mesh. Support is a bonded
Euler--Bernoulli frame with shared rotations, condensed onto rock translations.
Each newly installed segment takes the current displacement AND rotation as
its stress-free reference. Implicit AD includes those installation references.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib

import numpy as np
import scipy.sparse as sp
import torch

from .baseline import digest, load_modules
from .cad_geometry import CHANNELS
from .datums import ProfiledDatumEvaluator
from .field_evaluator import CadFieldEvaluator, _physical_graph
from . import rheo_material_policy
from .temporal import interpolate_numpy, interpolate_torch


@dataclass(frozen=True)
class Construction:
    lower_day: float = 7.
    closure_day: float = 15.
    casting_day: float = 38.
    delay_days: float = .5
    bench_depth_m: float = 6.
    support_stiffness_MPa_per_m: float = 3.82
    support_reference_radius_m: float = 4.9
    primary_thickness_m: float = .29
    invert_thickness_m: float = .40
    invert_modulus_MPa: float = 30000.
    support: bool = True
    casting: bool = True
    step_days: float = 2.

    def __post_init__(self):
        if not 0 < self.delay_days < self.lower_day < self.closure_day < self.casting_day:
            raise ValueError("Construction dates and the assumed first-reading delay are inconsistent")
        if min(self.bench_depth_m, self.support_stiffness_MPa_per_m,
               self.primary_thickness_m, self.step_days) <= 0:
            raise ValueError("Construction dimensions, stiffness and time step must be positive")


def frame_matrix(xy, selected, EA, EI):
    """Assemble frame energy in [ux,uy per node; rotation per node]."""
    n = len(xy)
    K = np.zeros((3*n, 3*n))
    for i in np.flatnonzero(selected):
        j = (i+1) % n
        delta = xy[j]-xy[i]
        L = np.linalg.norm(delta)
        c, s = delta/L
        local = np.zeros((6, 6))
        local[np.ix_([0, 3], [0, 3])] = EA/L*np.array([[1., -1.], [-1., 1.]])
        local[np.ix_([1, 2, 4, 5], [1, 2, 4, 5])] = EI/L**3*np.array([
            [12, 6*L, -12, 6*L], [6*L, 4*L**2, -6*L, 2*L**2],
            [-12, -6*L, 12, -6*L], [6*L, 2*L**2, -6*L, 4*L**2]])
        rotation = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1.]])
        transform = np.zeros((6, 6))
        transform[:3, :3] = transform[3:, 3:] = rotation
        ids = [2*i, 2*i+1, 2*n+i, 2*j, 2*j+1, 2*n+j]
        K[np.ix_(ids, ids)] += transform.T@local@transform
    return K


def condense_frame(K):
    n = len(K)//3
    rr = K[2*n:, 2*n:]
    active = np.flatnonzero(np.diag(rr) > 0)
    inverse = np.zeros_like(rr)
    if len(active):
        inverse[np.ix_(active, active)] = np.linalg.inv(rr[np.ix_(active, active)])
    ru = inverse@K[2*n:, :2*n]
    reduced = K[:2*n, :2*n]-K[:2*n, 2*n:]@ru
    return dict(K=K, inverse=inverse, ru=ru, reduced=.5*(reduced+reduced.T))


def segmented_load_basis(mesh, bench_y):
    """Consistent three-point boundary quadrature, split by physical height."""
    from fedev_src.meshing import G3_P, G3_W
    nodes = mesh.ring_nodes("inner")
    nxt = np.roll(nodes, -1)
    x0, x1 = mesh.xy[nodes], mesh.xy[nxt]
    lengths = np.linalg.norm(x1-x0, axis=1)
    th0 = mesh.theta_nodes
    th1 = np.r_[th0[1:], 2*np.pi]
    basis = np.zeros((2, 2, mesh.n_dof))
    for p, weight in zip(G3_P, G3_W):
        n0, n1 = .5*(1-p), .5*(1+p)
        th = n0*th0+n1*th1
        normal = mesh.section_map.wall_normals(th)
        upper = mesh.section_map.wall_points(th).imag >= bench_y
        for region, mask in enumerate((upper, ~upper)):
            for component, direction in enumerate((normal.real, normal.imag)):
                force = -weight*.5*lengths*direction*mask
                np.add.at(basis[region, component], 2*nodes+component, n0*force)
                np.add.at(basis[region, component], 2*nxt+component, n1*force)
    return basis


class ConstructionEvaluator(CadFieldEvaluator):
    def __init__(self, layout, *, construction, **kwargs):
        self.construction = construction
        super().__init__(layout, **kwargs)
        cfg = construction
        crown = layout.record["depth_datum_y_m"]
        bench_y = crown-cfg.bench_depth_m
        self.load_basis = segmented_load_basis(self.mesh, bench_y)
        nodes = self.mesh.ring_nodes("inner")
        self.inner = self.mesh.ring_dofs("inner")
        xy = self.mesh.xy[nodes]
        middle = .5*(xy+np.roll(xy, -1, axis=0))
        upper = middle[:, 1] >= bench_y
        # The CAD invert joins the two knee arcs at this ordinate.
        cad = load_modules().cad.boundary("initial_support_outer")
        invert_arc = next(a for a in cad["circular_arcs"] if a["arc_id"] == "invert")
        invert_y = invert_arc["center"][1]+invert_arc["radius"]*np.sin(invert_arc["start_rad"])
        invert_y = layout.record["scale"]*invert_y+layout.record["translation_m"][1]
        invert = middle[:, 1] < invert_y
        lower = ~upper & ~invert
        EA = cfg.support_stiffness_MPa_per_m*cfg.support_reference_radius_m**2
        EI = EA*cfg.primary_thickness_m**2/12
        primary = [frame_matrix(xy, mask, EA, EI) for mask in (upper, lower, invert)]
        cast_EA = cfg.invert_modulus_MPa/(1-.2**2)*cfg.invert_thickness_m
        cast = frame_matrix(xy, invert, cast_EA, cast_EA*cfg.invert_thickness_m**2/12)
        self.groups = primary+[cast]
        self.install_times = np.array([cfg.delay_days, cfg.delay_days+cfg.lower_day+.5,
            cfg.delay_days+cfg.closure_day, cfg.delay_days+cfg.casting_day])
        if not cfg.support:
            self.groups = [np.zeros_like(K) for K in self.groups]
        if not cfg.casting:
            self.groups[-1] *= 0
        self.frames = [condense_frame(sum(self.groups[:i], np.zeros_like(cast))) for i in range(5)]
        n = self.mesh.n_dof
        self.support_matrices = []
        for frame in self.frames:
            row, col = np.meshgrid(self.inner, self.inner, indexing="ij")
            self.support_matrices.append(sp.csc_matrix((frame["reduced"].ravel(),
                (row.ravel(), col.ravel())), shape=(n, n)))
        _, material = rheo_material_policy.build_return()
        self.mechanical_identity = {**self.mechanical_identity, "construction": vars(cfg),
            "material_policy": material, "construction_code": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        self.mechanical_fingerprint = digest(self.mechanical_identity)
        self.identity = {**self.identity, "construction": vars(cfg), "mechanical_fingerprint": self.mechanical_fingerprint,
            "version": "dated-segment-release-installed-frame-v1",
            "observation_clock": "calendar events relative to monitoring date; digitized abscissae remain graphical day coordinates",
            "release_basis": "spatially partitioned prescribed effective release; no inferred construction measurements",
            "support_model": "bonded equivalent frame, staged stress-free installation, condensed shared rotations"}
        self.fingerprint = digest(self.identity)

    def releases(self, t):
        from exec19_src.event19 import lam_ramped
        return np.asarray(lam_ramped(np.array([t, t-self.construction.delay_days-self.construction.lower_day])))

    def activate(self, count, t, u, reference):
        while count < 4 and self.install_times[count] <= t+1e-9:
            frame = self.frames[count]
            rotation = frame["inverse"]@reference[len(u):]-frame["ru"]@u
            reference = reference+self.groups[count]@np.r_[u, rotation]
            count += 1
        return count, reference

    def forward(self, theta):
        from fedev_src import solver as fe, coupling
        from types import MethodType
        self._check_configuration()
        scenario, _ = self.scenario_of(theta)
        system = self._system(scenario)
        old_residual, old_jacobian = system.mode_residual, system.mode_jacobian
        system.stage_matrix = self.support_matrices[0]
        system.stage_reference = np.zeros(system.mesh.n_dof)
        def residual(this, mode, bulk, u, omega):
            return old_residual(mode, bulk+this.stage_matrix@u-this.stage_reference, u, omega)
        def jacobian(this, mode, K):
            return old_jacobian(mode, K+this.stage_matrix)
        system.mode_residual = MethodType(residual, system)
        system.mode_jacobian = MethodType(jacobian, system)
        boundary = scenario.E/self._E_ref*self._K_ref
        bc = fe.CondensedOuter(boundary)
        committed, executed, failures = system.initial_state(), [], []
        reference = np.zeros(3*self.n_t)
        count = 0
        with rheo_material_policy.active():
            for k, (ta, tb) in enumerate(zip(self.times[:-1], self.times[1:])):
                count, reference = self.activate(count, ta, committed.u[self.inner], reference)
                frame = self.frames[count]
                support_load = reference[:2*self.n_t]-frame["K"][:2*self.n_t, 2*self.n_t:]@frame["inverse"]@reference[2*self.n_t:]
                system.stage_matrix = self.support_matrices[count]
                system.stage_reference = np.zeros(self.mesh.n_dof)
                system.stage_reference[self.inner] = support_load
                fraction, step = 0., 1.
                while fraction < 1-1e-12:
                    end = min(fraction+step, 1.)
                    t = ta+end*(tb-ta)
                    release = self.releases(t)
                    system.f1 = scenario.sigma_v*np.einsum("r,rd->d", release,
                        scenario.parameters["K0"]*self.load_basis[:, 0]+self.load_basis[:, 1])
                    trial = fe.trial_increment(system, committed, 1., bc, newton_tol=1e-11,
                        max_iter=self.max_iter, dt=(end-fraction)*(tb-ta))
                    if not trial.converged:
                        failures.append(dict(knot=k, t=t, diagnostics=trial.diagnostics))
                        step *= .5
                        if step < self.min_substep:
                            raise fe.IncrementFailure("Staged increment failed", failures[-1])
                        continue
                    trial.mode.update(support_count=count, support_reference=reference.copy(), releases=release)
                    executed.append(fe.ExecutedIncrement(k, fraction, end, committed, trial))
                    committed = fe.commit(trial)
                    fraction = end
        run = fe.RunResult(system, self.lams.copy(), [bc]*len(self.times), executed,
            committed, self.times.copy(), failures)
        front = coupling.plastic_front(self.mesh, committed.kappa)
        if front[-1]:
            raise coupling.CouplingFailure("Plastic deformation reached the elastic far boundary")
        run.field_evaluator_fingerprint, run.field_mechanical_fingerprint = self.fingerprint, self.mechanical_fingerprint
        run.field_theta = dict(theta)
        run.plastic_front = front
        self.n_forward += 1
        return run

    def model_np(self, run, lam0):
        self._check_history(run)
        series = np.stack([self.W@step.trial.u for step in run.executed])
        times = self._commit_times(run)
        values = []
        for col, c in enumerate(CHANNELS):
            query = self.construction.delay_days+self.stamps.stamps[c]-self.stamps.stamp0
            pred = interpolate_numpy(query, times, series[:, col], method=self.time_interpolation)
            values.append(pred-pred[0])
        return np.concatenate(values)

    def jacobian(self, run, theta, names, *, mode=None):
        from exec19_src.directional import jacobian_columns, lu_solve_t
        from fedev_src import assembly, material_visc
        self._check_history(run, theta)
        if "lam0" in names:
            raise ValueError("Construction observation delay is a fixed assumption, not a fitted phase")
        scenario, _ = self.scenario_of(theta)
        tensor = lambda a: torch.as_tensor(np.array(a), dtype=torch.float64)
        inner = torch.tensor(self.inner, dtype=torch.long)
        outer = self.mesh.ring_dofs_outer_t
        frames = [{k: tensor(v) for k, v in f.items()} for f in self.frames]
        groups = [tensor(K) for K in self.groups]
        W = tensor(self.W)
        basis = tensor(self.load_basis)
        def evaluate(inputs):
            phys = _physical_graph(scenario, inputs)
            E, nu, sv, k0, mat = [phys[k] for k in ("E", "nu", "sigma_v", "K0", "mat")]
            zero = E*0
            sig0 = torch.stack([-k0*sv, -sv, -k0*sv, zero])
            sig = sig0.unsqueeze(0).expand(self.mesh.n_qp, 4)
            kelvin = torch.zeros(self.mesh.n_qp, 4, dtype=torch.float64)
            kappa = torch.zeros(self.mesh.n_qp, dtype=torch.float64)
            previous = torch.zeros(self.mesh.n_dof, dtype=torch.float64)
            reference = torch.zeros(3*self.n_t, dtype=torch.float64)
            count, histories = 0, []
            for ex in run.executed:
                tr = ex.trial
                while count < tr.mode["support_count"]:
                    f = frames[count]
                    u = previous[inner]
                    rot = f["inverse"]@reference[2*self.n_t:]-f["ru"]@u
                    reference = reference+groups[count]@torch.cat((u, rot))
                    count += 1
                f = frames[count]
                udet = tensor(tr.u)
                dt = tr.t-ex.committed_before.t
                deps = self.mesh.qp_strains_t(udet-previous)
                stress = material_visc.integrate_visc(deps, sig, kelvin, kappa, dt, E, nu, mat, run.system.kind)[0]
                force = sv*torch.einsum("r,rd->d", tensor(tr.mode["releases"]), k0*basis[:, 0]+basis[:, 1])
                residual = assembly.f_int_qp_t(self.mesh, stress[:, [0, 1, 3]]-sig0[[0, 1, 3]])-force
                support = f["reduced"]@udet[inner]-reference[:2*self.n_t]+f["K"][:2*self.n_t, 2*self.n_t:]@f["inverse"]@reference[2*self.n_t:]
                residual = residual.index_add(0, inner, support)
                boundary = tensor(tr.mode["K_bnd"])*E/scenario.E
                residual = residual.index_add(0, outer, boundary@udet[outer])
                C = tensor(run.system.rigid[:, :2])
                augmented = torch.cat((residual+C@tensor(tr.mode["omega"]), C.T@udet))
                unew = udet-lu_solve_t(tr.lu, augmented-augmented.detach())[:self.mesh.n_dof]
                out = material_visc.integrate_visc(self.mesh.qp_strains_t(unew-previous), sig, kelvin,
                    kappa, dt, E, nu, mat, run.system.kind)
                sig, kelvin, kappa = out[0], out[3], kappa+out[4]
                previous = unew
                histories.append(W@unew)
            series = torch.stack(histories)
            times = tensor(self._commit_times(run))
            output = []
            for col, c in enumerate(CHANNELS):
                query = tensor(self.construction.delay_days+self.stamps.stamps[c]-self.stamps.stamp0)
                pred = interpolate_torch(query, times, series[:, col], method=self.time_interpolation, validate=False)
                output.append(pred-pred[0])
            return torch.cat(output)
        with rheo_material_policy.active():
            values, jac = jacobian_columns(evaluate, {n: theta[n] for n in names})
        self.n_jac += 1
        return values, jac


def create_evaluator(layout, *, construction, **kwargs):
    base = ConstructionEvaluator(layout, construction=construction, **kwargs)
    wrapped = ProfiledDatumEvaluator(base, CHANNELS)
    wrapped.identity = {**base.identity, "observation_version": wrapped.observation_version}
    wrapped.fingerprint = digest(wrapped.identity)
    return wrapped
