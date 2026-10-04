
from dataclasses import dataclass

import numpy as np

from . import solver
from .meshing import G3_P, G3_W, Mesh


# --------------------------------------------------------------- map twin

@dataclass(frozen=True)
class LaurentMap:
    """Exterior Laurent map omega(zeta) = R (zeta + sum c_j zeta^{-j}).

    Data container + evaluation twin (module docstring: form authority
    is exec6b ``general_map.GeneralMappedKM``; runner cross-checked).

    Parameters
    ----------
    R : float
        Mapping radius (> 0); for the circle c = (0,) it is the hole
        radius.
    c : tuple of complex
        Map coefficients (c[0] is a pure translation); at least the
        translation entry.  Univalence is NOT enforced here -- gate
        candidate maps through the exec6b acceptance
        (``general_map.validate_univalence``) upstream; the mesh
        constructor additionally guards element-Jacobian positivity.
    """

    R: float
    c: tuple

    def __post_init__(self):
        if not self.R > 0.0:
            raise ValueError("mapping radius R must be positive")
        cc = tuple(complex(v) for v in np.asarray(self.c).ravel())
        if len(cc) < 1:
            raise ValueError("c needs at least the translation entry c_0")
        object.__setattr__(self, "c", cc)

    @property
    def n_map(self):
        return len(self.c) - 1

    def omega(self, zeta):
        """z = omega(zeta) = R (zeta + sum_{j=0..n_map} c_j zeta^{-j})."""
        zeta = np.asarray(zeta, dtype=complex)
        inv = 1.0 / zeta
        s = np.zeros(zeta.shape, dtype=complex)
        p = np.ones(zeta.shape, dtype=complex)      # zeta^{-j} at j = 0
        for j in range(self.n_map + 1):
            s = s + self.c[j] * p
            p = p * inv
        return self.R * (zeta + s)

    def omega_prime(self, zeta):
        """omega'(zeta) = R (1 - sum_{j=1..n_map} j c_j zeta^{-(j+1)})."""
        zeta = np.asarray(zeta, dtype=complex)
        inv = 1.0 / zeta
        s = np.zeros(zeta.shape, dtype=complex)
        p = inv * inv                               # zeta^{-(j+1)} at j = 1
        for j in range(1, self.n_map + 1):
            s = s + (j * self.c[j]) * p
            p = p * inv
        return self.R * (1.0 - s)

    def wall_points(self, theta):
        """z on the wall image, zeta = e^{i theta}."""
        return self.omega(np.exp(1j * np.asarray(theta, dtype=float)))

    def wall_normals(self, theta):
        """Unit OUTWARD (tunnel -> rock) wall normal, complex form.

        nu = zeta omega'(zeta) / |zeta omega'(zeta)| at zeta =
        e^{i theta}: the image of the radial direction (family
        convention; ``general_map.boundary_traction`` normal).  For
        the circle map it is e^{i theta} = e_r exactly.
        """
        s = np.exp(1j * np.asarray(theta, dtype=float))
        n = s * self.omega_prime(s)
        return n / np.abs(n)

    def wall_radius_extrema(self, n_scan=4096):
        """(min |z|, max |z|) over a dense wall scan (origin distance)."""
        r = np.abs(self.wall_points(
            (2.0 * np.pi / n_scan) * np.arange(n_scan)))
        return float(r.min()), float(r.max())


# ----------------------------------------------------------- mapped mesh

class MappedAnnulusMesh(Mesh):
    """Structured Q4 mesh: mapped wall (inner) -> exact circle (outer).

    Node coordinates (ring i = 0..n_r, node j at zeta-angle
    theta_j = 2 pi j / n_t)::

        rho_i   = rho_out^{i/n_r},     rho_out = R_Gamma / R_map
        s_i     = (rho_i - 1) / (rho_out - 1)
        w_i     = s_i^2 (3 - 2 s_i)                   (smoothstep)
        z_(i,j) = (1 - w_i) omega(rho_i e^{i theta_j})
                  + w_i R_map rho_i e^{i theta_j}

    i.e. conformal-image coordinates near the wall (w'(0) = 0: the
    first layers follow the orthogonal mapped grid, physical cell
    size ~ |omega'| * d_rho -- naturally finer where the wall curves
    hardest), blended onto the leading-term circle R_map rho e^{i
    theta} so ring n_r is EXACTLY the circle R_Gamma at the uniform
    angles ``theta_nodes`` (pinned to machine precision) -- the outer
    ring every interface operator assumes.  Ring 0 is EXACTLY the
    mapped wall (w_0 = 0).  For the circle map c = (0,) the blend is
    a no-op and the mesh coincides with the circular
    ``Mesh(R_map, R_Gamma, n_r, n_t)`` (junction test).

    Topology, quadrature and torch precomputations are the parent's
    (fully isoparametric -- they read only ``self.xy`` and the
    connectivity); the constructor rebuilds them on the mapped
    coordinates.

    Validity guards (constructor):

    * clearance: R_Gamma >= clearance_min * max_theta |omega(e^{i
      theta})| (the section must sit inside Gamma with margin -- part
      of the circular-interface applicability criterion);
    * untangling: det J > 0 at ALL 2x2 Gauss points (the univalent
      map guarantees the pure conformal grid; the blend correction is
      additionally certified here).  ``min_detJ`` is kept as a mesh
      quality readout.

    ``a`` (parent attribute) is set to the mapping radius R_map -- the
    equivalent-circle scale used for r/a normalizations; ``radii``
    holds the nominal schedule R_map * rho_i.
    """

    def __init__(self, section_map, R_gamma, n_r, n_t,
                 clearance_min=1.15):
        R_map = float(section_map.R)
        if not R_gamma > R_map:
            raise ValueError("R_gamma must exceed the mapping radius")
        rmin, rmax = section_map.wall_radius_extrema()
        if R_gamma < clearance_min * rmax:
            raise ValueError(
                f"interface circle R_Gamma = {R_gamma:g} clears the "
                f"section (max wall radius {rmax:g}) by less than the "
                f"factor {clearance_min:g}: enlarge R_Gamma or drop "
                "to the non-circular-interface (Schwarz-fallback) "
                "architecture")
        # parent: topology + circular coordinates + precomputations
        super().__init__(R_map, R_gamma, n_r, n_t)
        # mapped coordinates (module docstring construction)
        rho_out = R_gamma / R_map
        rho = rho_out ** (np.arange(self.n_r + 1) / self.n_r)
        s = (rho - 1.0) / (rho_out - 1.0)
        w = s * s * (3.0 - 2.0 * s)
        zeta = rho[:, None] * np.exp(1j * self.theta_nodes[None, :])
        z = ((1.0 - w)[:, None] * section_map.omega(zeta)
             + w[:, None] * (R_map * zeta))
        z[0] = section_map.wall_points(self.theta_nodes)   # exact wall
        z[-1] = R_gamma * np.exp(1j * self.theta_nodes)    # exact circle
        self.section_map = section_map
        self.R_map = R_map
        self.wall_r_min, self.wall_r_max = rmin, rmax
        self.clearance_ratio = float(R_gamma / rmax)
        self.radii = R_map * rho
        self.rho_knots = rho
        flat = z.ravel()                                   # ring-major
        self.xy = np.column_stack([flat.real, flat.imag])
        self.r_node = np.abs(flat)
        self.th_node = np.angle(flat)
        # rebuild the isoparametric machinery on the mapped coordinates
        self._precompute_quadrature()
        self._precompute_torch()
        self.min_detJ = float(self.w_qp.min())
        if self.min_detJ <= 0.0:
            raise ValueError(
                "mapped mesh is tangled (min det J = "
                f"{self.min_detJ:g}); the blend correction broke the "
                "conformal grid -- enlarge R_Gamma or refine")


# ------------------------------------------------------- excavation load

def noncirc_inner_unit_load(mesh, sigma_v, K0):
    """Unit (lambda = 1) perturbation load on the mapped wall.

    Physics: the staged-excavation TOTAL traction on the wall is
    (1 - lambda) * (sigma0 . n) with n the domain outward normal
    (pointing INTO the tunnel), so the perturbation load is
    lambda * (sigma0 . nu) with nu = -n the unit normal pointing FROM
    the tunnel INTO the rock -- for the circle nu = e_r, reproducing
    ``assembly.inner_unit_load`` exactly in the continuum.

    Discretization: 3-point Gauss on each inner-ring chord (the
    parent's consistent-ring-load rule, chord length measure); the
    traction is evaluated with the ANALYTIC wall normal
    nu(theta) = zeta omega' / |zeta omega'| at the chord-Gauss
    parameter mapped linearly between the two nodal
    boundary-correspondence angles theta_j, theta_{j+1} -- the mapped
    analogue of the circular implementation's analytic arc normal at
    the chord points.  sigma0 is the tension-positive in-situ stress
    of ``params.BaseParams.sigma0`` (s_xy0 = 0).
    """
    sm = mesh.section_map
    n_t = mesh.n_t
    sxx0, syy0 = -K0 * sigma_v, -sigma_v
    nodes = mesh.ring_nodes("inner")
    n0 = nodes
    n1 = nodes[np.r_[1:n_t, 0]]
    x0, x1 = mesh.xy[n0], mesh.xy[n1]
    L = np.hypot(*(x1 - x0).T)
    th0 = mesh.theta_nodes
    th1 = np.concatenate([mesh.theta_nodes[1:], [2.0 * np.pi]])
    f = np.zeros(mesh.n_dof)
    for p, wg in zip(G3_P, G3_W):
        N0, N1 = 0.5 * (1.0 - p), 0.5 * (1.0 + p)
        th = N0 * th0 + N1 * th1
        nu = sm.wall_normals(th)
        tx = sxx0 * nu.real
        ty = syy0 * nu.imag
        cw = wg * 0.5 * L
        np.add.at(f, 2 * n0, cw * N0 * tx)
        np.add.at(f, 2 * n0 + 1, cw * N0 * ty)
        np.add.at(f, 2 * n1, cw * N1 * tx)
        np.add.at(f, 2 * n1 + 1, cw * N1 * ty)
    return f


# ---------------------------------------------------------------- system

class NonCircFESystem(solver.FESystem):
    """FESystem on a :class:`MappedAnnulusMesh`.

    The ONLY change against the parent is the excavation unit load:
    the parent's ``assembly.inner_unit_load`` applies the in-situ
    traction along the RADIAL direction (exact for a circular wall
    only); here it is replaced by the mapped-wall normal-field load
    :func:`noncirc_inner_unit_load`.  Every other operator --
    material bridge, mode setup/residual/jacobian, outer-ring
    interface I/O, initial state -- is geometry-agnostic given the
    mesh and inherited unchanged.

    ``noncircular = True`` marks the system for the ``diff.replay``
    rejection (replay rebuilds the load through the circular twin and
    would silently mis-differentiate a mapped-wall run).
    """

    noncircular = True

    def __init__(self, mesh, base, mat):
        if not isinstance(mesh, MappedAnnulusMesh):
            raise TypeError("NonCircFESystem needs a MappedAnnulusMesh")
        super().__init__(mesh, base, mat)
        self.f1 = noncirc_inner_unit_load(mesh, base.sigma_v, base.K0)


# ------------------------------------------------------------ wall QoIs

def wall_disp_metrics(u, mesh):
    """Wall-convergence readouts on the (possibly non-circular) wall.

    Returns a dict of INWARD-POSITIVE displacement measures over the
    inner-ring nodes of ``mesh``:

    * ``u_wall_radial_mean``: mean of -u_r with u_r the radial
      component at the ACTUAL node angles atan2(y, x) (reduces to
      ``-solver.mean_inner_radial_disp`` on circular meshes, where
      node angles equal ``theta_nodes``);
    * ``u_wall_normal_mean``: mean of -u.nu with nu the analytic
      outward wall normal (coincides with the radial measure on the
      circle);
    * ``u_wall_normal_max``: max of -u.nu (peak wall convergence --
      corner-sensitive, the measure the mean hides).
    """
    idx = mesh.ring_dofs("inner")
    ux, uy = np.asarray(u)[idx[0::2]], np.asarray(u)[idx[1::2]]
    xy = mesh.xy[mesh.ring_nodes("inner")]
    phi = np.arctan2(xy[:, 1], xy[:, 0])
    u_r = ux * np.cos(phi) + uy * np.sin(phi)
    if isinstance(mesh, MappedAnnulusMesh):
        nu = mesh.section_map.wall_normals(mesh.theta_nodes)
        u_n = ux * nu.real + uy * nu.imag
    else:
        u_n = u_r
    return {
        "u_wall_radial_mean": float(-np.mean(u_r)),
        "u_wall_normal_mean": float(-np.mean(u_n)),
        "u_wall_normal_max": float(np.max(-u_n)),
    }
