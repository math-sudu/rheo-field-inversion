
from __future__ import annotations

import dataclasses

import numpy as np
import torch

from . import _paths  # noqa: F401
from .field19 import CHANNELS, FieldObservation
from .replay19 import theta_graph, replay_visc

from fedev_src.meshing import DTYPE                        # noqa: E402
from exec8_src.information import coordinate_crb, fisher_condition

SV_THRESHOLD = 3.0        # exec8 identify convention (inherited)

# quotient boxes for standardization halfwidths (PROTOCOL section 4;
# self-set scenario boxes declared there + inherited exec8 boxes)
BOX = {
    "s_star": (1.0e-2, 6.25e-2),
    "K0": (0.69, 1.18),
    "lam0": (0.35, 0.85),
    "r_K": (0.1, 1.0),
    "tau_K": (0.2, 10.0),
    "tau_M": (50.0, 2000.0),
    "r_s": (0.8, 3.0),
    "tau_vp": (5.0, 300.0),
}

LADDER = {
    "L0": ("s_star", "K0", "lam0"),
    "L1": ("s_star", "K0", "lam0", "r_K", "tau_K"),
    "L2": ("s_star", "K0", "lam0", "r_K", "tau_K", "tau_M"),
    "L3": ("s_star", "K0", "lam0", "r_K", "tau_K", "tau_M", "r_s",
           "tau_vp"),
}


@dataclasses.dataclass
class Spectrum:
    names: tuple
    singular_values: np.ndarray
    fisher_cond: float
    crb_scaled: np.ndarray      # box-halfwidth units
    crb_abs: np.ndarray
    estimable: tuple
    dropped: tuple


def field_jacobian(run, scn, lam0, names, obs: FieldObservation, mode="forward"):
    """(model_vec, J) of the flat own-clock observation in the
    coordinates ``names`` (lam0 handled as an extra leaf)."""
    if mode == "forward":
        from .directional import jacobian_columns
        def evaluate(inputs):
            _, phys = theta_graph(scn, overrides={n:v for n,v in inputs.items() if n != "lam0"})
            t_list, u_list, _ = replay_visc(run, phys)
            phase = inputs.get("lam0", torch.tensor(float(lam0), dtype=DTYPE))
            return obs.flat(obs.model(torch.tensor(t_list, dtype=DTYPE),
                                      obs.channel_series(u_list), phase))
        return jacobian_columns(evaluate, {n: (lam0 if n == "lam0" else getattr(scn, n)) for n in names})
    if mode != "reverse":
        raise ValueError(mode)
    q_names = tuple(n for n in names if n != "lam0")
    leaves, phys = theta_graph(scn, requires_grad=q_names)
    lam0_t = torch.tensor(float(lam0), dtype=DTYPE,
                          requires_grad=("lam0" in names))
    t_list, u_list, _ = replay_visc(run, phys)
    U = obs.channel_series(u_list)
    t_commit = torch.tensor(t_list, dtype=DTYPE)
    m = obs.model(t_commit, U, lam0_t)
    flat = obs.flat(m)
    params = [(leaves[n] if n != "lam0" else lam0_t) for n in names]
    J = np.zeros((len(flat), len(names)))
    vals = flat.detach().numpy()
    for i in range(len(flat)):
        grads = torch.autograd.grad(flat[i], params,
                                    retain_graph=(i < len(flat) - 1),
                                    allow_unused=False)
        J[i] = [float(g) for g in grads]
    return vals, J


def sigma_vector(obs: FieldObservation):
    out = []
    for ch in CHANNELS:
        out.append(np.full(len(obs.stamps.stamps[ch]), obs.sig[ch]))
    return np.concatenate(out)


def analyze(J, names, sigma, box=None):
    """exec8-convention spectrum analysis on a raw Jacobian."""
    box = BOX if box is None else box
    half = np.array([(box[n][1] - box[n][0]) / 2.0 for n in names])
    J_std = (J / sigma[:, None]) * half[None, :]
    sv = np.linalg.svd(J_std, compute_uv=False)
    F = J_std.T @ J_std
    crb_scaled = coordinate_crb(J_std)
    crb_abs = crb_scaled * half
    cond = fisher_condition(J_std)
    # greedy minimal estimable set at the 3-sigma direction threshold
    est = []
    remaining = list(range(len(names)))
    while remaining:
        best, best_sv = None, -1.0
        for j in remaining:
            cols = est + [j]
            sv_min = np.linalg.svd(J_std[:, cols],
                                   compute_uv=False)[-1]
            if sv_min > best_sv:
                best, best_sv = j, sv_min
        if best_sv < SV_THRESHOLD:
            break
        est.append(best)
        remaining.remove(best)
    est_names = tuple(names[j] for j in est)
    dropped = tuple(n for n in names if n not in est_names)
    return Spectrum(names=tuple(names), singular_values=sv,
                    fisher_cond=cond, crb_scaled=crb_scaled,
                    crb_abs=crb_abs, estimable=est_names,
                    dropped=dropped)


def runA_scaling_residual(run, scn, lam0, obs: FieldObservation):
    """Multiplicative-degeneracy machine check (D8 analogue, Run A'):
    directional derivative of the observation along the uniform
    log-scaling of ALL stress-dimensioned parameters (sigma_v, G0,
    G_K, eta_K, eta_M, eta_vp, c) -- structurally zero; reported as a
    standardized residual against the typical gradient scale."""
    kappa = torch.tensor(1.0, dtype=DTYPE, requires_grad=True)
    _, phys0 = theta_graph(scn)
    phys = {"E": phys0["E"] * kappa, "nu": phys0["nu"],
            "sigma_v": phys0["sigma_v"] * kappa,
            "K0": phys0["K0"],
            "mat": {k: (v * kappa if k in ("c", "E_K", "eta_K",
                                           "eta_M", "eta_vp")
                        else v) for k, v in phys0["mat"].items()}}
    lam0_t = torch.tensor(float(lam0), dtype=DTYPE)
    t_list, u_list, _ = replay_visc(run, phys)
    U = obs.channel_series(u_list)
    m = obs.model(torch.tensor(t_list, dtype=DTYPE), U, lam0_t)
    flat = obs.flat(m)
    g = torch.autograd.grad(flat.sum(), kappa)[0]
    # scale: typical magnitude of d(sum obs)/d(log s_star)
    _, J = field_jacobian(run, scn, lam0, ("s_star",), obs)
    scale = abs(float(np.sum(J[:, 0])) * scn.s_star)
    return abs(float(g)) / max(scale, 1e-300)
