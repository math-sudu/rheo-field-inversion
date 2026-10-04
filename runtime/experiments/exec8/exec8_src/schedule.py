
from __future__ import annotations

import numpy as np
import torch


LAM_F = 0.30
BETA = 1.178
T_REL = 7.576  # days


def release_increment(dt_day, lam0, *, lam_f: float = LAM_F,
                      beta: float = BETA, t_rel: float = T_REL):
    """g(dt; lambda0) = lambda(t0+dt) - lambda(t0); torch or numpy.

    ``dt_day`` >= 0 (days since monitoring start), ``lam0`` scalar in
    [lam_f, 1).  Returns the same backend type as the inputs.
    """
    if isinstance(dt_day, torch.Tensor) or isinstance(lam0, torch.Tensor):
        dt = torch.as_tensor(dt_day)
        l0 = torch.as_tensor(lam0)
        x0 = torch.log((1.0 - lam_f) / (1.0 - l0)) ** (1.0 / beta)
        return (1.0 - l0) * (1.0 - torch.exp(
            x0 ** beta - ((x0 * t_rel + dt) / t_rel) ** beta))
    dt = np.asarray(dt_day, dtype=float)
    x0 = np.log((1.0 - lam_f) / (1.0 - float(lam0))) ** (1.0 / beta)
    return (1.0 - float(lam0)) * (1.0 - np.exp(
        x0 ** beta - ((x0 * t_rel + dt) / t_rel) ** beta))


def release_increment_dlam0(dt_day, lam0, *, lam_f: float = LAM_F,
                            beta: float = BETA, t_rel: float = T_REL):
    """Closed-form d g / d lambda0 (numpy).

    With X = x0 + dt/T and E = exp(x0**B - X**B):
    d(x0**B)/dl = 1/(1-l);  d(X**B)/dl = X**(B-1) x0**(1-B) / (1-l);
    dE/dl = E/(1-l) * (1 - (X/x0)**(B-1));
    dg/dl = -(1-E) - (1-l) dE/dl = -1 + E * (X/x0)**(B-1).
    The beta = 1 limit gives -(1 - e^{-dt/T}) (E then lambda0-free),
    consistent with the exact scale-merge degeneracy of the plain
    exponential (module docstring).  Verified against torch autograd
    of :func:`release_increment` in tests (<= 1e-10 relative).
    """
    dt = np.asarray(dt_day, dtype=float)
    l0 = float(lam0)
    x0 = np.log((1.0 - lam_f) / (1.0 - l0)) ** (1.0 / beta)
    X = x0 + dt / t_rel
    E = np.exp(x0 ** beta - X ** beta)
    return -1.0 + E * (X / x0) ** (beta - 1.0)
