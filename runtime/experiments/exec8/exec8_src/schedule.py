"""Frozen release schedule of the synthetic generator (kickoff freeze).

Global release law (compressed exponential in time)
---------------------------------------------------
    lambda(t) = 1 - (1 - LAM_F) * exp(-((t - t_p) / T_REL)**BETA),
    t >= t_p,

with lambda the classical convergence-confinement release fraction
(0 at undisturbed state, -> 1 at full release), t_p the face-passage
time of the section and (LAM_F, T_REL, BETA) FROZEN constants:

* BETA = 1.178, T_REL = 7.576 d -- the unique two-parameter member
  matching BOTH measured chord medians t50 = 5.550 d and
  t90 = 15.368 d of the six doc290 convergence chords (stage-1
  calibration CSV).  Derivation: (t50/T)**B = ln 2 and
  (t90/T)**B = ln 10 give B = ln(ln2/ln10) / ln(t50/t90) and
  T = t50 / (ln 2)**(1/B).
* LAM_F = 0.30 -- release fraction already spent when the face
  reaches the section: a DECLARED GENERATOR CONVENTION (placeholder
  convention lane, same status as fe_dev DOC272_CONV_PHI), not a
  field-calibrated value; no on-disk measurement of it exists.

Structural degeneracy note (freeze item "structural degeneracies")
------------------------------------------------------------------
The monitored increment is u_obs(t) = (lambda(t) - lambda0) * u_inf
with lambda0 = lambda(t_install).  For BETA = 1 (plain exponential)
the family is CLOSED under time shift + affine rescale:
lambda(t) - lambda0 = (1 - lambda0) * (1 - exp(-(t-t0)/T)), so
(1 - lambda0) merges exactly into the displacement scale and lambda0
is STRUCTURALLY non-identifiable from increments.  BETA != 1 breaks
the shift closure: the post-install shape depends on lambda0 through
x0 = (ln((1-LAM_F)/(1-lambda0)))**(1/BETA), so lambda0 is separated
by amplitude-shape coupling.  Whether that separation survives the
FROZEN noise is exactly what the stage-2 machine measures -- the
identifiability is NOT assumed.

Parameterization used everywhere: unknown lambda0 (release at
monitoring start), with t_p eliminated via lambda(t0) = lambda0:

    g(dt; lambda0) := lambda(t0 + dt) - lambda0
                    = (1 - lambda0)
                      * (1 - exp(x0**BETA - ((x0*T + dt)/T)**BETA)),
    x0 = (ln((1 - LAM_F) / (1 - lambda0)))**(1/BETA)   (>= 0
    requires lambda0 >= LAM_F, enforced by the frozen boxes).

g is torch-differentiable in lambda0 (through-system AD carries it).
"""

from __future__ import annotations

import numpy as np
import torch

# frozen constants (kickoff freeze section "release schedule")
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
