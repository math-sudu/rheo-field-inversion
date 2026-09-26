"""Carrier 2: potential coefficient network (trainable coefficient table).

Skeleton-chapter form "coefficient network with constant input slot"
reduced to its minimal prototype: the far-zone Laurent coefficients are
a trainable real parameter table (4N,) optimized by gradient descent
(Adam) on the interface boundary-data misfit

    L(q) = || D_s q - data ||^2 / (2M)

with the SAME scaled design matrix D_s as the least-squares carrier, so
any behavioral difference vs carrier 1 isolates "network optimization
error" from "Schwarz iteration convergence".

Training protocol (documented choice): the table is retrained at every
Schwarz iteration on the current interface data, WARM-STARTED from the
previous iteration's table (first fit: cold start from zeros, budget
``steps_first``; subsequent fits: ``steps_warm``).  Adam moments are
re-initialized at each fit; the learning rate follows a cosine decay to
1e-3 of its initial value, which anneals the iterate into the quadratic
minimum.  Per-fit achieved misfits are recorded in ``fit_misfits``.
"""

import numpy as np

from .farfield import FarBase


class FarCoeffNet(FarBase):
    name = "coeff_net"

    def __init__(self, R, theta_nodes, prm, n_coeff=8,
                 lr=0.2, steps_first=4000, steps_warm=1200):
        super().__init__(R, theta_nodes, prm, n_coeff=n_coeff)
        self.lr = float(lr)
        self.steps_first = int(steps_first)
        self.steps_warm = int(steps_warm)
        self.q = None                 # scaled table, warm-started across fits
        self.fit_misfits = []
        self.fit_steps = []

    def _solve(self, Ds, data, key):
        warm = self.q is not None
        q = self.q.copy() if warm else np.zeros(Ds.shape[1])
        steps = self.steps_warm if warm else self.steps_first
        M = Ds.shape[0]
        H = Ds.T @ Ds                # (4N, 4N) small; exact gradient uses it
        b = Ds.T @ data
        beta1, beta2, eps = 0.9, 0.999, 1e-12
        m = np.zeros_like(q)
        v = np.zeros_like(q)
        best_q = q.copy()
        best_loss = np.inf
        for t in range(1, steps + 1):
            g = (H @ q - b) / M
            m = beta1 * m + (1.0 - beta1) * g
            v = beta2 * v + (1.0 - beta2) * g * g
            mhat = m / (1.0 - beta1**t)
            vhat = v / (1.0 - beta2**t)
            lr_t = self.lr * (1e-3 + 0.999 * 0.5 *
                              (1.0 + np.cos(np.pi * (t - 1) / steps)))
            q -= lr_t * mhat / (np.sqrt(vhat) + eps)
            if t % 50 == 0 or t == steps:
                r = Ds @ q - data
                loss = 0.5 * float(r @ r) / M
                if loss < best_loss:
                    best_loss = loss
                    best_q = q.copy()
        self.q = best_q
        r = Ds @ best_q - data
        self.fit_misfits.append(
            np.linalg.norm(r) / max(np.linalg.norm(data), 1e-300))
        self.fit_steps.append(steps)
        return best_q
