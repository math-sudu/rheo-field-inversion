"""Profile unknown chart offsets while retaining every raw measurement.

The first digitized value is not evidence of an installation zero. One
constant nuisance offset per channel is eliminated analytically from the
weighted least-squares objective. The mechanical time stamps are unchanged.
"""
from __future__ import annotations

import numpy as np


class ProfiledDatumEvaluator:
    observation_version = "raw-record-profiled-channel-datum-v1"

    def __init__(self, evaluator, channels):
        self.mechanical_evaluator = evaluator
        self.channels = tuple(channels)
        self.channel_slices = {}
        offset = 0
        for channel in self.channels:
            n = len(evaluator.stamps.stamps[channel])
            self.channel_slices[channel] = slice(offset, offset + n)
            offset += n
        self.raw_y = np.concatenate([evaluator.stamps.y_mm[c] for c in self.channels]).copy()
        self.raw_y.setflags(write=False)
        self.n_profiled_offsets = len(self.channels)

    def __getattr__(self, name):
        return getattr(self.mechanical_evaluator, name)

    def apply_offsets(self, mechanical_values, jacobian=None):
        prediction = np.asarray(mechanical_values, dtype=float).copy()
        jac = None if jacobian is None else np.asarray(jacobian, dtype=float).copy()
        offsets = {}
        for channel, selection in self.channel_slices.items():
            weight = 1 / np.square(self.sigma[selection])
            weight = weight / weight.sum()
            b = float(weight @ (self.raw_y[selection] - prediction[selection]))
            offsets[channel] = b
            prediction[selection] += b
            if jac is not None:
                jac[selection] -= weight @ jac[selection]
        return prediction, jac, offsets

    def model(self, theta):
        values, history = self.mechanical_evaluator.model(theta)
        prediction, _, _ = self.apply_offsets(values)
        return prediction, history

    def jacobian(self, history, theta, names):
        values, jac = self.mechanical_evaluator.jacobian(history, theta, names)
        prediction, projected, _ = self.apply_offsets(values, jac)
        return prediction, projected

    def fitted_offsets(self, history, theta):
        values = self.mechanical_evaluator.model_np(history, float(theta["lam0"]))
        return self.apply_offsets(values)[2]

