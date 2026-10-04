from contextlib import contextmanager
import hashlib
from pathlib import Path
from types import FunctionType

from .baseline import digest, load_modules
from .rheo_material_policy import PolicyEvaluator


VERSION = "cad-committed-state-implicit-replay-v1"
FROZEN_SHA256 = "b034c43858ef3f9ccea03f2cf4ec3ac8095352eed7915dfb00e816525619f75b"


def build_replay():
    load_modules()
    from exec19_src import replay19

    if hashlib.sha256(Path(replay19.__file__).read_bytes()).hexdigest() != FROZEN_SHA256:
        raise ValueError("Committed-state replay requires the supplied source version")
    original = replay19.replay_visc
    if original.__globals__ is not vars(replay19):
        raise ValueError("Replay scopes cannot nest over a changed binding")
    solve = replay19.lu_solve_t

    def derivative_solve(lu, residual):
        return solve(lu, residual - residual.detach())

    namespace = dict(vars(replay19), lu_solve_t=derivative_solve)
    replay = FunctionType(original.__code__, namespace, original.__name__, original.__defaults__)
    identity = {"version": VERSION, "historical_replay_sha256": FROZEN_SHA256,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "linear_rhs": "R_aug - stop_gradient(R_aug)",
        "derivative": "first-order implicit derivative at committed FE state and final consistent LU"}
    return replay, identity


@contextmanager
def active():
    """Scope the adapter to callers that import the RHEO replay at evaluation."""
    replay, identity = build_replay()
    from exec19_src import replay19

    original = replay19.replay_visc
    replay19.replay_visc = replay
    try:
        yield identity
    finally:
        replay19.replay_visc = original


class ConsistentReplayEvaluator(PolicyEvaluator):

    def __init__(self, evaluator):
        super().__init__(evaluator)
        _, self.replay_policy = build_replay()
        self.identity = {**self.identity, "replay_policy": self.replay_policy}
        self.policy_fingerprint = digest(self.identity)

    def jacobian(self, run, theta, names):
        if not getattr(run, "executed", None) or any(
                not step.trial.converged or step.trial.lu is None for step in run.executed):
            raise ValueError("Implicit replay requires converged increments and their final LU")
        with active() as identity:
            if identity != self.replay_policy:
                raise ValueError("Replay source changed after evaluator construction")
            return super().jacobian(run, theta, names)
