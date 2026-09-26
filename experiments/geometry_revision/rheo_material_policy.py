"""Versioned positive-viscosity return policy over the immutable CAD snapshot.

The sole constitutive change restricts value pinning to visc_H == 0. Both
primal integration and AD replay use the same function. Installation is scoped
to a calculation; historical evaluators and the 95 archived files stay intact.
"""
from contextlib import contextmanager
import ast
import hashlib
import inspect
from pathlib import Path

from .baseline import load_modules, digest


VERSION = "cad-continuous-perzyna-return-v1"
FROZEN_SHA256 = "df20d41410ce6a244370ceb7abfe6ca0707fd3b1619e94a44263e375fb77c1a4"


def build_return():
    load_modules()
    from fedev_src import material

    path = Path(material.__file__)
    if hashlib.sha256(path.read_bytes()).hexdigest() != FROZEN_SHA256:
        raise ValueError("The material policy requires its exact archived source")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    function, = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_mc_sorted"]
    assignments = [n for n in ast.walk(function) if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == "admissible" for t in n.targets)]
    assignment, = assignments
    original = ast.parse("plastic & (f13 <= ftol)", mode="eval").body
    if ast.dump(assignment.value) != ast.dump(original):
        raise ValueError("The reviewed material-mask expression changed")
    assignment.value = ast.parse("plastic & (f13 <= ftol) & (visc_H == 0)", mode="eval").body
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    namespace = dict(vars(material))
    exec(compile(module, str(Path(__file__).resolve()), "exec"), namespace)
    adopted = namespace["_mc_sorted"]
    identity = {"version": VERSION, "historical_material_sha256": FROZEN_SHA256,
                "policy_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "adopted_function_ast_sha256": hashlib.sha256(ast.dump(module).encode()).hexdigest(),
                "change": "admissible = plastic & (f13 <= ftol) & (visc_H == 0)"}
    return adopted, identity


@contextmanager
def active():
    """Install both imported bindings and reject unversioned nested mutation."""
    function, identity = build_return()
    from fedev_src import material, material_visc

    original = material._mc_sorted
    if (material_visc._mc_sorted is not original
            or Path(inspect.getsourcefile(original)).resolve() != Path(material.__file__).resolve()):
        raise ValueError("Material bindings are already changed; policy scopes cannot nest")
    material._mc_sorted = material_visc._mc_sorted = function
    try:
        yield identity
        if material._mc_sorted is not function or material_visc._mc_sorted is not function:
            raise ValueError("Material policy changed during calculation")
    finally:
        material._mc_sorted = material_visc._mc_sorted = original


class PolicyEvaluator:
    """Explicit calculation identity; scope the adopted policy to FE and J."""

    def __init__(self, evaluator):
        if evaluator.family != "rheo":
            raise ValueError("This adoption is restricted to the RHEO family")
        self.base = evaluator
        _, self.policy = build_return()
        self.identity = {"parent_evaluator": evaluator.identity, "material_policy": self.policy}
        self.policy_fingerprint = digest(self.identity)

    def __getattr__(self, name):
        return getattr(self.base, name)

    def forward(self, theta):
        with active() as identity:
            if identity != self.policy:
                raise ValueError("Material source changed after evaluator construction")
            run = self.base.forward(theta)
        run.adopted_material_fingerprint = self.policy_fingerprint
        return run

    def model(self, theta):
        run = self.forward(theta)
        return self.base.model_np(run, theta["lam0"]), run

    def jacobian(self, run, theta, names):
        if getattr(run, "adopted_material_fingerprint", None) != self.policy_fingerprint:
            raise ValueError("A historical FE state cannot enter an adopted-policy J")
        with active() as identity:
            if identity != self.policy:
                raise ValueError("Material source changed after evaluator construction")
            return self.base.jacobian(run, theta, names)
