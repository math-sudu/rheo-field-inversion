"""Portable file adapter for the manuscript's construction FE-PINC workflow."""
from pathlib import Path
import json

import torch

from run_construction import build_evaluator
from experiments.geometry_revision.trf_solver import solve_trf
from experiments.geometry_revision.field_response import response_metrics

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "data/field"
OUTPUT = ROOT / "output/field"
VARIANTS = {"primary": {}, "soft_support": {"support_stiffness_MPa_per_m": 1.91},
            "stiff_support": {"support_stiffness_MPa_per_m": 7.64},
            "no_casting": {"casting": False}, "half_step": {"step_days": 1.0}}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def initial(branch):
    return dict(read(SOURCE/"config.json")["initializations"][branch])


def setup(variant="primary", section="DK0+935", mesh=(8, 32)):
    settings = read(SOURCE/"config.json")
    data = read(SOURCE/f"{section[-3:]}.json")
    data["construction"].update(VARIANTS[variant])
    data["mesh"] = mesh
    ev = build_evaluator(data, SOURCE)
    box = settings["box"]
    phase = data["theta"]["lam0"]
    box["lam0"] = [min(box["lam0"][0], phase), max(box["lam0"][1], phase)]
    return ev, box, {"lam0": phase}, settings["solver"]


def fit(variant="primary", branch=0, tau=None, start=None, directory=None, mesh=(8, 32)):
    """Fresh-FE correction using the same bounded solver and endpoint selection."""
    torch.set_num_threads(1)
    directory = directory or OUTPUT/variant/f"branch{branch}"
    path = directory/"result.json"
    if path.exists():
        return read(path)
    ev, box, fixed, settings = setup(variant, mesh=mesh)
    if tau is not None:
        fixed["tau_M"] = float(tau)
    seed = {**(start or initial(branch)), **fixed}
    from fedev_src.solver import IncrementFailure
    from fedev_src.coupling import CouplingFailure
    solved = solve_trf(ev, ev.raw_y, box, [n for n in box if n not in fixed], fixed, seed,
                      failure_type=(IncrementFailure, CouplingFailure), **settings)
    candidates = [c for c in solved["candidates"] if c.get("terminal")]
    if not candidates:
        write(directory/"failure.json", solved)
        raise RuntimeError("Construction solve has no finite verified endpoint")
    selected = min(candidates, key=lambda c: c["terminal"]["exact_objective"])
    theta = selected["terminal"]["terminal_theta"]
    values, history = ev.model(theta)
    result = dict(variant=variant, branch=branch, theta=theta, box=box,
                  response=response_metrics(ev, values, ev.fitted_offsets(history, theta)),
                  solve=solved)
    write(path, result)
    print({"finished": str(path), "SSE": result["response"]["weighted_SSE"]}, flush=True)
    return result
