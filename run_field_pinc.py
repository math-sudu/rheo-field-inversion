"""FE-PINC continuation regenerated for the staged field mechanics."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import itertools
import multiprocessing
from pathlib import Path
import time

import numpy as np
import torch

from experiments.fe_pinc.core import CoordinateMap
from field_workflow import OUTPUT, SOURCE, VARIANTS, read, write, setup, fit, initial
from experiments.geometry_revision.field_pinc_network import train

NAMES = ("s_star", "K0", "r_K", "tau_K", "r_s", "tau_vp")
NODES = (50., 100., 250., 500., 1000., 2000.)
QUERIES = (75., 375., 750., 1700.)
SEEDS = (11, 29, 47)
PINC = OUTPUT/"pinc"


def pinc_directory(variant):
    return PINC if variant == "primary" else OUTPUT/variant/"pinc"


def profile_track(track, variant="primary"):
    """Continue each old Kelvin initialization at fixed Maxwell times."""
    previous = None
    for tau in reversed(NODES):
        start = initial(track) if previous is None else previous["theta"]
        previous = fit(variant, branch=track, tau=tau, start=start,
            directory=pinc_directory(variant)/"profiles"/f"track{track}"/f"q{tau:g}")
    return track


def prepare(variant="primary"):
    torch.set_num_threads(1)
    directory = pinc_directory(variant)
    path = directory/"inputs.json"
    if path.exists():
        return read(path)
    ev, box, _, _ = setup(variant)
    mapping = CoordinateMap(box)
    arrays = {k: [] for k in ("value_x", "value_branch", "value_target", "value_weight",
        "physics_x", "physics_branch", "physics_target", "corrected_target", "gradient", "hessian")}
    centers = []
    for track, tau in itertools.product((0, 1), NODES):
        source = directory/"profiles"/f"track{track}"/f"q{tau:g}"/"result.json"
        row = read(source)
        theta = row["theta"]
        pred, history = ev.model(theta)
        replay, jac = ev.jacobian(history, theta, (*NAMES, "tau_M"))
        np.testing.assert_allclose(replay, pred, atol=1e-8, rtol=0.)
        residual = (pred-ev.raw_y)/ev.sigma
        scales = np.array([mapping.dphysical_dunit(n, theta[n]) for n in (*NAMES, "tau_M")])
        J = jac/ev.sigma[:, None]*scales[None, :]
        gradient, hessian = J[:, :-1].T@residual, J[:, :-1].T@J[:, :-1]
        unit = np.array([mapping.to_unit(n, theta[n]) for n in NAMES])
        x = 2*mapping.to_unit("tau_M", tau)-1
        for k, value in dict(value_x=x, value_branch=track, value_target=unit.tolist(), value_weight=1.,
            physics_x=x, physics_branch=track, physics_target=unit.tolist(), corrected_target=unit.tolist(),
            gradient=gradient.tolist(), hessian=hessian.tolist()).items():
            arrays[k].append(value)
        centers.append(dict(track=track, tau_M=tau, theta=theta, SSE=float(residual@residual),
            predictions_mm=pred.tolist(), physical_jacobian=jac.tolist(),
            cross=(J[:, :-1].T@J[:, -1]).tolist(), source=str(source.relative_to(OUTPUT)),
            replay_error_mm=float(np.max(np.abs(replay-pred)))))
        print({"prepared": [track, tau]}, flush=True)
    arrays.update(tangent_x=[], tangent_branch=[], tangent=[])
    result = dict(arrays=arrays, centers=centers, nuisance_order=NAMES, box=box,
        evaluator=ev.identity, query_times_days=QUERIES, seeds=SEEDS,
        physics_meaning="Selected-branch residual linearizations at corrected FE anchors. Corrected targets equal the centers; no smooth-profile tangent is asserted.")
    write(path, result)
    return result


def training(variant="primary"):
    data = prepare(variant)
    settings = read(SOURCE/"config.json")["training"]
    arrays = {k: np.asarray(v, dtype=int if k.endswith("branch") else float) for k, v in data["arrays"].items()}
    arrays["tangent"] = arrays["tangent"].reshape(-1, len(NAMES))
    mapping = CoordinateMap(data["box"])
    for seed in SEEDS:
        directory = pinc_directory(variant)/"models"/str(seed)
        if (directory/"training.json").exists():
            continue
        began = time.perf_counter()
        model, history, final = train(arrays, settings, seed)
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), directory/"model.pt")
        proposals, dense = [], []
        model.eval()
        for track in (0, 1):
            for tau in QUERIES:
                x = torch.tensor([[2*mapping.to_unit("tau_M", tau)-1]], dtype=torch.float32)
                with torch.no_grad():
                    unit = model(x, torch.tensor([track], dtype=torch.long)).numpy()[0]
                theta = {n: mapping.from_unit(n, float(u), clip=False) for n, u in zip(NAMES, unit)}
                theta["tau_M"] = tau
                proposals.append(dict(seed=seed, track=track, tau_M=tau, theta=theta))
            x = np.linspace(-1, 1, 101)
            with torch.no_grad():
                unit = model(torch.tensor(x[:, None], dtype=torch.float32),
                    torch.full((len(x),), track, dtype=torch.long)).numpy()
            dense.append(dict(track=track, x=x.tolist(), unit=unit.tolist()))
        write(directory/"training.json", dict(seed=seed, settings=settings, history=history,
            final_loss=final, proposals=proposals, dense_paths=dense, wall_seconds=time.perf_counter()-began))
        print({"trained_seed": seed, "loss": final}, flush=True)


def query(task, variant="primary"):
    seed, track, tau = task
    root = pinc_directory(variant)
    row = read(root/"models"/str(seed)/"training.json")
    proposal = next(p for p in row["proposals"] if p["track"] == track and p["tau_M"] == tau)
    directory = root/"queries"/f"seed{seed}_track{track}_q{tau:g}"
    if not (directory/"proposal.json").exists():
        torch.set_num_threads(1)
        ev, _, fixed, _ = setup(variant)
        theta = {**proposal["theta"], **fixed}
        pred, _ = ev.model(theta)
        write(directory/"proposal.json", dict(**proposal, predictions_mm=pred.tolist(),
            weighted_SSE=float(np.sum(((pred-ev.raw_y)/ev.sigma)**2))))
    return fit(variant, branch=track, tau=tau, start=proposal["theta"], directory=directory)["response"]["weighted_SSE"]


def pool(function, tasks, workers):
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as executor:
        for result in executor.map(function, tasks):
            print({"completed": result}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("profiles", "train", "queries", "all"))
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--variant", choices=VARIANTS, default="primary")
    args = parser.parse_args()
    if args.stage in ("profiles", "all"):
        pool(partial(profile_track, variant=args.variant), (0, 1), args.workers)
    if args.stage in ("train", "all"):
        training(args.variant)
    if args.stage in ("queries", "all"):
        pool(partial(query, variant=args.variant), itertools.product(SEEDS, (0, 1), QUERIES), args.workers)
