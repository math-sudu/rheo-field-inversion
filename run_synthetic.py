"""Generate the noise-free synthetic observations from the supplied inputs."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments.geometry_revision.baseline import bind_cad_source, load_modules
from experiments.geometry_revision.cad_geometry import build_layout, OUTER
from experiments.geometry_revision.field_evaluator import CadFieldEvaluator
from experiments.geometry_revision.rheo_replay import ConsistentReplayEvaluator

ROOT = Path(__file__).resolve().parent


def evaluator():
    config = json.loads((ROOT/"data/synthetic.json").read_text(encoding="utf-8"))
    bind_cad_source(ROOT/"data/geometry/tangent_arcs.json")
    layout = build_layout(datum_role=OUTER, measured_surface=OUTER, scale=1., n_map=32,
        size_source="original-size synthetic reference", observation_source="outer contour synthetic targets",
        engineering_status="synthetic reference")
    stamps = {c: np.asarray(t) for c, t in config["timestamps_days"].items()}
    obs = load_modules().field.FieldStamps("synthetic", stamps,
                {c: np.zeros(len(t)) for c, t in stamps.items()}, 0.)
    base = CadFieldEvaluator(layout, family="rheo", parameter_source="synthetic generator",
        section="synthetic", time_interpolation="pchip", stamps=obs,
        axis=(config["lambdas"], config["times_days"]), sigma_section=config["channel_sigma_mm"],
        **config["FE"], **config["anchors"])
    return ConsistentReplayEvaluator(base), config


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, default=ROOT/"output/synthetic.json")
    parser.add_argument("--jacobian", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    ev, config = evaluator()
    values, history = ev.model(config["generating_theta"])
    result = {"y_mm": values.tolist(), "sigma_mm": ev.sigma.tolist()}
    if args.jacobian:
        names = list(config["box"])
        replay, jac = ev.jacobian(history, config["generating_theta"], names)
        result.update(names=names, physical_jacobian=jac.tolist(),
                      replay_error_mm=float(np.max(abs(replay-values))))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(f"Generated {len(values)} noise-free observations: {args.output}")


if __name__ == "__main__":
    main()
