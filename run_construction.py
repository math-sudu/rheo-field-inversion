"""Evaluate dated construction from a caller-supplied JSON input file."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments.geometry_revision.baseline import bind_cad_source, load_modules
from experiments.geometry_revision.cad_geometry import build_layout, CHANNELS
from experiments.geometry_revision.construction import Construction, create_evaluator


def build_evaluator(data, input_directory):
    """Build the same dated model for forward runs and profile preparation."""
    bind_cad_source(Path(input_directory)/data["cad_source"])
    layout_args = dict(data["layout"])
    layout_args["translation"] = complex(*layout_args.pop("translation_m"))
    layout = build_layout(**layout_args)
    modules = load_modules()
    observed = data["observations"]
    stamps = modules.field.FieldStamps(data["section"],
        {c: np.array(observed["stamps"][c]) for c in CHANNELS},
        {c: np.array(observed["raw_y"][c]) for c in CHANNELS}, observed["stamp0"])
    cfg = Construction(**data["construction"])
    times = np.unique(np.r_[0., .05, cfg.delay_days, cfg.delay_days+cfg.lower_day,
        cfg.delay_days+cfg.lower_day+.05, cfg.delay_days+cfg.lower_day+.5,
        cfg.delay_days+cfg.closure_day, cfg.delay_days+cfg.casting_day,
        np.arange(cfg.delay_days+cfg.step_days, cfg.delay_days+stamps.span(), cfg.step_days),
        cfg.delay_days+stamps.span()+.1])
    ev = create_evaluator(layout, construction=cfg, family="rheo", **data["anchors"],
        parameter_source=data["parameter_source"], section=data["section"], stamps=stamps,
        sigma_section=data["sigma_mm"], time_interpolation="pchip",
        n_r=data["mesh"][0], n_t=data["mesh"][1],
        axis=(np.r_[0., np.ones(len(times)-1)], times), max_iter=80, min_substep=1/64)
    return ev


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jacobian", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    data = json.loads(args.input.read_text(encoding="utf-8"))
    ev = build_evaluator(data, args.input.parent)
    values, history = ev.model(data["theta"])
    residual = (values-ev.raw_y)/ev.sigma
    result = dict(predictions_mm=values.tolist(), weighted_SSE=float(residual@residual),
                  channel_order=CHANNELS, all_increments_converged=all(s.trial.converged for s in history.executed))
    if args.jacobian:
        names = ("s_star", "K0", "r_K", "tau_K", "tau_M", "r_s", "tau_vp")
        replay, jac = ev.jacobian(history, data["theta"], names)
        result.update(derivative_coordinates=names, physical_jacobian=jac.tolist(),
                      replay_error_mm=float(np.max(np.abs(replay-values))))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(f"Weighted SSE: {result['weighted_SSE']:.8f}; output: {args.output}")


if __name__ == "__main__":
    main()
