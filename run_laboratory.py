"""Regenerate specimen fits and the 25-node Maxwell-viscosity profiles."""
import argparse
import csv
from pathlib import Path

import numpy as np

from experiments.geometry_revision.baseline import load_modules

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, default=ROOT/"output/laboratory")
    args = parser.parse_args()
    load_modules()
    import fit_creep_doc272 as fitting
    from exec19_src import lab19
    data = ROOT/"data/laboratory"
    fitting.DATA_DIR = data
    fits = fitting.main(["--data", str(data), "--out", str(args.output)])
    lab19.PARAMS_CSV = args.output/"nishihara_params.csv"
    with (args.output/"laboratory_profile.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["group", "beta_deg", "eta_M_GPa_h", "SSE_pct2", "D_lab"])
        for fit in fits:
            group = lab19.load_group(fit.spec.key)
            grid, sse = lab19.profile_eta_M(group, seed=20260712)
            scaled = (sse-np.min(sse))/group.sigma_lab**2
            for eta, value, d in zip(grid, sse, scaled):
                writer.writerow([group.key, fit.spec.beta_deg, eta/1000., value, d])
            print({"group": group.key, "observations": len(group.t_abs), "profile_nodes": len(grid)}, flush=True)


if __name__ == "__main__":
    main()
