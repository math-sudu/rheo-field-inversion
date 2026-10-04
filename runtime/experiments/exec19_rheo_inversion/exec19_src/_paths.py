"""Relative imports for the supplied numerical runtime."""
from pathlib import Path
import sys
EXPERIMENTS = Path(__file__).resolve().parents[2]
REPO = EXPERIMENTS.parent
RESULTS = REPO / "results"
OUT_DIR = RESULTS
for name in ("exec8", "exec9", "fe_dev"):
    sys.path.insert(0, str(EXPERIMENTS / name))
