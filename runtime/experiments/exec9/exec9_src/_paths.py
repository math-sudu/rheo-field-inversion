"""sys.path bootstrap for the read-only sibling trees.

exec8_src (inversion machine) lives under experiments/exec8;
fedev_src (coupled FE machine) under experiments/fe_dev.  exec8's
geometry module inserts the exec6b tree itself.  Import this module
before any `exec8_src` / `fedev_src` import.
"""

import sys
from pathlib import Path

EXPERIMENTS = Path(__file__).resolve().parents[2]
RESULTS = EXPERIMENTS.parent / "results"

for _sub in ("exec8", "fe_dev"):
    _p = str(EXPERIMENTS / _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)
