"""Read-only import bridge to the frozen linear-elastic prototype.

Loads ``experiments/coupling_schwarz_proto/proto_src`` under the module
name ``proto_src`` (via importlib, no sys.path pollution beyond the
name itself) so its modules can be used for equivalence tests and
reference data alongside this experiment's own package ``fedev_src``
(disk names and module names coincide since the D-052-1 namespace
ADR renames).

The prototype is READ-ONLY: nothing here writes to it; its own
``kirsch_ref`` module self-manages the path to the trusted Kirsch
reference solver in ``experiments/seed_skeleton_km``.
"""

import importlib
import importlib.util
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXPERIMENTS = os.path.normpath(os.path.join(_HERE, os.pardir, os.pardir))
PROTO_ROOT = os.path.join(_EXPERIMENTS, "coupling_schwarz_proto")
_PROTO_SRC = os.path.join(PROTO_ROOT, "proto_src")


def _load_package():
    if "proto_src" in sys.modules:
        return sys.modules["proto_src"]
    spec = importlib.util.spec_from_file_location(
        "proto_src",
        os.path.join(_PROTO_SRC, "__init__.py"),
        submodule_search_locations=[_PROTO_SRC],
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["proto_src"] = mod
    spec.loader.exec_module(mod)
    return mod


_load_package()

kirsch_ref = importlib.import_module("proto_src.kirsch_ref")
fe_annulus = importlib.import_module("proto_src.fe_annulus")
dtn = importlib.import_module("proto_src.dtn")
farfield = importlib.import_module("proto_src.farfield")
schwarz = importlib.import_module("proto_src.schwarz")
coeff_net = importlib.import_module("proto_src.coeff_net")
