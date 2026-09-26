"""Load only the numerical modules shipped with the synthetic example."""
from functools import lru_cache
import hashlib
import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


@lru_cache(maxsize=1)
def load_modules():
    root = Path(__file__).resolve().parents[2] / "runtime/experiments"
    for name in ("exec19_rheo_inversion", "exec8", "exec9", "fe_dev"):
        sys.path.insert(0, str(root / name))
    modules = SimpleNamespace(snapshot={"snapshot_fingerprint": "supplied-synthetic-runtime"})
    for alias, name in {
        "geometry": "exec8_src.geometry", "coupled": "exec9_src.coupled",
        "field": "exec19_src.field19", "identify": "exec19_src.identify19",
        "cad": "exec8_src.cad_source", "invert": "exec19_src.invert19",
    }.items():
        setattr(modules, alias, importlib.import_module(name))
    return modules


def bind_cad_source(path):
    """Use an explicit numeric arc descriptor without requiring archived DXFs."""
    raw = Path(path).read_bytes()
    data = json.loads(raw)
    if data["confirmed_metres_per_drawing_unit"] != 1.0:
        raise ValueError("Convert the supplied CAD arc coordinates to metres")
    cad = load_modules().cad
    cad.SOURCE_VERSION = data["geometry_id"]
    cad.SOURCE_SHA256 = data["source_sha256"]
    cad.ORIGINAL_ARCS_SHA256 = data["source_arcs_sha256"]
    cad.ARCS_SHA256 = hashlib.sha256(raw).hexdigest()
    cad.TANGENT_DXF_SHA256 = None
    cad.load_source = lambda: data
