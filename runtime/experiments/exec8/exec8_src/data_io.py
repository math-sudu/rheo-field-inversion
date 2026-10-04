
from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np


DATA_DIR = Path(__file__).resolve().parents[3] / "results" / "exec8" / "data"

PINNED_SHA256 = {
    "mx4_dk0+935_cumulative_convergence_deformation_vs_time.csv":
        "43dd65b56068092bf0719e08e3d7a5ad37549a28c5f4bee557d179419e1e0dba",
    "mx4_dk0+915_cumulative_convergence_deformation_vs_time.csv":
        "579dd419f1b6db5f39a9bd47575c283a11f76acf7e626d323b0df35dcbd7a4c2",
    "moxi_dk0p665_monitoring_deformation_vs_time.csv":
        "df898320b5e86bdb8c7992b4b13dd00994ffd6cb5f38d9933910b652bba619b4",
}

SECTION_FILES = {
    "DK0+935": "mx4_dk0+935_cumulative_convergence_deformation_vs_time.csv",
    "DK0+915": "mx4_dk0+915_cumulative_convergence_deformation_vs_time.csv",
    "DK0+665": "moxi_dk0p665_monitoring_deformation_vs_time.csv",
}



# 6.0 / 9.5 m below the crown, top to bottom SL01-SL02 / SL03-SL04 /
# SL05-SL06)
CHORD_CHANNELS = ("SL01-SL02", "SL03-SL04", "SL05-SL06")
CROWN_CHANNEL = "GD"


@dataclass(frozen=True)
class Series:
    """One digitized monitoring series, time-sorted."""

    section: str
    channel: str          # e.g. "SL03-SL04", "GD", "data1"
    t_day: np.ndarray     # sorted, campaign/day-of-year frame of the source
    y_mm: np.ndarray


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def verify_pins(data_dir: Path = DATA_DIR) -> None:
    """Raise if any pinned snapshot is missing or drifted."""
    for name, want in PINNED_SHA256.items():
        p = data_dir / name
        if not p.exists():
            raise FileNotFoundError(f"pinned snapshot missing: {p}")
        got = _sha256(p)
        if got != want:
            raise ValueError(
                f"pinned snapshot drifted: {name} sha256 {got} != {want}")


def load_section(section: str, data_dir: Path = DATA_DIR) -> list[Series]:
    """Load all series of one section, each sorted by time."""
    path = data_dir / SECTION_FILES[section]
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    tcol = ("day_of_year_day" if "day_of_year_day" in rows[0]
            else "time_day")
    ycol = ("cumulative_deformation_mm" if "cumulative_deformation_mm"
            in rows[0] else "deformation_mm")
    out: list[Series] = []
    for name in sorted({r["series"] for r in rows}):
        sel = [r for r in rows if r["series"] == name]
        t = np.array([float(r[tcol]) for r in sel])
        y = np.array([float(r[ycol]) for r in sel])
        order = np.argsort(t, kind="stable")
        channel = name.replace(section, "")  # "DK0+935SL03-SL04" -> chord id
        out.append(Series(section=section, channel=channel,
                          t_day=t[order], y_mm=y[order]))
    return out


def load_all(data_dir: Path = DATA_DIR) -> dict[str, list[Series]]:
    verify_pins(data_dir)
    return {s: load_section(s, data_dir) for s in SECTION_FILES}
