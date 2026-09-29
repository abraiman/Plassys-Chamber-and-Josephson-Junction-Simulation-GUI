"""
resistance_interpretation.py -- pure-Python (numpy/scipy only, no Qt)
analysis core for the "Interpret Results" tab: room-temperature
junction resistance vs. inverse area (Ambegaokar-Baratoff validation).

This is a direct, careful port of the verified filtering/regression
method from earlier standalone analysis scripts used for this
measurement workflow -- same open/short classification, same
row-wise Z-score outlier rejection, same linregress + J_c formula --
generalized so the GUI can drive it with live data (the Measure
Resistance tab's internal table, or an imported CSV) and live,
per-row junction geometry (instead of the scripts' own hardcoded
per-chip configuration dict) rather than a fixed set of named
datasets.

Kept dependency-free of PySide6/Qt on purpose so this can be unit
tested headlessly with zero GUI setup, exactly like scan_pattern.py
and resistance_parser.py already are.
"""

from __future__ import annotations

import csv
import datetime
import json
import re
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np
from scipy.stats import linregress
from scipy.optimize import minimize_scalar

# --- superconducting constants (aluminum) -----------------------------------
K_B_EV = 8.617333262145e-5  # Boltzmann constant, eV/K


def ambegaokar_baratoff_ic_rn_volts(T_c_kelvin: float) -> float:
    """I_c * R_n product (volts) from the Ambegaokar-Baratoff relation,
    at T=0, for a superconductor with critical temperature T_c_kelvin.
    Same formula as the reference scripts this module was ported from."""
    delta_0_eV = 1.764 * K_B_EV * T_c_kelvin
    return (np.pi * delta_0_eV) / 2.0


# --- cell parsing -------------------------------------------------------------

_NUMBER_RE = re.compile(r"[-+]?\d[\d,]*\.?\d*")


def parse_cell(raw):
    """Parse one raw measurement-table cell into a float (kOhm), the
    sentinel string 'OPEN', the sentinel string 'SHORT', or None if it
    can't be interpreted at all. Ported logic from the original
    scripts' own parse_cell, so behavior (including hyphenated-range
    averaging, e.g. '212.1-212.4' -> 212.25) matches exactly."""
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    s = str(raw).strip().lower()
    if s == "":
        return None
    if s in ("open", "o", "open circuit"):
        return "OPEN"
    if s in ("short", "s", "shorted"):
        return "SHORT"
    if "-" in s:
        parts = s.split("-")
        if len(parts) == 2:
            try:
                lo, hi = float(parts[0]), float(parts[1])
                return (lo + hi) / 2.0
            except ValueError:
                pass
    try:
        return float(s)
    except ValueError:
        cleaned = re.sub(r"\.\.+", ".", s)
        try:
            return float(cleaned)
        except ValueError:
            return None


# --- Step 1+2: open/short filtering, then per-row Z-score outlier filtering --

def filter_and_regress(raw_rows, r_base_kohm: float, short_threshold_kohm: float,
                        open_threshold_kohm: float, z_threshold: float) -> dict:
    """Runs the two-stage filter (open/short classification, then
    row-wise mean+Z-score outlier rejection) on a 2D list of raw cell
    values (numbers, numeric strings, 'open'/'short' text, hyphenated
    ranges, or blanks). Returns clean_rows (one list of net-of-baseline
    kOhm values per row, survivors only) plus full counts and an
    audit log of exactly what got dropped and why -- same shape/
    semantics as the reference scripts' own Step 1-2, just
    parameterized instead of read from module-level globals."""
    dropped_log = []
    open_count = 0
    shorted_count = 0
    initial_parsed_rows = []

    for r_idx, row in enumerate(raw_rows, start=1):
        clean_items = []
        for c_idx, raw_val in enumerate(row, start=1):
            coord_label = f"C{c_idx}R{r_idx}"
            parsed = parse_cell(raw_val)

            if parsed is None or parsed == "OPEN" or parsed == 0 or (
                    isinstance(parsed, float) and parsed >= open_threshold_kohm):
                open_count += 1
                dropped_log.append((coord_label, r_idx, c_idx, raw_val,
                                     f"Open Circuit (>= {open_threshold_kohm:.0f} kΩ Overload)"))
            elif parsed == "SHORT" or (isinstance(parsed, float) and abs(parsed - r_base_kohm) <= short_threshold_kohm):
                shorted_count += 1
                dropped_log.append((coord_label, r_idx, c_idx, raw_val, "Shorted Junction (Probe Baseline)"))
            else:
                net_val = parsed - r_base_kohm
                clean_items.append({"col": c_idx, "raw": raw_val, "net": net_val})

        initial_parsed_rows.append(clean_items)

    clean_rows = []
    outlier_count = 0

    for r_idx, row_items in enumerate(initial_parsed_rows, start=1):
        if len(row_items) <= 2:
            clean_rows.append([it["net"] for it in row_items])
            continue

        net_vals = np.array([it["net"] for it in row_items])
        row_mean = np.mean(net_vals)
        row_std = np.std(net_vals, ddof=1)

        if row_std == 0:
            clean_rows.append([it["net"] for it in row_items])
            continue

        kept_net = []
        for item in row_items:
            coord_label = f"C{item['col']}R{r_idx}"
            z = abs(item["net"] - row_mean) / row_std
            if z <= z_threshold:
                kept_net.append(item["net"])
            else:
                outlier_count += 1
                raw_mean_display = row_mean + r_base_kohm
                dropped_log.append((
                    coord_label, r_idx, item["col"], item["raw"],
                    f"Statistical Outlier (Z={z:.2f} > {z_threshold:.1f}, Row Mean ~{raw_mean_display:.2f} kΩ)"
                ))
        clean_rows.append(kept_net)

    dropped_log.sort(key=lambda x: (x[1], x[2]))

    return {
        "dropped_log": dropped_log,
        "open_count": open_count,
        "shorted_count": shorted_count,
        "outlier_count": outlier_count,
        "clean_rows": clean_rows,
    }


# --- geometry: per-row nominal area, with optional per-row offset/override --

def compute_row_areas(lw1_list, lw2_list, offset_um_list, area_override_list=None) -> np.ndarray:
    """Per-row junction area in um^2. For row i: if area_override_list[i]
    is set (not None/''), that number is used directly (the "I already
    know the real area, e.g. from an SEM polygon measurement" escape
    hatch). Otherwise area = (lw1_list[i] + offset_um_list[i]) *
    (lw2_list[i] + offset_um_list[i]) -- the same "apply one SEM/
    experimental offset to both nominal linewidths" model the reference
    scripts use, just per-row instead of one offset for the whole chip."""
    n = len(lw1_list)
    areas = np.zeros(n, dtype=float)
    for i in range(n):
        override = area_override_list[i] if area_override_list else None
        if override not in (None, ""):
            areas[i] = float(override)
        else:
            areas[i] = (float(lw1_list[i]) + float(offset_um_list[i])) * (float(lw2_list[i]) + float(offset_um_list[i]))
    return areas


def regress(clean_rows, areas_um2: np.ndarray, T_c_kelvin: float = 1.2) -> Optional[dict]:
    """Row means/stds, inverse-area linear regression, and the derived
    critical current density J_c -- same math as the reference
    scripts' Step 4. Returns None if fewer than 2 rows have any
    surviving data (can't fit a line through 0 or 1 points)."""
    y_means = np.array([np.mean(r) if len(r) > 0 else np.nan for r in clean_rows])
    y_stds = np.array([np.std(r, ddof=1) if len(r) > 1 else np.nan for r in clean_rows])

    # A row with no (or zero/blank) nominal geometry has area 0 -> 1/area
    # is inf, and once more than one row collapses to the same inf,
    # linregress raises ("all x values are identical") instead of
    # returning a usable result -- exclude any row without a real,
    # finite, positive area up front rather than letting that happen.
    with np.errstate(divide="ignore", invalid="ignore"):
        inverse_area = 1.0 / areas_um2
    geometry_ok = np.isfinite(inverse_area) & (areas_um2 > 0)
    valid_mask = ~np.isnan(y_means) & geometry_ok
    if valid_mask.sum() < 2:
        return None

    x_reg = inverse_area[valid_mask]
    y_reg = y_means[valid_mask]

    reg = linregress(x_reg, y_reg)
    a1, a0, r_value = reg.slope, reg.intercept, reg.rvalue
    r2 = r_value ** 2

    n = len(x_reg)
    if n > 2:
        y_hat = a0 + a1 * x_reg
        sr = np.sum((y_reg - y_hat) ** 2)
        standard_error = np.sqrt(sr / (n - 2))
    else:
        standard_error = float("nan")

    ic_rn_volts = ambegaokar_baratoff_ic_rn_volts(T_c_kelvin)
    slope_ohm_m2 = a1 * 1e-9
    if slope_ohm_m2 != 0:
        j_c_a_m2 = ic_rn_volts / slope_ohm_m2
    else:
        j_c_a_m2 = float("nan")

    return {
        "valid_mask": valid_mask,
        "inverse_area": inverse_area,
        "x_reg": x_reg,
        "y_reg": y_reg,
        "y_stds_reg": y_stds[valid_mask],
        "y_means": y_means,
        "y_stds": y_stds,
        "a0": a0,
        "a1": a1,
        "r2": r2,
        "standard_error": standard_error,
        "J_c_A_cm2": j_c_a_m2 / 1e4,
        "J_c_uA_um2": j_c_a_m2 / 1e6,
    }


def fit_optimal_offset_um(lw1_list, lw2_list, clean_rows, area_override_list=None,
                           bounds_um=(-0.10, 0.10)) -> float:
    """Auto-fits a single scalar offset (added to every row's LW1 and
    LW2 alike, um) that maximizes the regression R^2 -- same idea as
    the reference scripts' USE_OPTIMAL_OFFSET path (there implemented
    inline; here it's reusable). Rows with an area override are left
    untouched by the fit (their area doesn't depend on the offset) but
    still included in the R^2 being maximized."""
    y_means = np.array([np.mean(r) if len(r) > 0 else np.nan for r in clean_rows])
    valid_mask = ~np.isnan(y_means)
    n = len(lw1_list)

    def loss(delta):
        offsets = [delta] * n
        areas = compute_row_areas(lw1_list, lw2_list, offsets, area_override_list)
        inv_a = 1.0 / areas
        reg = linregress(inv_a[valid_mask], y_means[valid_mask])
        return -(reg.rvalue ** 2)

    result = minimize_scalar(loss, bounds=bounds_um, method="bounded")
    return float(result.x)


# --- dataset container (what gets saved for later comparison) --------------

@dataclass
class InterpretationDataset:
    """Everything needed to reproduce one processed R_n vs 1/A dataset
    later, without needing the original measurement table or GUI state
    still open -- the unit of save/load for the Compare Experiments
    subtab. Round-trips through to_json_dict/from_json_dict so several
    of these can be written to and read back from plain JSON files."""
    name: str
    raw_rows: list  # list[list[raw cell values]] -- as-entered, pre-filter
    lw1_nominal_um: list
    lw2_nominal_um: list
    offset_um: list  # per-row, um (already includes any global-apply)
    area_override_um2: list = field(default_factory=list)  # per-row, '' or None = not overridden
    r_base_kohm: float = 19.65
    short_threshold_kohm: float = 2.0
    open_threshold_kohm: float = 500.0
    z_threshold: float = 3.0
    T_c_kelvin: float = 1.2
    created_at: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        if not self.area_override_um2:
            self.area_override_um2 = ["" for _ in self.raw_rows]

    def process(self) -> dict:
        """Runs the full filter -> area -> regression pipeline and
        returns the combined result dict (filter_and_regress's keys
        plus regress's keys, or None under 'fit' if too few points)."""
        filt = filter_and_regress(self.raw_rows, self.r_base_kohm, self.short_threshold_kohm,
                                   self.open_threshold_kohm, self.z_threshold)
        areas = compute_row_areas(self.lw1_nominal_um, self.lw2_nominal_um, self.offset_um,
                                   self.area_override_um2)
        fit = regress(filt["clean_rows"], areas, self.T_c_kelvin)
        return {**filt, "areas_um2": areas, "fit": fit}

    def to_json_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_json_dict(d: dict) -> "InterpretationDataset":
        known = {f for f in InterpretationDataset.__dataclass_fields__}
        return InterpretationDataset(**{k: v for k, v in d.items() if k in known})


def save_datasets(datasets, path: str):
    with open(path, "w") as f:
        json.dump({"datasets": [d.to_json_dict() for d in datasets]}, f, indent=2)


def load_datasets(path: str) -> list:
    with open(path) as f:
        payload = json.load(f)
    return [InterpretationDataset.from_json_dict(d) for d in payload["datasets"]]


# --- measurement CSV import (matches the Measure Resistance tab's own export

_UNIT_SCALE_FROM_KOHM = {"kOhm": 1.0, "Ohm": 1000.0, "MOhm": 0.001}


def parse_measurement_csv(path: str) -> tuple:
    """Reads a CSV exported by the Measure Resistance tab (or the
    standalone measurement_tab.py prototype -- same format): a small
    header block (Experiment:/Pattern:/Exported:), then a
    'Junction Resistances (<unit>)' title row, a column-label row, then
    one row per array row (leading row-index column, then one cell per
    array column). Returns (meta_dict, raw_rows) where raw_rows values
    are converted back to kOhm (dividing out whatever unit the file was
    exported in) so they can be fed directly into filter_and_regress,
    which expects kOhm throughout, matching r_base_kohm."""
    with open(path, newline="") as f:
        rows = list(csv.reader(f))

    meta = {"experiment": "", "pattern": "", "exported": "", "unit": "kOhm"}
    data_start = None
    for i, row in enumerate(rows):
        if not row or not row[0].strip():
            continue
        key = row[0].strip().rstrip(":").lower()
        if key == "experiment":
            meta["experiment"] = row[1].strip() if len(row) > 1 else ""
        elif key == "pattern":
            meta["pattern"] = row[1].strip() if len(row) > 1 else ""
        elif key == "exported":
            meta["exported"] = row[1].strip() if len(row) > 1 else ""
        elif row[0].strip().lower().startswith("junction resistances"):
            m = re.search(r"\(([^)]*)\)", row[0])
            if m:
                meta["unit"] = m.group(1).strip()
            data_start = i + 2  # skip this title row + the "Column 1, Column 2, ..." header row
            break

    if data_start is None:
        raise ValueError("Not a recognized measurement CSV -- missing a "
                          "'Junction Resistances (...)' header row.")

    scale = _UNIT_SCALE_FROM_KOHM.get(meta["unit"], 1.0)
    raw_rows = []
    for row in rows[data_start:]:
        if not row or not row[0].strip():
            continue
        converted = []
        for cell in row[1:]:
            c = cell.strip()
            low = c.lower()
            if low in ("open", "short", ""):
                converted.append(c if c else 0)
            else:
                try:
                    converted.append(float(c) / scale)
                except ValueError:
                    converted.append(c)  # leave text (e.g. a hyphenated range) for parse_cell to handle
        raw_rows.append(converted)

    return meta, raw_rows
