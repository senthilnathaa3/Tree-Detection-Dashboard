"""Calibration helpers for model tree-density outputs against FIA references."""

from __future__ import annotations

import csv
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _parse_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    t = str(v).strip()
    if not t:
        return None
    try:
        return float(t)
    except ValueError:
        return None


def load_calibration_samples_csv(csv_path: str) -> List[Tuple[float, float]]:
    """
    Load (model_tph, fia_tph) sample pairs from CSV.
    Accepted headers:
    - model_tph, fia_tph
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Calibration CSV not found: {csv_path}")

    rows: List[Tuple[float, float]] = []
    with open(csv_path, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("Calibration CSV is empty or missing header")

        fields = {str(h).strip().lower(): h for h in reader.fieldnames if h is not None}
        if "model_tph" not in fields or "fia_tph" not in fields:
            raise ValueError("Calibration CSV must contain headers: model_tph, fia_tph")

        model_key = fields["model_tph"]
        fia_key = fields["fia_tph"]

        for row in reader:
            m = _parse_float(row.get(model_key))
            y = _parse_float(row.get(fia_key))
            if m is None or y is None:
                continue
            rows.append((m, y))

    if not rows:
        raise ValueError("No valid sample rows found in calibration CSV")

    return rows


def fit_linear_tph_calibration(samples: List[Tuple[float, float]]) -> Dict[str, Any]:
    """
    Fit linear calibration: fia_tph = slope * model_tph + intercept.
    """
    if len(samples) < 2:
        raise ValueError("Need at least 2 samples to fit calibration")

    x = np.array([s[0] for s in samples], dtype=np.float64)
    y = np.array([s[1] for s in samples], dtype=np.float64)

    slope, intercept = np.polyfit(x, y, deg=1)
    y_hat = slope * x + intercept

    mae_before = float(np.mean(np.abs(x - y)))
    mae_after = float(np.mean(np.abs(y_hat - y)))
    rmse_before = float(np.sqrt(np.mean((x - y) ** 2)))
    rmse_after = float(np.sqrt(np.mean((y_hat - y) ** 2)))

    sst = float(np.sum((y - np.mean(y)) ** 2))
    sse = float(np.sum((y_hat - y) ** 2))
    r2 = 1.0 - (sse / sst) if sst > 0 else None

    return {
        "sample_count": len(samples),
        "slope": round(float(slope), 8),
        "intercept": round(float(intercept), 8),
        "metrics": {
            "mae_before": round(mae_before, 6),
            "mae_after": round(mae_after, 6),
            "rmse_before": round(rmse_before, 6),
            "rmse_after": round(rmse_after, 6),
            "r2": round(float(r2), 6) if r2 is not None else None,
        },
    }


def apply_linear_tph_calibration(model_tph: float, slope: float, intercept: float) -> float:
    calibrated = slope * float(model_tph) + intercept
    return max(0.0, float(calibrated))


def load_regional_calibration_samples_csv(
    csv_path: str,
    region_column: str = "region",
) -> Dict[str, List[Tuple[float, float]]]:
    """
    Load region -> [(model_tph, fia_tph), ...] from CSV.
    Required columns: model_tph, fia_tph, <region_column>
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Calibration CSV not found: {csv_path}")

    out: Dict[str, List[Tuple[float, float]]] = {}
    with open(csv_path, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("Calibration CSV is empty or missing header")

        fields = {str(h).strip().lower(): h for h in reader.fieldnames if h is not None}
        rc = region_column.strip().lower()
        if "model_tph" not in fields or "fia_tph" not in fields or rc not in fields:
            raise ValueError(
                f"Calibration CSV must contain headers: model_tph, fia_tph, {region_column}"
            )

        model_key = fields["model_tph"]
        fia_key = fields["fia_tph"]
        region_key = fields[rc]

        for row in reader:
            m = _parse_float(row.get(model_key))
            y = _parse_float(row.get(fia_key))
            region = str(row.get(region_key, "")).strip()
            if m is None or y is None or not region:
                continue
            out.setdefault(region, []).append((m, y))

    if not out:
        raise ValueError("No valid regional rows found in calibration CSV")
    return out


def fit_regional_linear_tph_calibration(
    regional_samples: Dict[str, List[Tuple[float, float]]],
    min_samples_per_region: int = 5,
) -> Dict[str, Any]:
    """
    Fit per-region calibration where sample count is sufficient, plus global fallback.
    """
    all_samples: List[Tuple[float, float]] = []
    for pairs in regional_samples.values():
        all_samples.extend(pairs)
    if len(all_samples) < 2:
        raise ValueError("Need at least 2 total samples for global fallback calibration")

    global_fit = fit_linear_tph_calibration(all_samples)
    regions: Dict[str, Any] = {}

    for region, pairs in regional_samples.items():
        if len(pairs) < min_samples_per_region:
            regions[region] = {
                "sample_count": len(pairs),
                "fit": None,
                "reason": f"insufficient_samples(<{min_samples_per_region})",
            }
            continue
        regions[region] = {
            "sample_count": len(pairs),
            "fit": fit_linear_tph_calibration(pairs),
        }

    return {
        "global_fallback": global_fit,
        "regions": regions,
        "min_samples_per_region": min_samples_per_region,
    }


def save_calibration_profile(profile: Dict[str, Any], output_path: str):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2)


def load_calibration_profile(path: str) -> Dict[str, Any]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Calibration profile not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def pick_calibration_from_profile(
    profile: Dict[str, Any],
    region: Optional[str],
) -> Tuple[float, float, str]:
    """
    Select slope/intercept from regional profile.
    Returns (slope, intercept, source) where source is region or 'global_fallback'.
    """
    if region:
        regions = profile.get("regions", {})
        region_block = regions.get(region)
        if region_block and region_block.get("fit"):
            fit = region_block["fit"]
            return float(fit["slope"]), float(fit["intercept"]), region

    global_fit = profile.get("global_fallback")
    if not global_fit:
        raise ValueError("Calibration profile missing global_fallback fit")
    return float(global_fit["slope"]), float(global_fit["intercept"]), "global_fallback"
