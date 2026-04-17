"""
FIA-based AOI cross-check utilities.

This module expects an FIA-like CSV containing point records with latitude/longitude
and optional inventory year, species/group, and tree density fields.
"""

import csv
import os
import statistics
from typing import Any, Dict, List, Optional


LAT_KEYS = ["lat", "latitude", "plot_lat", "plot_latitude", "plt_lat"]
LON_KEYS = ["lon", "lng", "longitude", "plot_lon", "plot_longitude", "plt_lon"]
YEAR_KEYS = ["year", "inventory_year", "measurement_year", "inv_year"]
SPECIES_KEYS = ["species", "species_code", "species_group", "forest_type", "spcd"]
TPH_KEYS = ["trees_per_hectare", "trees_ha", "tph", "tpha"]
TPA_KEYS = ["trees_per_acre", "tpa"]


def _pick(row: Dict[str, Any], keys: List[str]) -> Optional[str]:
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return row[key]
    return None


def _parse_float(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def _parse_int(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return None


def _normalize_species(name: Optional[str]) -> str:
    if not name:
        return "unknown"
    cleaned = "".join(ch.lower() for ch in str(name) if ch.isalnum())
    return cleaned or "unknown"


def load_fia_csv(csv_path: str) -> Dict[str, Any]:
    """
    Load FIA-like records from CSV.

    Required logical fields: latitude, longitude.
    Optional fields: year, species, trees_per_hectare or trees_per_acre.
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"FIA CSV not found: {csv_path}")

    records: List[Dict[str, Any]] = []
    skipped = 0

    with open(csv_path, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("FIA CSV appears empty or has no header row")

        for raw in reader:
            row = {str(k).strip().lower(): v for k, v in raw.items() if k is not None}

            lat = _parse_float(_pick(row, LAT_KEYS))
            lon = _parse_float(_pick(row, LON_KEYS))
            if lat is None or lon is None:
                skipped += 1
                continue

            tph = _parse_float(_pick(row, TPH_KEYS))
            if tph is None:
                tpa = _parse_float(_pick(row, TPA_KEYS))
                tph = tpa * 2.47105381 if tpa is not None else None

            record = {
                "lat": lat,
                "lon": lon,
                "year": _parse_int(_pick(row, YEAR_KEYS)),
                "species": (_pick(row, SPECIES_KEYS) or "Unknown").strip(),
                "species_norm": _normalize_species(_pick(row, SPECIES_KEYS)),
                "trees_per_hectare": tph,
            }
            records.append(record)

    if not records:
        raise ValueError("No valid FIA records found after parsing CSV")

    return {
        "records": records,
        "source": os.path.abspath(csv_path),
        "rows_loaded": len(records),
        "rows_skipped": skipped,
    }


def filter_fia_records(
    records: List[Dict[str, Any]],
    west: float,
    south: float,
    east: float,
    north: float,
    year_start: Optional[int] = None,
    year_end: Optional[int] = None,
) -> List[Dict[str, Any]]:
    filtered: List[Dict[str, Any]] = []
    for r in records:
        if not (west <= r["lon"] <= east and south <= r["lat"] <= north):
            continue

        year = r.get("year")
        if year_start is not None and (year is None or year < year_start):
            continue
        if year_end is not None and (year is None or year > year_end):
            continue

        filtered.append(r)
    return filtered


def summarize_fia(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not records:
        return {
            "plots_in_aoi": 0,
            "years": None,
            "trees_per_hectare": {},
            "dominant_species": {},
        }

    tph_values = [r["trees_per_hectare"] for r in records if r.get("trees_per_hectare") is not None]

    species_counts: Dict[str, int] = {}
    for r in records:
        species = r.get("species") or "Unknown"
        species_counts[species] = species_counts.get(species, 0) + 1

    sorted_species = dict(sorted(species_counts.items(), key=lambda x: x[1], reverse=True)[:20])

    years = [r["year"] for r in records if r.get("year") is not None]
    year_summary = (
        {"min": min(years), "max": max(years), "count_with_year": len(years)}
        if years
        else None
    )

    tph_summary: Dict[str, Any] = {}
    if tph_values:
        tph_summary = {
            "count": len(tph_values),
            "mean": round(statistics.fmean(tph_values), 2),
            "median": round(statistics.median(tph_values), 2),
            "min": round(min(tph_values), 2),
            "max": round(max(tph_values), 2),
            "stdev": round(statistics.pstdev(tph_values), 2) if len(tph_values) > 1 else 0.0,
        }

    return {
        "plots_in_aoi": len(records),
        "years": year_summary,
        "trees_per_hectare": tph_summary,
        "dominant_species": sorted_species,
    }


def compare_model_to_fia(
    model_summary: Dict[str, Any],
    fia_summary: Dict[str, Any],
) -> Dict[str, Any]:
    model_tph = model_summary.get("mean_trees_per_hectare")
    fia_tph = fia_summary.get("trees_per_hectare", {}).get("mean")

    density_agreement: Dict[str, Any] = {
        "model_mean_trees_per_hectare": model_tph,
        "fia_mean_trees_per_hectare": fia_tph,
    }

    if model_tph is not None and fia_tph is not None:
        abs_diff = model_tph - fia_tph
        pct_diff = (abs_diff / fia_tph * 100.0) if fia_tph != 0 else None
        density_agreement.update(
            {
                "absolute_difference": round(abs_diff, 2),
                "percent_difference": round(pct_diff, 2) if pct_diff is not None else None,
            }
        )

    model_species = model_summary.get("dominant_species_distribution", {}) or {}
    fia_species = fia_summary.get("dominant_species", {}) or {}

    model_norm = {_normalize_species(k): v for k, v in model_species.items()}
    fia_norm = {_normalize_species(k): v for k, v in fia_species.items()}

    overlap = sorted(set(model_norm.keys()) & set(fia_norm.keys()))
    overlap_pairs = [
        {
            "species_norm": sp,
            "model_count": model_norm[sp],
            "fia_count": fia_norm[sp],
        }
        for sp in overlap
    ]

    return {
        "density_agreement": density_agreement,
        "species_overlap": {
            "overlap_count": len(overlap),
            "model_species_count": len(model_norm),
            "fia_species_count": len(fia_norm),
            "overlap_details": overlap_pairs,
        },
        "notes": [
            "FIA cross-check is AOI-level consistency, not per-pixel ground truth.",
            "Species overlap uses normalized string matching and may require a taxonomy crosswalk.",
        ],
    }
