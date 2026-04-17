"""
Helpers to convert FIA DataMart exports (ZIP or directory of CSVs)
into the simplified FIA CSV schema used by this dashboard.
"""

from __future__ import annotations

import csv
import os
import zipfile
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple


PLOT_FILE_NAMES = {"plot.csv"}
TREE_FILE_NAMES = {"tree.csv"}

PLOT_ID_KEYS = ["cn", "plot_cn", "plt_cn", "pltcn"]
PLOT_LAT_KEYS = ["lat", "latitude", "plot_lat", "plt_lat"]
PLOT_LON_KEYS = ["lon", "lng", "longitude", "plot_lon", "plt_lon"]
PLOT_YEAR_KEYS = ["invyr", "year", "inventory_year", "measyear"]

TREE_PLOT_ID_KEYS = ["plt_cn", "plot_cn", "pltcnt", "cn", "plot"]
TREE_SPECIES_KEYS = ["spcd", "species", "species_code"]
TREE_TPA_KEYS = ["tpa_unadj", "tpa", "trees_per_acre"]


def _norm(s: str) -> str:
    return "".join(ch.lower() for ch in str(s).strip() if ch.isalnum() or ch == "_")


def _find_key(row: Dict[str, Any], candidates: List[str]) -> Optional[str]:
    keys = { _norm(k): k for k in row.keys() }
    for c in candidates:
        n = _norm(c)
        if n in keys:
            return keys[n]
    return None


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


def _parse_int(v: Any) -> Optional[int]:
    f = _parse_float(v)
    if f is None:
        return None
    return int(f)


def _iter_csv_rows_from_dir(root_dir: str, target_filename: str) -> Iterable[Dict[str, Any]]:
    target = target_filename.lower()
    for entry in os.listdir(root_dir):
        if entry.lower() == target:
            with open(os.path.join(root_dir, entry), "r", newline="", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    yield row
            return


def _iter_csv_rows_from_zip(zip_path: str, target_filename: str) -> Iterable[Dict[str, Any]]:
    target = target_filename.lower()
    with zipfile.ZipFile(zip_path, "r") as zf:
        candidate = None
        for name in zf.namelist():
            if name.lower().endswith(target):
                candidate = name
                break

        if candidate is None:
            return

        with zf.open(candidate, "r") as fh:
            import io
            wrapper = io.TextIOWrapper(fh, encoding="utf-8-sig", newline="")
            reader = csv.DictReader(wrapper)
            for row in reader:
                yield row


def _iter_rows(source_path: str, filename: str) -> Iterable[Dict[str, Any]]:
    if os.path.isdir(source_path):
        return _iter_csv_rows_from_dir(source_path, filename)
    if zipfile.is_zipfile(source_path):
        return _iter_csv_rows_from_zip(source_path, filename)
    raise FileNotFoundError(f"Expected directory or zip file: {source_path}")


def build_fia_csv_from_datamart(source_path: str, output_csv_path: str) -> Dict[str, Any]:
    """
    Convert DataMart PLOT/TREE CSVs into simplified FIA CSV for validation.

    Output columns:
    - source_plot_id
    - lat
    - lon
    - year
    - species
    - trees_per_acre
    - trees_per_hectare
    """
    plot_rows = list(_iter_rows(source_path, "PLOT.csv"))
    if not plot_rows:
        raise ValueError("Could not find PLOT.csv in provided DataMart source")

    first_plot = plot_rows[0]
    plot_id_key = _find_key(first_plot, PLOT_ID_KEYS)
    lat_key = _find_key(first_plot, PLOT_LAT_KEYS)
    lon_key = _find_key(first_plot, PLOT_LON_KEYS)
    year_key = _find_key(first_plot, PLOT_YEAR_KEYS)

    if not plot_id_key:
        raise ValueError("PLOT.csv is missing a plot id column (CN/PLT_CN/...) ")
    if not lat_key or not lon_key:
        raise ValueError("PLOT.csv is missing latitude/longitude columns")

    plots: Dict[str, Dict[str, Any]] = {}
    skipped_plot_rows = 0

    for row in plot_rows:
        pid = str(row.get(plot_id_key, "")).strip()
        if not pid:
            skipped_plot_rows += 1
            continue

        lat = _parse_float(row.get(lat_key))
        lon = _parse_float(row.get(lon_key))
        if lat is None or lon is None:
            skipped_plot_rows += 1
            continue

        plots[pid] = {
            "source_plot_id": pid,
            "lat": lat,
            "lon": lon,
            "year": _parse_int(row.get(year_key)) if year_key else None,
        }

    if not plots:
        raise ValueError("No valid PLOT rows with lat/lon were found")

    tree_rows = list(_iter_rows(source_path, "TREE.csv"))

    trees_per_plot = defaultdict(float)
    species_weight = defaultdict(lambda: defaultdict(float))

    missing_tree_plot_links = 0

    if tree_rows:
        first_tree = tree_rows[0]
        tree_plot_key = _find_key(first_tree, TREE_PLOT_ID_KEYS)
        tree_species_key = _find_key(first_tree, TREE_SPECIES_KEYS)
        tree_tpa_key = _find_key(first_tree, TREE_TPA_KEYS)

        if tree_plot_key and tree_tpa_key:
            for row in tree_rows:
                pid = str(row.get(tree_plot_key, "")).strip()
                if not pid or pid not in plots:
                    missing_tree_plot_links += 1
                    continue

                tpa = _parse_float(row.get(tree_tpa_key))
                if tpa is None:
                    continue

                trees_per_plot[pid] += tpa

                if tree_species_key:
                    species = str(row.get(tree_species_key, "")).strip() or "Unknown"
                    species_weight[pid][species] += tpa

    os.makedirs(os.path.dirname(output_csv_path) or ".", exist_ok=True)

    rows_out: List[Dict[str, Any]] = []
    for pid, meta in plots.items():
        tpa = trees_per_plot.get(pid)
        dominant_species = "Unknown"

        if species_weight.get(pid):
            dominant_species = max(species_weight[pid].items(), key=lambda x: x[1])[0]

        row = {
            "source_plot_id": meta["source_plot_id"],
            "lat": round(float(meta["lat"]), 8),
            "lon": round(float(meta["lon"]), 8),
            "year": meta["year"],
            "species": dominant_species,
            "trees_per_acre": round(float(tpa), 6) if tpa is not None else None,
            "trees_per_hectare": round(float(tpa * 2.47105381), 6) if tpa is not None else None,
        }
        rows_out.append(row)

    with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "source_plot_id",
                "lat",
                "lon",
                "year",
                "species",
                "trees_per_acre",
                "trees_per_hectare",
            ],
        )
        writer.writeheader()
        writer.writerows(rows_out)

    return {
        "status": "success",
        "source_path": os.path.abspath(source_path),
        "output_csv": os.path.abspath(output_csv_path),
        "plots_total": len(plot_rows),
        "plots_valid": len(plots),
        "plots_skipped": skipped_plot_rows,
        "tree_rows_total": len(tree_rows),
        "tree_rows_unlinked": missing_tree_plot_links,
        "rows_written": len(rows_out),
    }
