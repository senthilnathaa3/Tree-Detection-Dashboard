"""
AOI-bounded dataset analysis utilities.

This module filters paired S1/S2 tiles by a WGS84 AOI bounding box,
runs model inference on intersecting tiles, and returns aggregated outputs.
"""

import math
from typing import Any, Dict, List, Optional, Tuple

import rasterio
from rasterio.crs import CRS
from rasterio.warp import transform_bounds

from .inference import discover_dataset, run_inference_paired
from .metrics import (
    compute_batch_biodiversity,
    compute_density_statistics,
    compute_species_summary,
)

Bounds = Dict[str, float]


def _to_wgs84_bounds(path: str) -> Optional[Bounds]:
    with rasterio.open(path) as src:
        if not src.bounds:
            return None

        if src.crs and src.crs != CRS.from_epsg(4326):
            west, south, east, north = transform_bounds(
                src.crs,
                CRS.from_epsg(4326),
                src.bounds.left,
                src.bounds.bottom,
                src.bounds.right,
                src.bounds.top,
            )
        else:
            west, south, east, north = (
                src.bounds.left,
                src.bounds.bottom,
                src.bounds.right,
                src.bounds.top,
            )

    return {
        "west": float(west),
        "south": float(south),
        "east": float(east),
        "north": float(north),
    }


def _intersects(a: Bounds, b: Bounds) -> bool:
    return not (
        a["east"] < b["west"]
        or a["west"] > b["east"]
        or a["north"] < b["south"]
        or a["south"] > b["north"]
    )


def _rounded_bounds(bounds: Bounds) -> Bounds:
    return {
        "west": round(bounds["west"], 6),
        "south": round(bounds["south"], 6),
        "east": round(bounds["east"], 6),
        "north": round(bounds["north"], 6),
    }


def _aoi_area_km2(aoi: Bounds) -> float:
    center_lat = (aoi["south"] + aoi["north"]) / 2
    lat_km = abs(aoi["north"] - aoi["south"]) * 111.32
    lon_km = abs(aoi["east"] - aoi["west"]) * 111.32 * math.cos(math.radians(center_lat))
    return round(lat_km * lon_km, 2)


def _aggregate_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    densities = [r["density"] for r in results]
    distributions = [r["species_distribution"] for r in results]

    dominant_counts: Dict[str, int] = {}
    for r in results:
        sp = r["dominant_species"]
        dominant_counts[sp] = dominant_counts.get(sp, 0) + 1

    avg_tree_count = sum(r["tree_count"] for r in results) / len(results) if results else 0.0
    avg_trees_per_hectare = (
        sum(r["trees_per_hectare"] for r in results) / len(results) if results else 0.0
    )

    return {
        "total_tiles_analyzed": len(results),
        "mean_density": round(sum(densities) / len(densities), 4) if densities else 0.0,
        "avg_tree_count": round(avg_tree_count, 1),
        "mean_trees_per_hectare": round(avg_trees_per_hectare, 2),
        "dominant_species_distribution": dominant_counts,
        "density_statistics": compute_density_statistics(densities) if densities else {},
        "species_summary": compute_species_summary(distributions) if distributions else {},
        "biodiversity_aggregate": compute_batch_biodiversity(results) if results else {},
    }


def analyze_dataset_with_aoi(
    dataset_path: str,
    aoi: Bounds,
    species_threshold: float = 0.5,
    max_tiles: Optional[int] = None,
    include_per_tile: bool = False,
) -> Dict[str, Any]:
    """
    Filter paired S1/S2 tiles by AOI and run inference on intersecting tiles.
    """
    pairs: List[Tuple[str, str]] = discover_dataset(dataset_path)

    selected: List[Tuple[str, str, Bounds]] = []
    scan_errors = 0

    for s2_path, s1_path in pairs:
        try:
            tile_bounds = _to_wgs84_bounds(s2_path)
            if tile_bounds and _intersects(tile_bounds, aoi):
                selected.append((s2_path, s1_path, tile_bounds))
                if max_tiles and len(selected) >= max_tiles:
                    break
        except Exception:
            scan_errors += 1

    if not selected:
        return {
            "aoi": {
                **_rounded_bounds(aoi),
                "area_km2": _aoi_area_km2(aoi),
            },
            "tiles_scanned": len(pairs),
            "tiles_intersecting": 0,
            "scan_errors": scan_errors,
            "summary": {},
            "per_tile_results": [] if include_per_tile else None,
            "message": "No dataset tiles intersect the AOI.",
        }

    results: List[Dict[str, Any]] = []
    for s2_path, s1_path, tile_bounds in selected:
        r = run_inference_paired(s2_path, s1_path, species_threshold=species_threshold)
        r["tile_bounds"] = _rounded_bounds(tile_bounds)
        results.append(r)

    payload: Dict[str, Any] = {
        "aoi": {
            **_rounded_bounds(aoi),
            "area_km2": _aoi_area_km2(aoi),
        },
        "tiles_scanned": len(pairs),
        "tiles_intersecting": len(selected),
        "scan_errors": scan_errors,
        "summary": _aggregate_results(results),
        "intersecting_files": [r["filename"] for r in results],
    }

    if include_per_tile:
        payload["per_tile_results"] = results

    return payload
