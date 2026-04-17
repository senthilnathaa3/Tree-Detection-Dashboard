"""
ESA WorldCover AOI summary and model consistency helpers.

Supports local raster path (GeoTIFF). If GDAL/rasterio in your runtime supports
remote URLs, HTTP(S) paths may also work.
"""

import math
from typing import Any, Dict

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import rowcol
from rasterio.windows import Window
from rasterio.warp import transform_bounds


WORLD_COVER_LABELS = {
    10: "Tree cover",
    20: "Shrubland",
    30: "Grassland",
    40: "Cropland",
    50: "Built-up",
    60: "Bare / sparse vegetation",
    70: "Snow and ice",
    80: "Permanent water bodies",
    90: "Herbaceous wetland",
    95: "Mangroves",
    100: "Moss and lichen",
}

TREE_CLASS = 10


def summarize_worldcover_aoi(
    worldcover_path: str,
    west: float,
    south: float,
    east: float,
    north: float,
) -> Dict[str, Any]:
    with rasterio.open(worldcover_path) as src:
        if src.crs and src.crs != CRS.from_epsg(4326):
            left, bottom, right, top = transform_bounds(
                CRS.from_epsg(4326),
                src.crs,
                west,
                south,
                east,
                north,
            )
        else:
            left, bottom, right, top = west, south, east, north

        # Convert AOI bounds to pixel window.
        row_min, col_min = rowcol(src.transform, left, top)
        row_max, col_max = rowcol(src.transform, right, bottom)

        r0 = max(0, min(row_min, row_max))
        r1 = min(src.height - 1, max(row_min, row_max))
        c0 = max(0, min(col_min, col_max))
        c1 = min(src.width - 1, max(col_min, col_max))

        if r0 > r1 or c0 > c1:
            return {
                "pixels_in_aoi": 0,
                "tree_cover_fraction": None,
                "class_distribution": {},
                "message": "AOI is outside provided WorldCover raster extent.",
            }

        window = Window(c0, r0, (c1 - c0 + 1), (r1 - r0 + 1))
        arr = src.read(1, window=window)

        nodata = src.nodata
        if nodata is not None:
            valid = arr[arr != nodata]
        else:
            valid = arr

        valid = valid[np.isfinite(valid)]

        if valid.size == 0:
            return {
                "pixels_in_aoi": 0,
                "tree_cover_fraction": None,
                "class_distribution": {},
                "message": "No valid WorldCover pixels found in AOI window.",
            }

        unique, counts = np.unique(valid.astype(np.int32), return_counts=True)
        total = int(np.sum(counts))

        class_distribution: Dict[str, Any] = {}
        tree_pixels = 0

        for cls, cnt in zip(unique.tolist(), counts.tolist()):
            frac = cnt / total if total > 0 else 0.0
            class_distribution[str(cls)] = {
                "label": WORLD_COVER_LABELS.get(cls, "Unknown"),
                "pixels": int(cnt),
                "fraction": round(float(frac), 6),
            }
            if cls == TREE_CLASS:
                tree_pixels = int(cnt)

        tree_fraction = tree_pixels / total if total > 0 else 0.0

        dominant_cls = int(unique[int(np.argmax(counts))])

        return {
            "pixels_in_aoi": total,
            "tree_cover_fraction": round(float(tree_fraction), 6),
            "tree_cover_percent": round(float(tree_fraction * 100.0), 2),
            "dominant_class": {
                "code": dominant_cls,
                "label": WORLD_COVER_LABELS.get(dominant_cls, "Unknown"),
            },
            "class_distribution": class_distribution,
        }


def compare_model_to_worldcover(
    model_summary: Dict[str, Any],
    worldcover_summary: Dict[str, Any],
) -> Dict[str, Any]:
    model_density = model_summary.get("mean_density")
    wc_tree_fraction = worldcover_summary.get("tree_cover_fraction")

    consistency: Dict[str, Any] = {
        "model_mean_density": model_density,
        "worldcover_tree_fraction": wc_tree_fraction,
    }

    if model_density is not None and wc_tree_fraction is not None:
        abs_diff = abs(float(model_density) - float(wc_tree_fraction))
        consistency_score = max(0.0, 1.0 - abs_diff)
        consistency.update(
            {
                "absolute_difference": round(abs_diff, 6),
                "consistency_score_0_to_1": round(consistency_score, 6),
            }
        )

    return {
        "density_vs_treecover": consistency,
        "notes": [
            "WorldCover tree fraction is canopy land-cover proportion, while model density is a learned tree-density index.",
            "Use as an external consistency check, not strict per-pixel ground truth.",
        ],
    }
