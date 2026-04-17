"""
Lightweight tree crown candidate detection from multispectral imagery.

This is a classical image-processing detector (NDVI + connected components),
intended as an object-level visualization baseline.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, List

import numpy as np
import rasterio


def _normalize_ndvi(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    denom = nir + red
    ndvi = np.where(denom > 1e-6, (nir - red) / denom, 0.0)
    ndvi = np.clip(ndvi, -1.0, 1.0)
    return ndvi


def _connected_components(mask: np.ndarray, min_area_px: int) -> List[Dict[str, Any]]:
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=np.uint8)
    detections: List[Dict[str, Any]] = []

    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or visited[r, c]:
                continue

            q = deque([(r, c)])
            visited[r, c] = 1

            area = 0
            rmin = rmax = r
            cmin = cmax = c
            rs = 0.0
            cs = 0.0

            while q:
                cr, cc = q.popleft()
                area += 1
                rs += cr
                cs += cc
                rmin = min(rmin, cr)
                rmax = max(rmax, cr)
                cmin = min(cmin, cc)
                cmax = max(cmax, cc)

                for dr, dc in neighbors:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not visited[nr, nc]:
                        visited[nr, nc] = 1
                        q.append((nr, nc))

            if area < min_area_px:
                continue

            detections.append(
                {
                    "bbox_px": {
                        "xmin": int(cmin),
                        "ymin": int(rmin),
                        "xmax": int(cmax),
                        "ymax": int(rmax),
                    },
                    "area_px": int(area),
                    "centroid_px": {
                        "x": float(cs / area),
                        "y": float(rs / area),
                    },
                }
            )

    return detections


def detect_tree_crowns_ndvi(
    tif_path: str,
    ndvi_threshold: float = 0.45,
    min_area_px: int = 12,
) -> Dict[str, Any]:
    """
    Detect tree crown candidates from a GeoTIFF using NDVI thresholding.

    Expected bands:
    - If >= 8 bands: uses Sentinel convention red=B4(index3), nir=B8(index7)
    - If < 8 bands and >= 2: uses first two bands as (red, nir)
    """
    with rasterio.open(tif_path) as src:
        arr = src.read().astype(np.float32)
        transform = src.transform
        crs = str(src.crs) if src.crs else None

    if arr.shape[0] >= 8:
        red = arr[3]
        nir = arr[7]
    elif arr.shape[0] >= 2:
        red = arr[0]
        nir = arr[1]
    else:
        raise ValueError("Need at least 2 bands for NDVI-based crown detection")

    ndvi = _normalize_ndvi(red, nir)
    mask = ndvi >= float(ndvi_threshold)

    detections = _connected_components(mask, min_area_px=int(min_area_px))

    # Attach rough geo-centroids where transform is available.
    if transform is not None:
        for det in detections:
            x = det["centroid_px"]["x"]
            y = det["centroid_px"]["y"]
            gx, gy = rasterio.transform.xy(transform, y, x)
            det["centroid_geo"] = {"x": float(gx), "y": float(gy)}

    scores = [float(np.mean(ndvi[d["bbox_px"]["ymin"]:d["bbox_px"]["ymax"] + 1,
                                d["bbox_px"]["xmin"]:d["bbox_px"]["xmax"] + 1])) for d in detections]
    for i, det in enumerate(detections):
        det["score"] = round(scores[i], 4)

    detections.sort(key=lambda d: d["score"], reverse=True)

    return {
        "method": "ndvi_connected_components",
        "ndvi_threshold": float(ndvi_threshold),
        "min_area_px": int(min_area_px),
        "candidate_count": len(detections),
        "crs": crs,
        "detections": detections,
    }
