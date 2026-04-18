"""
Lightweight tree crown candidate detection from multispectral imagery.

This is a classical image-processing detector (NDVI + connected components),
intended as an object-level visualization baseline.
"""

from __future__ import annotations

import json
import numpy as np
import rasterio
from typing import Any, Dict, List, Optional
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed


def _normalize_ndvi(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    denom = nir + red + 1e-10
    ndvi = (nir - red) / denom
    ndvi = np.clip(ndvi, -1.0, 1.0)
    return ndvi


def _pick_red_nir(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    bands = int(arr.shape[0])
    if bands >= 15:
        return arr[3], arr[7]
    if bands >= 4:
        # NAIP RGBN
        return arr[0], arr[3]
    if bands >= 2:
        return arr[0], arr[1]
    raise ValueError("Need at least 2 bands for NDVI-based crown detection")


def detect_tree_crowns_advanced(
    tif_path: str,
    ndvi_threshold: float = 0.45,
    min_area_px: int = 12,
    model_tree_count: Optional[float] = None,
    max_candidates: int = 5000,
    include_geojson: bool = False,
) -> Dict[str, Any]:
    """
    Advanced tree crown detection using NDVI peak finding + watershed segmentation.
    If model_tree_count is provided, it uses it to calibrate the final detection list.
    """
    with rasterio.open(tif_path) as src:
        arr = src.read().astype(np.float32)
        transform = src.transform
        crs = str(src.crs) if src.crs else None

    red, nir = _pick_red_nir(arr)

    ndvi = _normalize_ndvi(red, nir)
    
    # 1. Thresholding to get forest mask
    mask = ndvi >= float(ndvi_threshold)
    
    # 2. Find local maxima in NDVI (peaks)
    # distance = min distance between peaks
    coordinates = peak_local_max(ndvi, min_distance=2, labels=mask)
    
    # 3. Watershed segmentation starting from peaks
    peaks_mask = np.zeros(ndvi.shape, dtype=bool)
    peaks_mask[tuple(coordinates.T)] = True
    markers, _ = ndi.label(peaks_mask)
    labels = watershed(-ndvi, markers, mask=mask)
    
    detections = []
    for i in range(1, np.max(labels) + 1):
        component_mask = labels == i
        area = np.sum(component_mask)
        
        if area < min_area_px:
            continue
            
        rows, cols = np.where(component_mask)
        rmin, rmax = rows.min(), rows.max()
        cmin, cmax = cols.min(), cols.max()
        
        # Calculate score as peak NDVI value in this segment
        score = float(np.max(ndvi[component_mask]))
        
        det = {
            "crown_id": int(i),
            "bbox_px": {
                "xmin": int(cmin),
                "ymin": int(rmin),
                "xmax": int(cmax),
                "ymax": int(rmax),
            },
            "area_px": int(area),
            "centroid_px": {
                "x": float(np.mean(cols)),
                "y": float(np.mean(rows)),
            },
            "score": round(score, 4),
        }
        
        if transform is not None:
            gx, gy = rasterio.transform.xy(transform, det["centroid_px"]["y"], det["centroid_px"]["x"])
            det["centroid_geo"] = {"x": float(gx), "y": float(gy)}
            
        detections.append(det)

    detections.sort(key=lambda d: d["score"], reverse=True)

    # 4. Align with model density if provided
    original_count = len(detections)
    if model_tree_count is not None and model_tree_count > 0:
        # We take top N candidates where N is roughly model_tree_count
        # We allow a 20% buffer for candidates that might be real but lower score
        target_count = int(round(model_tree_count))
        if len(detections) > target_count:
            detections = detections[:target_count]
    if max_candidates and max_candidates > 0 and len(detections) > int(max_candidates):
        detections = detections[: int(max_candidates)]

    geojson = None
    if include_geojson:
        features = []
        for d in detections:
            centroid = d.get("centroid_geo")
            if not centroid:
                continue
            features.append(
                {
                    "type": "Feature",
                    "properties": {
                        "crown_id": d.get("crown_id"),
                        "score": d.get("score"),
                        "area_px": d.get("area_px"),
                    },
                    "geometry": {
                        "type": "Point",
                        "coordinates": [centroid["x"], centroid["y"]],
                    },
                }
            )
        geojson = {"type": "FeatureCollection", "features": features}

    out = {
        "method": "ndvi_peak_watershed",
        "ndvi_threshold": float(ndvi_threshold),
        "min_area_px": int(min_area_px),
        "original_candidate_count": original_count,
        "candidate_count": len(detections),
        "model_target_count": model_tree_count,
        "crs": crs,
        "detections": detections,
    }
    if geojson is not None:
        out["detections_geojson"] = geojson
        out["detections_geojson_text"] = json.dumps(geojson)
    return out


def detect_tree_crowns_ndvi(
    tif_path: str,
    ndvi_threshold: float = 0.45,
    min_area_px: int = 12,
) -> Dict[str, Any]:
    """Legacy wrapper for compatibility."""
    return detect_tree_crowns_advanced(tif_path, ndvi_threshold, min_area_px)
