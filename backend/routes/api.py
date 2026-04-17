"""
API route definitions for TreeSat Analytics Dashboard.
Handles upload, predict, preview, batch, dataset analysis, and biodiversity endpoints.
"""

import os
import json
import asyncio
from typing import Optional
import numpy as np
import rasterio
from fastapi import APIRouter, UploadFile, File, HTTPException, Query, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from rasterio.transform import from_bounds as transform_from_bounds

from ..inference import (
    run_inference,
    get_tif_metadata,
    batch_inference,
    discover_dataset,
    save_results_csv,
)
from ..utils import validate_file, save_upload, tif_to_png_preview, UPLOAD_DIR
from ..metrics import (
    compute_density_statistics,
    compute_species_summary,
    compute_batch_biodiversity,
    generate_heatmap_data
)
from ..aoi_analysis import analyze_dataset_with_aoi
from ..fia_validation import (
    load_fia_csv,
    filter_fia_records,
    summarize_fia,
    compare_model_to_fia,
)
from ..evaluation import evaluate_offline
from ..worldcover_validation import (
    summarize_worldcover_aoi,
    compare_model_to_worldcover,
)
from ..remote_inference import (
    run_remote_inference_planetary_computer,
    run_remote_inference_planetary_computer_grid,
    fetch_remote_tensor_planetary_computer,
)
from ..fia_datamart import build_fia_csv_from_datamart
from ..calibration import (
    load_calibration_samples_csv,
    fit_linear_tph_calibration,
    apply_linear_tph_calibration,
    load_regional_calibration_samples_csv,
    fit_regional_linear_tph_calibration,
    save_calibration_profile,
    load_calibration_profile,
    pick_calibration_from_profile,
)
from ..object_detection import detect_tree_crowns_ndvi

router = APIRouter()

# Store results in memory for batch analysis
batch_results = []

# Store analysis job state
analysis_state = {
    "status": "idle",       # idle | running | completed | error
    "progress": 0,
    "total": 0,
    "results": None,
    "error": None,
}


class DatasetAnalysisRequest(BaseModel):
    dataset_path: str
    threshold: float = 0.5
    batch_size: int = 32


class AOIAnalysisRequest(BaseModel):
    dataset_path: str
    west: float
    south: float
    east: float
    north: float
    threshold: float = 0.5
    max_tiles: Optional[int] = None
    include_per_tile: bool = False


class AOIFIAValidationRequest(AOIAnalysisRequest):
    fia_csv_path: str
    year_start: Optional[int] = None
    year_end: Optional[int] = None


class OfflineEvaluationRequest(BaseModel):
    dataset_path: str
    ground_truth_csv: str
    species_threshold: float = 0.5
    threshold_grid: Optional[list[float]] = None
    output_dir: Optional[str] = None


class LocationValidationRequest(BaseModel):
    dataset_path: Optional[str] = None
    lat: float
    lon: float
    radius_km: float = 10.0
    threshold: float = 0.5
    max_tiles: Optional[int] = None
    validation_source: str = "fia"  # fia | esa_worldcover
    provider: str = "planetary_computer"
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    cloud_cover_max: float = 40.0
    fia_csv_path: Optional[str] = None
    worldcover_path: Optional[str] = None
    year_start: Optional[int] = None
    year_end: Optional[int] = None
    calibration_slope: Optional[float] = None
    calibration_intercept: Optional[float] = None
    calibration_profile_path: Optional[str] = None
    calibration_region: Optional[str] = None
    sample_grid_size: int = 1


class FIADatamartConvertRequest(BaseModel):
    source_path: str
    output_csv_path: Optional[str] = None


class FIACalibrationFitRequest(BaseModel):
    calibration_csv_path: str


class FiaRegionalCalibrationFitRequest(BaseModel):
    calibration_csv_path: str
    region_column: str = "region"
    min_samples_per_region: int = 5
    output_profile_path: Optional[str] = None


class RemoteGeoTiffFetchRequest(BaseModel):
    lat: float
    lon: float
    start_date: str
    end_date: str
    radius_km: float = 0.2
    provider: str = "planetary_computer"
    cloud_cover_max: float = 40.0


def _validate_dataset_structure(dataset_path: str):
    if not os.path.isdir(dataset_path):
        raise HTTPException(status_code=400, detail=f"Dataset path not found: {dataset_path}")

    s1_dir = os.path.join(dataset_path, "s1")
    s2_dir = os.path.join(dataset_path, "s2")
    if not os.path.isdir(s1_dir) or not os.path.isdir(s2_dir):
        raise HTTPException(
            status_code=400,
            detail="Dataset must contain 's1/' and 's2/' subdirectories.",
        )


def _validate_aoi_bounds(west: float, south: float, east: float, north: float):
    if not (-180.0 <= west <= 180.0 and -180.0 <= east <= 180.0):
        raise HTTPException(status_code=400, detail="AOI longitude must be between -180 and 180.")
    if not (-90.0 <= south <= 90.0 and -90.0 <= north <= 90.0):
        raise HTTPException(status_code=400, detail="AOI latitude must be between -90 and 90.")
    if west >= east:
        raise HTTPException(status_code=400, detail="AOI bounds invalid: west must be less than east.")
    if south >= north:
        raise HTTPException(status_code=400, detail="AOI bounds invalid: south must be less than north.")


def _validate_lat_lon(lat: float, lon: float):
    if not (-90.0 <= lat <= 90.0):
        raise HTTPException(status_code=400, detail="Latitude must be between -90 and 90.")
    if not (-180.0 <= lon <= 180.0):
        raise HTTPException(status_code=400, detail="Longitude must be between -180 and 180.")


def _bbox_from_center_radius(lat: float, lon: float, radius_km: float) -> dict:
    import math

    lat_delta = radius_km / 111.32
    cos_lat = max(0.01, math.cos(math.radians(lat)))
    lon_delta = radius_km / (111.32 * cos_lat)

    west = max(-180.0, lon - lon_delta)
    east = min(180.0, lon + lon_delta)
    south = max(-90.0, lat - lat_delta)
    north = min(90.0, lat + lat_delta)

    return {
        "west": west,
        "south": south,
        "east": east,
        "north": north,
    }


def _model_summary_from_single_result(result: dict) -> dict:
    dominant_counts = {}
    dominant = result.get("dominant_species")
    if dominant:
        dominant_counts[dominant] = 1
    return {
        "total_tiles_analyzed": 1,
        "mean_density": result.get("density", 0.0),
        "avg_tree_count": result.get("tree_count", 0.0),
        "mean_trees_per_hectare": result.get("trees_per_hectare", 0.0),
        "dominant_species_distribution": dominant_counts,
    }


# ─── Browse Server Directories ──────────────────────────────────────────

@router.get("/browse")
async def browse_directory(path: str = Query("/", description="Directory path to list")):
    """
    List subdirectories and .tif file counts at a given server path.
    Used by the frontend folder-picker to navigate the filesystem.
    """
    # Default to home directory if root
    if path == "/" or not path:
        path = os.path.expanduser("~")

    # Normalize and resolve
    target = os.path.abspath(os.path.expanduser(path))

    if not os.path.isdir(target):
        raise HTTPException(status_code=400, detail=f"Not a directory: {target}")

    entries = []
    try:
        for entry in sorted(os.scandir(target), key=lambda e: e.name.lower()):
            # Skip hidden dirs
            if entry.name.startswith('.'):
                continue
            if entry.is_dir(follow_symlinks=False):
                # Count .tif children (shallow)
                tif_count = 0
                try:
                    tif_count = sum(
                        1 for f in os.scandir(entry.path)
                        if f.is_file() and f.name.lower().endswith(('.tif', '.tiff'))
                    )
                except PermissionError:
                    pass

                entries.append({
                    "name": entry.name,
                    "path": entry.path,
                    "is_dir": True,
                    "tif_count": tif_count,
                })
    except PermissionError:
        raise HTTPException(status_code=403, detail=f"Permission denied: {target}")

    # Check if this directory has s1/ and s2/ subdirectories
    has_s1 = os.path.isdir(os.path.join(target, "s1"))
    has_s2 = os.path.isdir(os.path.join(target, "s2"))

    return {
        "current_path": target,
        "parent_path": os.path.dirname(target),
        "entries": entries,
        "is_dataset": has_s1 and has_s2,
        "has_s1": has_s1,
        "has_s2": has_s2,
    }


# ─── Dataset Geographic Bounds ──────────────────────────────────────────

@router.get("/dataset-bounds")
async def dataset_bounds(dataset_path: str = Query(..., description="Path to dataset directory")):
    """
    Compute the combined geographic bounding box from all .tif files in the dataset.
    Only reads metadata — does NOT load raster data, so it's fast for thousands of files.
    """
    import rasterio
    from rasterio.crs import CRS
    from rasterio.warp import transform_bounds

    target = os.path.abspath(os.path.expanduser(dataset_path))

    if not os.path.isdir(target):
        raise HTTPException(status_code=400, detail=f"Not a directory: {target}")

    # Collect all .tif files from s1/ and s2/ subdirectories (or root)
    tif_files = []
    for subdir in ["s1", "s2", "."]:
        scan_dir = os.path.join(target, subdir) if subdir != "." else target
        if not os.path.isdir(scan_dir):
            continue
        for fname in os.listdir(scan_dir):
            if fname.lower().endswith((".tif", ".tiff")):
                tif_files.append(os.path.join(scan_dir, fname))

    if not tif_files:
        raise HTTPException(status_code=404, detail="No .tif files found in the dataset directory")

    # Deduplicate by absolute path (s1/ and root might overlap)
    tif_files = list(set(tif_files))

    west, south, east, north = float("inf"), float("inf"), float("-inf"), float("-inf")
    total_tiles = 0
    errors = 0

    for fpath in tif_files:
        try:
            with rasterio.open(fpath) as src:
                bounds = src.bounds
                file_crs = src.crs

                # If the CRS is not EPSG:4326 (WGS84), reproject bounds
                if file_crs and file_crs != CRS.from_epsg(4326):
                    try:
                        b = transform_bounds(file_crs, CRS.from_epsg(4326),
                                             bounds.left, bounds.bottom,
                                             bounds.right, bounds.top)
                        left, bottom, right, top = b
                    except Exception:
                        left, bottom, right, top = bounds.left, bounds.bottom, bounds.right, bounds.top
                else:
                    left, bottom, right, top = bounds.left, bounds.bottom, bounds.right, bounds.top

                west = min(west, left)
                south = min(south, bottom)
                east = max(east, right)
                north = max(north, top)
                total_tiles += 1
        except Exception:
            errors += 1

    if total_tiles == 0:
        raise HTTPException(status_code=400, detail="Could not extract bounds from any .tif files")

    center_lat = (south + north) / 2
    center_lon = (west + east) / 2

    # Approximate area in km² using Haversine-style calculation
    import math
    lat_km = abs(north - south) * 111.32
    lon_km = abs(east - west) * 111.32 * math.cos(math.radians(center_lat))
    area_km2 = round(lat_km * lon_km, 2)

    return {
        "west": round(west, 6),
        "south": round(south, 6),
        "east": round(east, 6),
        "north": round(north, 6),
        "center_lat": round(center_lat, 6),
        "center_lon": round(center_lon, 6),
        "area_km2": area_km2,
        "total_tiles": total_tiles,
        "errors": errors,
    }


@router.post("/analyze-aoi")
async def analyze_aoi(request: AOIAnalysisRequest):
    """
    Run model inference on only those dataset tiles that intersect a user AOI bbox.
    """
    _validate_dataset_structure(request.dataset_path)
    _validate_aoi_bounds(request.west, request.south, request.east, request.north)

    result = analyze_dataset_with_aoi(
        dataset_path=request.dataset_path,
        aoi={
            "west": request.west,
            "south": request.south,
            "east": request.east,
            "north": request.north,
        },
        species_threshold=request.threshold,
        max_tiles=request.max_tiles,
        include_per_tile=request.include_per_tile,
    )

    return {
        "status": "success",
        "aoi_analysis": result,
    }


@router.post("/validate-aoi-fia")
async def validate_aoi_fia(request: AOIFIAValidationRequest):
    """
    Cross-check AOI model outputs against FIA-like plot data within the same AOI.
    """
    _validate_dataset_structure(request.dataset_path)
    _validate_aoi_bounds(request.west, request.south, request.east, request.north)

    if request.year_start is not None and request.year_end is not None:
        if request.year_start > request.year_end:
            raise HTTPException(status_code=400, detail="year_start must be <= year_end.")

    model_result = analyze_dataset_with_aoi(
        dataset_path=request.dataset_path,
        aoi={
            "west": request.west,
            "south": request.south,
            "east": request.east,
            "north": request.north,
        },
        species_threshold=request.threshold,
        max_tiles=request.max_tiles,
        include_per_tile=False,
    )

    try:
        fia_loaded = load_fia_csv(request.fia_csv_path)
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid FIA CSV: {e}")

    fia_records = filter_fia_records(
        fia_loaded["records"],
        west=request.west,
        south=request.south,
        east=request.east,
        north=request.north,
        year_start=request.year_start,
        year_end=request.year_end,
    )

    fia_summary = summarize_fia(fia_records)
    comparison = compare_model_to_fia(model_result.get("summary", {}), fia_summary)

    return {
        "status": "success",
        "aoi": model_result.get("aoi", {}),
        "model_summary": model_result.get("summary", {}),
        "model_tile_coverage": {
            "tiles_scanned": model_result.get("tiles_scanned", 0),
            "tiles_intersecting": model_result.get("tiles_intersecting", 0),
            "scan_errors": model_result.get("scan_errors", 0),
        },
        "fia": {
            "source_csv": fia_loaded["source"],
            "rows_loaded": fia_loaded["rows_loaded"],
            "rows_skipped": fia_loaded["rows_skipped"],
            "rows_in_aoi_after_filters": len(fia_records),
            "year_filter": {
                "start": request.year_start,
                "end": request.year_end,
            },
            "summary": fia_summary,
        },
        "comparison": comparison,
    }


@router.get("/fia-schema")
async def fia_schema():
    """
    Return accepted FIA CSV column aliases for AOI cross-check ingestion.
    """
    return {
        "required_logical_fields": {
            "latitude": ["lat", "latitude", "plot_lat", "plot_latitude", "plt_lat"],
            "longitude": ["lon", "lng", "longitude", "plot_lon", "plot_longitude", "plt_lon"],
        },
        "optional_fields": {
            "inventory_year": ["year", "inventory_year", "measurement_year", "inv_year"],
            "species": ["species", "species_code", "species_group", "forest_type", "spcd"],
            "trees_per_hectare": ["trees_per_hectare", "trees_ha", "tph", "tpha"],
            "trees_per_acre": ["trees_per_acre", "tpa"],
        },
        "notes": [
            "At least one latitude alias and one longitude alias are required.",
            "If trees_per_hectare is absent but trees_per_acre exists, conversion uses 1 acre = 0.404686 ha.",
        ],
    }


@router.post("/evaluate-offline")
async def evaluate_offline_endpoint(request: OfflineEvaluationRequest):
    """
    Run offline evaluation against a ground-truth CSV and export report artifacts.
    """
    _validate_dataset_structure(request.dataset_path)

    if not os.path.isfile(request.ground_truth_csv):
        raise HTTPException(status_code=400, detail=f"Ground-truth CSV not found: {request.ground_truth_csv}")

    if request.threshold_grid:
        invalid = [v for v in request.threshold_grid if v < 0.0 or v > 1.0]
        if invalid:
            raise HTTPException(status_code=400, detail="All threshold_grid values must be in [0, 1].")

    try:
        result = evaluate_offline(
            dataset_path=request.dataset_path,
            ground_truth_csv=request.ground_truth_csv,
            species_threshold=request.species_threshold,
            threshold_grid=request.threshold_grid,
            output_dir=request.output_dir,
        )
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Offline evaluation failed: {e}")

    return result


@router.post("/validate-location")
async def validate_location(request: LocationValidationRequest):
    """
    Lat/lon-first workflow:
    1) Convert center+radius to AOI bbox
    2) Run model inference on intersecting dataset tiles
    3) Validate against selected external source (FIA or ESA WorldCover)
    """
    _validate_lat_lon(request.lat, request.lon)

    if request.radius_km <= 0:
        raise HTTPException(status_code=400, detail="radius_km must be > 0")
    if request.sample_grid_size < 1:
        raise HTTPException(status_code=400, detail="sample_grid_size must be >= 1")

    if request.year_start is not None and request.year_end is not None:
        if request.year_start > request.year_end:
            raise HTTPException(status_code=400, detail="year_start must be <= year_end.")

    source = request.validation_source.strip().lower()
    if source not in {"fia", "esa_worldcover"}:
        raise HTTPException(status_code=400, detail="validation_source must be 'fia' or 'esa_worldcover'.")

    aoi = _bbox_from_center_radius(request.lat, request.lon, request.radius_km)
    _validate_aoi_bounds(aoi["west"], aoi["south"], aoi["east"], aoi["north"])

    model_result = None
    model_remote = None
    model_remote_grid = None
    if request.dataset_path:
        _validate_dataset_structure(request.dataset_path)
        model_result = analyze_dataset_with_aoi(
            dataset_path=request.dataset_path,
            aoi=aoi,
            species_threshold=request.threshold,
            max_tiles=request.max_tiles,
            include_per_tile=False,
        )
    else:
        provider = request.provider.strip().lower()
        if provider != "planetary_computer":
            raise HTTPException(
                status_code=400,
                detail="Unsupported provider. Currently only 'planetary_computer' is supported.",
            )
        if not request.start_date or not request.end_date:
            raise HTTPException(
                status_code=400,
                detail="start_date and end_date are required for remote fetch when dataset_path is not provided.",
            )
        try:
            if request.sample_grid_size > 1:
                model_remote_grid = run_remote_inference_planetary_computer_grid(
                    lat=request.lat,
                    lon=request.lon,
                    start_date=request.start_date,
                    end_date=request.end_date,
                    species_threshold=request.threshold,
                    radius_km=request.radius_km,
                    cloud_cover_max=request.cloud_cover_max,
                    grid_size=request.sample_grid_size,
                )
            else:
                model_remote = run_remote_inference_planetary_computer(
                    lat=request.lat,
                    lon=request.lon,
                    start_date=request.start_date,
                    end_date=request.end_date,
                    species_threshold=request.threshold,
                    radius_km=request.radius_km,
                    cloud_cover_max=request.cloud_cover_max,
                )
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Remote inference failed: {e}")

    model_summary = (
        model_result.get("summary", {})
        if model_result
        else model_remote_grid.get("summary", {}) if model_remote_grid
        else _model_summary_from_single_result(model_remote) if model_remote else {}
    )

    calibration = None
    calibrated_model_tph = None
    slope = request.calibration_slope
    intercept = request.calibration_intercept
    calibration_source = None
    if request.calibration_profile_path:
        try:
            profile = load_calibration_profile(
                os.path.abspath(os.path.expanduser(request.calibration_profile_path))
            )
            slope, intercept, calibration_source = pick_calibration_from_profile(
                profile,
                request.calibration_region,
            )
        except (ValueError, FileNotFoundError) as e:
            raise HTTPException(status_code=400, detail=f"Calibration profile error: {e}")

    if slope is not None and intercept is not None:
        model_tph = model_summary.get("mean_trees_per_hectare")
        if model_tph is not None:
            calibrated_model_tph = apply_linear_tph_calibration(
                model_tph,
                slope,
                intercept,
            )
            calibration = {
                "method": "linear",
                "slope": slope,
                "intercept": intercept,
                "model_tph_raw": model_tph,
                "model_tph_calibrated": round(calibrated_model_tph, 6),
                "source": calibration_source or "request_fields",
            }

    if source == "fia":
        if not request.fia_csv_path:
            raise HTTPException(status_code=400, detail="fia_csv_path is required when validation_source='fia'.")
        try:
            fia_loaded = load_fia_csv(request.fia_csv_path)
        except FileNotFoundError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Invalid FIA CSV: {e}")

        fia_records = filter_fia_records(
            fia_loaded["records"],
            west=aoi["west"],
            south=aoi["south"],
            east=aoi["east"],
            north=aoi["north"],
            year_start=request.year_start,
            year_end=request.year_end,
        )
        ext_summary = summarize_fia(fia_records)
        comparison = (
            compare_model_to_fia(model_summary, ext_summary)
            if model_result or model_remote or model_remote_grid
            else {
                "density_agreement": {
                    "model_mean_trees_per_hectare": None,
                    "fia_mean_trees_per_hectare": ext_summary.get("trees_per_hectare", {}).get("mean"),
                },
                "species_overlap": {
                    "overlap_count": 0,
                    "model_species_count": 0,
                    "fia_species_count": len(ext_summary.get("dominant_species", {})),
                    "overlap_details": [],
                },
                "notes": [
                    "Model inference not run because dataset_path was not provided.",
                    "FIA summary was computed from lat/lon AOI only.",
                ],
            }
        )
        if calibrated_model_tph is not None:
            fia_tph = ext_summary.get("trees_per_hectare", {}).get("mean")
            cal_block = {
                "model_tph_calibrated": round(calibrated_model_tph, 6),
                "fia_mean_trees_per_hectare": fia_tph,
            }
            if fia_tph is not None:
                abs_diff = calibrated_model_tph - float(fia_tph)
                pct = (abs_diff / float(fia_tph) * 100.0) if float(fia_tph) != 0 else None
                cal_block["absolute_difference"] = round(abs_diff, 6)
                cal_block["percent_difference"] = round(pct, 6) if pct is not None else None
            comparison["density_agreement_calibrated"] = cal_block

        return {
            "status": "success",
            "input": {
                "lat": request.lat,
                "lon": request.lon,
                "radius_km": request.radius_km,
                "validation_source": "fia",
                "provider": request.provider,
            },
            "aoi": (
                model_result.get("aoi", {})
                if model_result
                else aoi if model_remote_grid
                else model_remote.get("aoi", {}) if model_remote else aoi
            ),
            "calibration": calibration,
            "model": (
                {
                    "status": "ran",
                    "summary": model_result.get("summary", {}),
                    "tile_coverage": {
                        "tiles_scanned": model_result.get("tiles_scanned", 0),
                        "tiles_intersecting": model_result.get("tiles_intersecting", 0),
                        "scan_errors": model_result.get("scan_errors", 0),
                    },
                }
                if model_result
                else {
                    "status": "ran",
                    "mode": "remote_grid",
                    "provider": request.provider,
                    "summary": model_summary,
                    "tile_coverage": {
                        "tiles_scanned": model_remote_grid.get("samples_requested", 0) if model_remote_grid else 0,
                        "tiles_intersecting": model_remote_grid.get("samples_succeeded", 0) if model_remote_grid else 0,
                        "scan_errors": model_remote_grid.get("samples_failed", 0) if model_remote_grid else 0,
                    },
                    "remote_grid": model_remote_grid if model_remote_grid else {},
                }
                if model_remote_grid
                else {
                    "status": "ran",
                    "mode": "remote",
                    "provider": request.provider,
                    "summary": model_summary,
                    "tile_coverage": {
                        "tiles_scanned": 1,
                        "tiles_intersecting": 1,
                        "scan_errors": 0,
                    },
                    "remote_source": model_remote.get("remote_source", {}) if model_remote else {},
                    "prediction": model_remote if model_remote else {},
                }
                if model_remote
                else {
                    "status": "not_run",
                    "reason": "dataset_path not provided",
                    "summary": {},
                    "tile_coverage": {
                        "tiles_scanned": 0,
                        "tiles_intersecting": 0,
                        "scan_errors": 0,
                    },
                }
            ),
            "external_summary": {
                "source_csv": fia_loaded["source"],
                "rows_loaded": fia_loaded["rows_loaded"],
                "rows_skipped": fia_loaded["rows_skipped"],
                "rows_in_aoi_after_filters": len(fia_records),
                "year_filter": {"start": request.year_start, "end": request.year_end},
                "summary": ext_summary,
            },
            "comparison": comparison,
        }

    if not request.worldcover_path:
        raise HTTPException(
            status_code=400,
            detail="worldcover_path is required when validation_source='esa_worldcover'.",
        )

    try:
        wc_summary = summarize_worldcover_aoi(
            request.worldcover_path,
            west=aoi["west"],
            south=aoi["south"],
            east=aoi["east"],
            north=aoi["north"],
        )
    except FileNotFoundError:
        raise HTTPException(status_code=400, detail=f"WorldCover raster not found: {request.worldcover_path}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read WorldCover raster: {e}")

    comparison = (
        compare_model_to_worldcover(model_summary, wc_summary)
        if model_result or model_remote or model_remote_grid
        else {
            "density_vs_treecover": {
                "model_mean_density": None,
                "worldcover_tree_fraction": wc_summary.get("tree_cover_fraction"),
            },
            "notes": [
                "Model inference not run because dataset_path was not provided.",
                "WorldCover summary was computed from lat/lon AOI only.",
            ],
        }
    )
    if calibrated_model_tph is not None:
        comparison["density_vs_treecover_calibrated_tph"] = {
            "model_tph_calibrated": round(calibrated_model_tph, 6),
            "worldcover_tree_fraction": wc_summary.get("tree_cover_fraction"),
            "note": "WorldCover is fraction-based; calibrated TPH is density-based, so compare directionally.",
        }

    return {
        "status": "success",
        "input": {
            "lat": request.lat,
            "lon": request.lon,
            "radius_km": request.radius_km,
            "validation_source": "esa_worldcover",
            "provider": request.provider,
        },
        "aoi": (
            model_result.get("aoi", {})
            if model_result
            else aoi if model_remote_grid
            else model_remote.get("aoi", {}) if model_remote else aoi
        ),
        "calibration": calibration,
        "model": (
            {
                "status": "ran",
                "summary": model_result.get("summary", {}),
                "tile_coverage": {
                    "tiles_scanned": model_result.get("tiles_scanned", 0),
                    "tiles_intersecting": model_result.get("tiles_intersecting", 0),
                    "scan_errors": model_result.get("scan_errors", 0),
                },
            }
            if model_result
            else {
                "status": "ran",
                "mode": "remote_grid",
                "provider": request.provider,
                "summary": model_summary,
                "tile_coverage": {
                    "tiles_scanned": model_remote_grid.get("samples_requested", 0) if model_remote_grid else 0,
                    "tiles_intersecting": model_remote_grid.get("samples_succeeded", 0) if model_remote_grid else 0,
                    "scan_errors": model_remote_grid.get("samples_failed", 0) if model_remote_grid else 0,
                },
                "remote_grid": model_remote_grid if model_remote_grid else {},
            }
            if model_remote_grid
            else {
                "status": "ran",
                "mode": "remote",
                "provider": request.provider,
                "summary": model_summary,
                "tile_coverage": {
                    "tiles_scanned": 1,
                    "tiles_intersecting": 1,
                    "scan_errors": 0,
                },
                "remote_source": model_remote.get("remote_source", {}) if model_remote else {},
                "prediction": model_remote if model_remote else {},
            }
            if model_remote
            else {
                "status": "not_run",
                "reason": "dataset_path not provided",
                "summary": {},
                "tile_coverage": {
                    "tiles_scanned": 0,
                    "tiles_intersecting": 0,
                    "scan_errors": 0,
                },
            }
        ),
        "external_summary": {
            "worldcover_path": request.worldcover_path,
            "summary": wc_summary,
        },
        "comparison": comparison,
    }


@router.post("/convert-fia-datamart")
async def convert_fia_datamart(request: FIADatamartConvertRequest):
    """
    Convert FIA DataMart ZIP/folder (expects PLOT.csv and optional TREE.csv)
    into the simplified FIA CSV schema used by /validate-location.
    """
    source_path = os.path.abspath(os.path.expanduser(request.source_path))
    if not os.path.exists(source_path):
        raise HTTPException(status_code=400, detail=f"Source path not found: {source_path}")

    output_csv = request.output_csv_path
    if not output_csv:
        base_dir = (
            source_path
            if os.path.isdir(source_path)
            else os.path.dirname(source_path) or "."
        )
        output_csv = os.path.join(base_dir, "fia_converted.csv")

    output_csv = os.path.abspath(os.path.expanduser(output_csv))

    try:
        result = build_fia_csv_from_datamart(source_path, output_csv)
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"FIA conversion failed: {e}")

    return result


@router.post("/fit-fia-calibration")
async def fit_fia_calibration(request: FIACalibrationFitRequest):
    """
    Fit linear calibration from historical AOI runs:
    fia_tph = slope * model_tph + intercept
    """
    csv_path = os.path.abspath(os.path.expanduser(request.calibration_csv_path))
    if not os.path.isfile(csv_path):
        raise HTTPException(status_code=400, detail=f"Calibration CSV not found: {csv_path}")

    try:
        samples = load_calibration_samples_csv(csv_path)
        fit = fit_linear_tph_calibration(samples)
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Calibration fit failed: {e}")

    return {
        "status": "success",
        "calibration_csv_path": csv_path,
        "fit": fit,
    }


@router.post("/fit-fia-calibration-regional")
async def fit_fia_calibration_regional(request: FiaRegionalCalibrationFitRequest):
    """
    Fit region-aware linear calibration profile.
    CSV must include: model_tph, fia_tph, and region column.
    """
    csv_path = os.path.abspath(os.path.expanduser(request.calibration_csv_path))
    if not os.path.isfile(csv_path):
        raise HTTPException(status_code=400, detail=f"Calibration CSV not found: {csv_path}")

    try:
        regional_samples = load_regional_calibration_samples_csv(
            csv_path,
            region_column=request.region_column,
        )
        profile = fit_regional_linear_tph_calibration(
            regional_samples,
            min_samples_per_region=request.min_samples_per_region,
        )
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Regional calibration fit failed: {e}")

    saved_path = None
    if request.output_profile_path:
        saved_path = os.path.abspath(os.path.expanduser(request.output_profile_path))
        try:
            save_calibration_profile(profile, saved_path)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to save calibration profile: {e}")

    return {
        "status": "success",
        "calibration_csv_path": csv_path,
        "region_column": request.region_column,
        "profile_saved_path": saved_path,
        "profile": profile,
    }


@router.post("/detect-crowns")
async def detect_crowns(
    file: UploadFile = File(...),
    ndvi_threshold: float = Query(0.45, ge=-1.0, le=1.0),
    min_area_px: int = Query(12, ge=1),
    model_tree_count: Optional[float] = Query(None, description="Optional target tree count from model to align object detection results."),
):
    """
    Object-level tree crown candidate detection from an uploaded GeoTIFF.
    Uses NDVI peak finding + watershed segmentation, optionally aligned with model tree count.
    """
    content = await file.read()
    is_valid, message = validate_file(file.filename, len(content))
    if not is_valid:
        raise HTTPException(status_code=400, detail=message)

    save_path = save_upload(content, file.filename)
    try:
        result = detect_tree_crowns_advanced(
            tif_path=save_path,
            ndvi_threshold=ndvi_threshold,
            min_area_px=min_area_px,
            model_tree_count=model_tree_count,
        )
        result["filename"] = file.filename
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Crown detection failed: {e}")
    finally:
        if os.path.exists(save_path):
            try:
                os.remove(save_path)
            except OSError:
                pass


@router.post("/fetch-remote-geotiff")
async def fetch_remote_geotiff(request: RemoteGeoTiffFetchRequest, background_tasks: BackgroundTasks):
    """
    Fetch a remote Sentinel patch by lat/lon+date and return it as a 15-band GeoTIFF.
    """
    _validate_lat_lon(request.lat, request.lon)

    if request.provider.strip().lower() != "planetary_computer":
        raise HTTPException(
            status_code=400,
            detail="Unsupported provider. Currently only 'planetary_computer' is supported.",
        )
    if request.radius_km <= 0:
        raise HTTPException(status_code=400, detail="radius_km must be > 0.")

    try:
        tensor, metadata = fetch_remote_tensor_planetary_computer(
            lat=request.lat,
            lon=request.lon,
            start_date=request.start_date,
            end_date=request.end_date,
            radius_km=request.radius_km,
            cloud_cover_max=request.cloud_cover_max,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Remote fetch failed: {e}")

    arr = tensor.squeeze(0).cpu().numpy().astype(np.float32)
    if arr.ndim != 3 or arr.shape[0] < 2:
        raise HTTPException(status_code=500, detail="Fetched tensor has invalid shape for GeoTIFF export.")

    aoi = metadata.get("aoi", {})
    west = aoi.get("west")
    south = aoi.get("south")
    east = aoi.get("east")
    north = aoi.get("north")
    if west is None or south is None or east is None or north is None:
        raise HTTPException(status_code=500, detail="Missing AOI bounds in remote metadata.")

    out_dir = os.path.join(UPLOAD_DIR, "remote_exports")
    os.makedirs(out_dir, exist_ok=True)
    basename = (
        f"remote_{request.lat:.5f}_{request.lon:.5f}_{request.start_date}_{request.end_date}.tif"
        .replace(":", "_")
        .replace("/", "-")
        .replace(" ", "_")
    )
    out_path = os.path.join(out_dir, basename)

    height, width = int(arr.shape[1]), int(arr.shape[2])
    transform = transform_from_bounds(float(west), float(south), float(east), float(north), width, height)
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": int(arr.shape[0]),
        "dtype": "float32",
        "crs": "EPSG:4326",
        "transform": transform,
        "compress": "lzw",
    }

    try:
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(arr)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to write GeoTIFF: {e}")

    background_tasks.add_task(os.remove, out_path)
    return FileResponse(
        path=out_path,
        media_type="image/tiff",
        filename=basename,
    )


# ─── Single File Endpoints ──────────────────────────────────────────────

@router.post("/upload")
async def upload_tif(file: UploadFile = File(...)):
    """
    Upload a .tif file for analysis.
    Validates file type and size, saves to temp storage.
    """
    content = await file.read()
    is_valid, message = validate_file(file.filename, len(content))

    if not is_valid:
        raise HTTPException(status_code=400, detail=message)

    save_path = save_upload(content, file.filename)

    try:
        metadata = get_tif_metadata(save_path)
    except Exception as e:
        metadata = {"error": str(e)}

    return {
        "status": "success",
        "file_path": save_path,
        "filename": file.filename,
        "file_size": len(content),
        "metadata": metadata
    }


@router.post("/predict")
async def predict(
    file: UploadFile = File(...),
    threshold: float = Query(0.5, ge=0.0, le=1.0, description="Species detection threshold")
):
    """
    Run inference on an uploaded .tif file.
    Returns tree count, density, species distribution, and biodiversity metrics.
    """
    content = await file.read()
    is_valid, message = validate_file(file.filename, len(content))

    if not is_valid:
        raise HTTPException(status_code=400, detail=message)

    save_path = save_upload(content, file.filename)

    try:
        results = run_inference(save_path, species_threshold=threshold)

        metadata = get_tif_metadata(save_path)
        results["metadata"] = metadata
        results["filename"] = file.filename

        # Store for batch analysis
        batch_results.append(results)

        return results

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")
    finally:
        if os.path.exists(save_path):
            try:
                os.remove(save_path)
            except OSError:
                pass


@router.post("/preview")
async def preview(
    file: UploadFile = File(...),
    mode: str = Query("rgb", description="Preview mode: 'rgb' or 'ndvi'")
):
    """
    Generate a PNG preview from a .tif file.
    Supports RGB true-color and NDVI visualization modes.
    """
    content = await file.read()
    is_valid, message = validate_file(file.filename, len(content))

    if not is_valid:
        raise HTTPException(status_code=400, detail=message)

    save_path = save_upload(content, file.filename)

    try:
        preview_path = tif_to_png_preview(save_path, mode=mode)
        return FileResponse(
            preview_path,
            media_type="image/png",
            filename=f"preview_{mode}.png"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Preview generation failed: {str(e)}")
    finally:
        if os.path.exists(save_path):
            try:
                os.remove(save_path)
            except OSError:
                pass


# ─── Dataset Analysis Endpoints ──────────────────────────────────────────

def _run_analysis(dataset_path: str, threshold: float, batch_size: int):
    """Background task to run batch analysis."""
    global analysis_state

    def progress_cb(processed, total):
        analysis_state["progress"] = processed
        analysis_state["total"] = total

    try:
        analysis_state["status"] = "running"
        analysis_state["progress"] = 0
        analysis_state["error"] = None

        # Discover dataset
        pairs = discover_dataset(dataset_path)
        analysis_state["total"] = len(pairs)

        # Run batch inference
        results = batch_inference(
            dataset_path,
            species_threshold=threshold,
            batch_size=batch_size,
            progress_callback=progress_cb
        )

        # Save to CSV
        output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")
        os.makedirs(output_dir, exist_ok=True)
        csv_path = os.path.join(output_dir, "predictions.csv")
        save_results_csv(results, csv_path)

        # Compute aggregated analytics
        densities = [r["density"] for r in results]
        distributions = [r["species_distribution"] for r in results]

        from ..metrics import (
            compute_density_statistics,
            compute_species_summary,
            compute_batch_biodiversity,
        )

        # Dominant species distribution (count how many patches each species dominates)
        dominant_counts = {}
        for r in results:
            sp = r["dominant_species"]
            dominant_counts[sp] = dominant_counts.get(sp, 0) + 1

        avg_tree_count = sum(r["tree_count"] for r in results) / len(results) if results else 0

        # Average biodiversity
        avg_shannon = sum(r["biodiversity_metrics"]["shannon_index"] for r in results) / len(results) if results else 0

        analysis_state["results"] = {
            "total_images": len(results),
            "avg_tree_count": round(avg_tree_count, 1),
            "dominant_species_distribution": dominant_counts,
            "biodiversity_index": round(avg_shannon, 4),
            "density_statistics": compute_density_statistics(densities),
            "species_summary": compute_species_summary(distributions),
            "biodiversity_aggregate": compute_batch_biodiversity(results),
            "csv_path": csv_path,
            "per_file_results": results,
        }

        # Also store in batch_results for batch-stats endpoint
        batch_results.clear()
        batch_results.extend(results)

        analysis_state["status"] = "completed"
        analysis_state["progress"] = len(results)

    except Exception as e:
        analysis_state["status"] = "error"
        analysis_state["error"] = str(e)
        print(f"[AnalysisError] {e}")


@router.post("/analyze-dataset")
async def analyze_dataset(
    request: DatasetAnalysisRequest,
    background_tasks: BackgroundTasks
):
    """
    Analyze an entire dataset of paired S1/S2 .tif files.

    Input: dataset_path, threshold
    Process: discovers pairs, runs batch inference, computes aggregated analytics
    """
    if analysis_state["status"] == "running":
        raise HTTPException(
            status_code=409,
            detail="An analysis is already running. Check /api/analysis-status for progress."
        )

    _validate_dataset_structure(request.dataset_path)

    # Start in background thread
    import threading
    thread = threading.Thread(
        target=_run_analysis,
        args=(request.dataset_path, request.threshold, request.batch_size),
        daemon=True
    )
    thread.start()

    return {
        "status": "started",
        "message": "Dataset analysis started. Poll /api/analysis-status for progress.",
        "dataset_path": request.dataset_path
    }


@router.get("/analysis-status")
async def analysis_status():
    """Check the status and progress of a running dataset analysis."""
    response = {
        "status": analysis_state["status"],
        "progress": analysis_state["progress"],
        "total": analysis_state["total"],
    }

    if analysis_state["status"] == "completed" and analysis_state["results"]:
        # Return aggregated results (without per-file to keep response light)
        results = analysis_state["results"]
        response["results"] = {
            "total_images": results["total_images"],
            "avg_tree_count": results["avg_tree_count"],
            "dominant_species_distribution": results["dominant_species_distribution"],
            "biodiversity_index": results["biodiversity_index"],
            "density_statistics": results["density_statistics"],
            "species_summary": results["species_summary"],
            "biodiversity_aggregate": results["biodiversity_aggregate"],
            "csv_path": results.get("csv_path", ""),
        }

    if analysis_state["status"] == "error":
        response["error"] = analysis_state["error"]

    return response


# ─── Chart Data Endpoints ────────────────────────────────────────────────

@router.get("/species-distribution")
async def species_distribution():
    """
    Return species distribution chart data from the latest analysis.
    Returns species_counts: {species_name: count_of_patches_where_dominant}
    """
    if not analysis_state["results"] and not batch_results:
        return {
            "status": "no_data",
            "message": "Run an analysis first.",
            "species_counts": {}
        }

    # Use analysis results if available, otherwise compute from batch_results
    if analysis_state["results"]:
        return {
            "status": "success",
            "species_counts": analysis_state["results"]["dominant_species_distribution"],
            "species_summary": analysis_state["results"].get("species_summary", {}),
        }

    # Compute from batch_results
    dominant_counts = {}
    for r in batch_results:
        sp = r.get("dominant_species", "Unknown")
        dominant_counts[sp] = dominant_counts.get(sp, 0) + 1

    return {
        "status": "success",
        "species_counts": dominant_counts,
    }


@router.get("/density-map")
async def density_map():
    """
    Return density values for each file from the latest analysis.
    Returns: [{file, density}, ...]
    """
    if not analysis_state["results"] and not batch_results:
        return {
            "status": "no_data",
            "message": "Run an analysis first.",
            "data": []
        }

    source = (
        analysis_state["results"]["per_file_results"]
        if analysis_state["results"]
        else batch_results
    )

    data = [
        {
            "file": r.get("filename", f"patch_{i}"),
            "density": r["density"]
        }
        for i, r in enumerate(source)
    ]

    return {
        "status": "success",
        "total": len(data),
        "data": data
    }


@router.get("/biodiversity")
async def biodiversity_metrics():
    """
    Return aggregated biodiversity metrics from the latest analysis.
    """
    if not analysis_state["results"] and not batch_results:
        return {
            "status": "no_data",
            "message": "Run an analysis first."
        }

    if analysis_state["results"]:
        return {
            "status": "success",
            "biodiversity": analysis_state["results"]["biodiversity_aggregate"],
            "biodiversity_index": analysis_state["results"]["biodiversity_index"],
        }

    # Compute from batch_results
    from ..metrics import compute_batch_biodiversity
    return {
        "status": "success",
        "biodiversity": compute_batch_biodiversity(batch_results),
    }


# ─── Existing Endpoints ──────────────────────────────────────────────────

@router.get("/batch-stats")
async def batch_statistics():
    """
    Get aggregated statistics from all predictions made in this session.
    Useful for multi-patch analysis, heatmaps, and distribution plots.
    """
    if not batch_results:
        return {
            "status": "no_data",
            "message": "No predictions have been made yet. Upload and predict on .tif files first."
        }

    densities = [r["density"] for r in batch_results]
    distributions = [r["species_distribution"] for r in batch_results]

    return {
        "status": "success",
        "total_patches": len(batch_results),
        "density_statistics": compute_density_statistics(densities),
        "species_summary": compute_species_summary(distributions),
        "biodiversity_aggregate": compute_batch_biodiversity(batch_results),
        "heatmap_data": generate_heatmap_data(batch_results)
    }


@router.delete("/batch-clear")
async def clear_batch():
    """Clear all stored batch results and analysis state."""
    batch_results.clear()
    analysis_state["status"] = "idle"
    analysis_state["progress"] = 0
    analysis_state["total"] = 0
    analysis_state["results"] = None
    analysis_state["error"] = None
    return {"status": "cleared", "message": "Batch results and analysis state cleared."}


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "TreeSat Analytics API",
        "version": "1.0.0"
    }


@router.get("/species-labels")
async def species_labels():
    """Return the list of species labels used by the model."""
    from ..model_loader import SPECIES_LABELS
    return {"labels": SPECIES_LABELS, "count": len(SPECIES_LABELS)}
