"""
Remote EO fetch + inference utilities.

Current provider support:
- Planetary Computer STAC (Sentinel-2 L2A + Sentinel-1 RTC)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import rasterio
import torch
from rasterio.crs import CRS
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds

from .inference import _build_result, postprocess_density_output
from .model_loader import ModelSingleton


S2_BANDS = [
    "B01", "B02", "B03", "B04", "B05", "B06", "B07",
    "B08", "B8A", "B09", "B10", "B11", "B12",
]

S1_BANDS = ["vv", "vh"]


@dataclass
class SelectedItems:
    s2: Any
    s1: Any


def _bbox_from_center_radius(lat: float, lon: float, radius_km: float) -> Tuple[float, float, float, float]:
    import math

    lat_delta = radius_km / 111.32
    cos_lat = max(0.01, math.cos(math.radians(lat)))
    lon_delta = radius_km / (111.32 * cos_lat)

    west = max(-180.0, lon - lon_delta)
    east = min(180.0, lon + lon_delta)
    south = max(-90.0, lat - lat_delta)
    north = min(90.0, lat + lat_delta)
    return west, south, east, north


def _read_asset_window(
    asset_href: str, 
    west: float, 
    south: float, 
    east: float, 
    north: float,
    out_shape: Optional[Tuple[int, int]] = (64, 64)
) -> np.ndarray:
    with rasterio.open(asset_href) as src:
        if src.crs and src.crs != CRS.from_epsg(4326):
            left, bottom, right, top = transform_bounds(
                CRS.from_epsg(4326), src.crs, west, south, east, north
            )
        else:
            left, bottom, right, top = west, south, east, north

        win = from_bounds(left, bottom, right, top, src.transform)
        win = win.round_offsets().round_lengths()

        # Clamp the window to dataset extents.
        col_off = max(0, int(win.col_off))
        row_off = max(0, int(win.row_off))
        width = int(win.width)
        height = int(win.height)

        if col_off >= src.width or row_off >= src.height or width <= 0 or height <= 0:
            shape = out_shape or (64, 64)
            return np.zeros(shape, dtype=np.float32)

        width = min(width, src.width - col_off)
        height = min(height, src.height - row_off)

        # If out_shape is None, we read at native resolution for that window
        read_kwargs = {
            "window": ((row_off, row_off + height), (col_off, col_off + width)),
            "resampling": rasterio.enums.Resampling.bilinear,
        }
        if out_shape:
            read_kwargs["out_shape"] = out_shape

        arr = src.read(1, **read_kwargs).astype(np.float32)

        nodata = src.nodata
        if nodata is not None:
            arr = np.where(arr == nodata, 0.0, arr)

        arr[~np.isfinite(arr)] = 0.0
        return arr


def _pick_best_item(items: list, prefer_low_cloud: bool = False, reference_dt: Optional[datetime] = None):
    if not items:
        return None

    if prefer_low_cloud:
        def cloud_key(it):
            cc = it.properties.get("eo:cloud_cover")
            if cc is None:
                return 1e9
            return float(cc)
        return sorted(items, key=cloud_key)[0]

    if reference_dt is not None:
        def dt_distance(it):
            dt = it.datetime
            if dt is None:
                return timedelta(days=9999)
            dt_utc = dt.astimezone(timezone.utc) if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
            ref_utc = reference_dt.astimezone(timezone.utc) if reference_dt.tzinfo else reference_dt.replace(tzinfo=timezone.utc)
            return abs(dt_utc - ref_utc)
        return sorted(items, key=dt_distance)[0]

    return items[0]


def _search_planetary_computer_items(
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
    cloud_cover_max: float = 40.0,
) -> SelectedItems:
    try:
        from pystac_client import Client
        import planetary_computer
    except ImportError as e:
        raise RuntimeError(
            "Missing dependencies for remote fetch. Install: pystac-client planetary-computer"
        ) from e

    catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")

    west, south, east, north = _bbox_from_center_radius(lat, lon, radius_km=1.5)
    dt_range = f"{start_date}/{end_date}"

    s2_search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=[west, south, east, north],
        datetime=dt_range,
        query={"eo:cloud_cover": {"lt": cloud_cover_max}},
        limit=100,
    )
    s2_items = list(s2_search.items())
    s2 = _pick_best_item(s2_items, prefer_low_cloud=True)
    if s2 is None:
        raise ValueError("No Sentinel-2 L2A items found for the requested location/date range.")

    s2_dt = s2.datetime
    if s2_dt is None:
        s1_start = start_date
        s1_end = end_date
    else:
        ref = s2_dt.astimezone(timezone.utc) if s2_dt.tzinfo else s2_dt.replace(tzinfo=timezone.utc)
        s1_start = (ref - timedelta(days=10)).date().isoformat()
        s1_end = (ref + timedelta(days=10)).date().isoformat()

    s1_search = catalog.search(
        collections=["sentinel-1-rtc"],
        bbox=[west, south, east, north],
        datetime=f"{s1_start}/{s1_end}",
        limit=100,
    )
    s1_items = list(s1_search.items())
    s1 = _pick_best_item(s1_items, prefer_low_cloud=False, reference_dt=s2_dt)
    if s1 is None:
        raise ValueError("No Sentinel-1 RTC items found near the selected Sentinel-2 date.")

    # Sign assets for authorized access.
    s2 = planetary_computer.sign(s2)
    s1 = planetary_computer.sign(s1)
    return SelectedItems(s2=s2, s1=s1)


def _build_15ch_tensor_from_items(
    items: SelectedItems,
    west: float,
    south: float,
    east: float,
    north: float,
    out_shape: Optional[Tuple[int, int]] = (64, 64)
) -> torch.Tensor:
    s2_stack = []
    for band in S2_BANDS:
        asset = items.s2.assets.get(band)
        if asset is None or not asset.href:
            shape = out_shape or (64, 64)
            s2_stack.append(np.zeros(shape, dtype=np.float32))
            continue
        arr = _read_asset_window(asset.href, west, south, east, north, out_shape=out_shape)
        s2_stack.append(arr / 10000.0)

    s1_stack = []
    for band in S1_BANDS:
        asset = items.s1.assets.get(band)
        if asset is None or not asset.href:
            shape = out_shape or (64, 64)
            s1_stack.append(np.zeros(shape, dtype=np.float32))
            continue
        arr = _read_asset_window(asset.href, west, south, east, north, out_shape=out_shape)
        s1_stack.append(arr)

    stacked = np.stack(s2_stack + s1_stack, axis=0).astype(np.float32)
    tensor = torch.from_numpy(stacked).unsqueeze(0)
    return tensor


def run_remote_inference_planetary_computer(
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
    species_threshold: float = 0.5,
    radius_km: float = 0.1,
    cloud_cover_max: float = 40.0,
) -> Dict[str, Any]:
    """
    Fetch S1/S2 from Planetary Computer by location/date and run model inference.
    """
    # Always use a standard 200m patch (0.1km radius) for the model
    patch_radius = 0.1
    west, south, east, north = _bbox_from_center_radius(lat, lon, patch_radius)
    items = _search_planetary_computer_items(lat, lon, start_date, end_date, cloud_cover_max=cloud_cover_max)

    # For model inference, we MUST have 64x64 shape
    tensor = _build_15ch_tensor_from_items(items, west, south, east, north, out_shape=(64, 64))

    model, device = ModelSingleton.get_model()
    tensor = tensor.to(device)

    with torch.no_grad():
        density_out, species_out = model(tensor)

    density_raw = float(density_out.squeeze().cpu())
    density_mode = getattr(model, "density_mode", "normalized")
    density, trees_per_hectare = postprocess_density_output(density_raw, density_mode=density_mode)
    species_probs = species_out.squeeze().cpu().numpy()

    result = _build_result(
        density,
        species_probs,
        species_threshold,
        filename="remote_location",
        trees_per_hectare_override=trees_per_hectare,
    )
    result["remote_source"] = {
        "provider": "planetary_computer",
        "sentinel2_item_id": items.s2.id,
        "sentinel2_datetime": items.s2.datetime.isoformat() if items.s2.datetime else None,
        "sentinel1_item_id": items.s1.id,
        "sentinel1_datetime": items.s1.datetime.isoformat() if items.s1.datetime else None,
        "search_date_range": {"start": start_date, "end": end_date},
        "cloud_cover_max": cloud_cover_max,
    }
    result["model_patch_aoi"] = {
        "west": round(west, 6),
        "south": round(south, 6),
        "east": round(east, 6),
        "north": round(north, 6),
        "center_lat": round(lat, 6),
        "center_lon": round(lon, 6),
        "radius_km": patch_radius,
    }
    result["requested_radius_km"] = radius_km
    return result


def fetch_remote_tensor_planetary_computer(
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
    radius_km: float = 0.1,
    cloud_cover_max: float = 40.0,
    out_shape: Optional[Tuple[int, int]] = (64, 64)
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Fetch remote Sentinel-1/2 data and return a multi-channel tensor.
    Default shape is 64x64 for model inference, but can be customized (e.g. high-res for visualization).
    """
    west, south, east, north = _bbox_from_center_radius(lat, lon, radius_km)
    items = _search_planetary_computer_items(
        lat,
        lon,
        start_date,
        end_date,
        cloud_cover_max=cloud_cover_max,
    )
    tensor = _build_15ch_tensor_from_items(items, west, south, east, north, out_shape=out_shape)
    metadata = {
        "provider": "planetary_computer",
        "sentinel2_item_id": items.s2.id,
        "sentinel2_datetime": items.s2.datetime.isoformat() if items.s2.datetime else None,
        "sentinel1_item_id": items.s1.id,
        "sentinel1_datetime": items.s1.datetime.isoformat() if items.s1.datetime else None,
        "search_date_range": {"start": start_date, "end": end_date},
        "cloud_cover_max": cloud_cover_max,
        "aoi": {
            "west": round(west, 6),
            "south": round(south, 6),
            "east": round(east, 6),
            "north": round(north, 6),
            "center_lat": round(lat, 6),
            "center_lon": round(lon, 6),
            "radius_km": radius_km,
        },
    }
    return tensor, metadata


def _grid_points(lat: float, lon: float, radius_km: float, grid_size: int) -> List[Tuple[float, float]]:
    """
    Generate evenly spaced sample points in a square grid around center.
    grid_size=1 returns only center point.
    """
    import math

    if grid_size <= 1:
        return [(lat, lon)]

    lat_delta = radius_km / 111.32
    cos_lat = max(0.01, math.cos(math.radians(lat)))
    lon_delta = radius_km / (111.32 * cos_lat)

    lat_vals = np.linspace(lat - lat_delta, lat + lat_delta, grid_size)
    lon_vals = np.linspace(lon - lon_delta, lon + lon_delta, grid_size)
    pts: List[Tuple[float, float]] = []
    for la in lat_vals:
        for lo in lon_vals:
            pts.append((float(la), float(lo)))
    return pts


def run_remote_inference_planetary_computer_grid(
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
    species_threshold: float = 0.5,
    radius_km: float = 10.0,
    cloud_cover_max: float = 40.0,
    grid_size: int = 5,
) -> Dict[str, Any]:
    """
    Multi-sample AOI inference using an NxN grid around center.
    Returns aggregated summary plus per-point predictions.
    """
    if grid_size < 1:
        raise ValueError("grid_size must be >= 1")

    pts = _grid_points(lat, lon, radius_km=radius_km, grid_size=grid_size)
    model, device = ModelSingleton.get_model()

    per_point: List[Dict[str, Any]] = []
    failures = 0

    for idx, (plat, plon) in enumerate(pts):
        try:
            tensor, meta = fetch_remote_tensor_planetary_computer(
                lat=plat,
                lon=plon,
                start_date=start_date,
                end_date=end_date,
                radius_km=0.1,  # Each sample is a standard 200m patch
                cloud_cover_max=cloud_cover_max,
            )
            tensor = tensor.to(device)
            with torch.no_grad():
                density_out, species_out = model(tensor)

            density_raw = float(density_out.squeeze().cpu())
            density_mode = getattr(model, "density_mode", "normalized")
            density, trees_per_hectare = postprocess_density_output(density_raw, density_mode=density_mode)
            species_probs = species_out.squeeze().cpu().numpy()
            result = _build_result(
                density,
                species_probs,
                species_threshold,
                filename=f"grid_{idx:03d}",
                trees_per_hectare_override=trees_per_hectare,
            )
            result["remote_source"] = meta
            per_point.append(result)
        except Exception:
            failures += 1

    if not per_point:
        raise ValueError("No successful remote samples in AOI grid")

    densities = [r["density"] for r in per_point]
    tphs = [r["trees_per_hectare"] for r in per_point]
    tree_counts = [r["tree_count"] for r in per_point]

    dominant_counts: Dict[str, int] = {}
    for r in per_point:
        sp = r.get("dominant_species", "Unknown")
        dominant_counts[sp] = dominant_counts.get(sp, 0) + 1

    summary = {
        "total_tiles_analyzed": len(per_point),
        "mean_density": float(np.mean(densities)),
        "avg_tree_count": float(np.mean(tree_counts)),
        "mean_trees_per_hectare": float(np.mean(tphs)),
        "dominant_species_distribution": dominant_counts,
        "density_min": float(np.min(densities)),
        "density_max": float(np.max(densities)),
        "density_std": float(np.std(densities)),
    }

    return {
        "mode": "remote_grid",
        "grid_size": grid_size,
        "samples_requested": len(pts),
        "samples_succeeded": len(per_point),
        "samples_failed": failures,
        "summary": summary,
        "per_point_predictions": per_point,
    }
