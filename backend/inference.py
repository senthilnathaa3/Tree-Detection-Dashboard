"""
Inference engine for TreeSat model.
Handles tensor preprocessing, model forward pass, batch inference, and result extraction.
"""

import os
import csv
import torch
import numpy as np
import rasterio
from typing import Dict, Any, Optional, List, Tuple
from .model_loader import ModelSingleton, SPECIES_LABELS, INPUT_CHANNELS


MAX_TPH_REFERENCE = float(os.getenv("MODEL_MAX_TPH_REFERENCE", "1000"))


def postprocess_density_output(
    density_raw: float,
    density_mode: str = "normalized",
    max_tph_reference: float = MAX_TPH_REFERENCE,
) -> Tuple[float, float]:
    """
    Convert density head output into:
    - normalized density (0..1-like, for backward-compatible display)
    - trees_per_hectare (TPH)
    """
    mode = (density_mode or "normalized").strip().lower()
    if mode == "tph":
        tph = max(0.0, float(density_raw))
        density = float(np.clip(tph / max(1e-6, max_tph_reference), 0.0, 1.0))
        return density, tph

    density = float(np.clip(float(density_raw), 0.0, 1.0))
    tph = density * max_tph_reference
    return density, tph


def preprocess_tif(file_path: str, target_size: int = 64) -> torch.Tensor:
    """
    Read a single .tif file (15-band stacked) and preprocess into a model-ready tensor.

    Args:
        file_path: Path to the .tif file
        target_size: Target spatial dimension for the model input

    Returns:
        Tensor of shape (1, 15, target_size, target_size)
    """
    with rasterio.open(file_path) as src:
        data = src.read()  # shape: (bands, H, W)

    # Ensure we have exactly 15 channels
    num_bands = data.shape[0]

    if num_bands < INPUT_CHANNELS:
        # Pad with zeros if fewer bands
        padding = np.zeros(
            (INPUT_CHANNELS - num_bands, data.shape[1], data.shape[2]),
            dtype=data.dtype
        )
        data = np.concatenate([data, padding], axis=0)
    elif num_bands > INPUT_CHANNELS:
        # Truncate if more bands
        data = data[:INPUT_CHANNELS]

    # Convert to float32 and normalize
    tensor = torch.from_numpy(data.astype(np.float32))

    # Normalize each channel independently (min-max to 0-1)
    for i in range(tensor.shape[0]):
        channel = tensor[i]
        c_min, c_max = channel.min(), channel.max()
        if c_max - c_min > 0:
            tensor[i] = (channel - c_min) / (c_max - c_min)
        else:
            tensor[i] = torch.zeros_like(channel)

    # Resize to target size using interpolation
    tensor = tensor.unsqueeze(0)  # (1, C, H, W)
    tensor = torch.nn.functional.interpolate(
        tensor, size=(target_size, target_size), mode="bilinear", align_corners=False
    )

    return tensor


def preprocess_paired_s1_s2(
    s2_path: str, s1_path: str, target_size: int = 64
) -> torch.Tensor:
    """
    Load and preprocess paired Sentinel-2 and Sentinel-1 images.

    S2 → 13 bands, normalized by /10000
    S1 → 2 bands
    Stacked → 15 channels total

    Args:
        s2_path: Path to Sentinel-2 .tif file (13 bands)
        s1_path: Path to Sentinel-1 .tif file (2 bands)
        target_size: Target spatial dimension

    Returns:
        Tensor of shape (1, 15, target_size, target_size)
    """
    # Load S2 (13 bands)
    with rasterio.open(s2_path) as src:
        s2_data = src.read().astype(np.float32)  # (bands, H, W)

    # Load S1 (2 bands)
    with rasterio.open(s1_path) as src:
        s1_data = src.read().astype(np.float32)  # (bands, H, W)

    # Normalize S2
    s2_data = s2_data / 10000.0

    # Ensure correct band counts
    if s2_data.shape[0] < 13:
        pad = np.zeros((13 - s2_data.shape[0], s2_data.shape[1], s2_data.shape[2]), dtype=np.float32)
        s2_data = np.concatenate([s2_data, pad], axis=0)
    elif s2_data.shape[0] > 13:
        s2_data = s2_data[:13]

    if s1_data.shape[0] < 2:
        pad = np.zeros((2 - s1_data.shape[0], s1_data.shape[1], s1_data.shape[2]), dtype=np.float32)
        s1_data = np.concatenate([s1_data, pad], axis=0)
    elif s1_data.shape[0] > 2:
        s1_data = s1_data[:2]

    # Stack: [S2(13), S1(2)] → 15 channels
    stacked = np.concatenate([s2_data, s1_data], axis=0)  # (15, H, W)

    tensor = torch.from_numpy(stacked)
    tensor = tensor.unsqueeze(0)  # (1, 15, H, W)

    # Resize to target size
    tensor = torch.nn.functional.interpolate(
        tensor, size=(target_size, target_size), mode="bilinear", align_corners=False
    )

    return tensor


def run_inference(file_path: str, species_threshold: float = 0.5) -> Dict[str, Any]:
    """
    Run full inference pipeline on a single .tif file (15-band stacked).

    Args:
        file_path: Path to the uploaded .tif file
        species_threshold: Probability threshold for species detection

    Returns:
        Dictionary containing all predictions and computed metrics
    """
    model, device = ModelSingleton.get_model()
    tensor = preprocess_tif(file_path)
    tensor = tensor.to(device)

    with torch.no_grad():
        density_out, species_out = model(tensor)

    # Extract predictions
    density_raw = float(density_out.squeeze().cpu())
    density_mode = getattr(model, "density_mode", "normalized")
    density, trees_per_hectare = postprocess_density_output(density_raw, density_mode=density_mode)
    species_probs = species_out.squeeze().cpu().numpy()

    # Compute tree count metrics
    patch_area_hectares = (200 * 200) / 10000  # 4 hectares
    estimated_trees = trees_per_hectare * patch_area_hectares
    trees_per_sqkm = trees_per_hectare * 100  # 1 km² = 100 hectares

    # Species analysis
    species_distribution = []
    for i, (name, prob) in enumerate(zip(SPECIES_LABELS, species_probs)):
        species_distribution.append({
            "species": name,
            "probability": float(prob),
            "detected": bool(prob >= species_threshold)
        })

    # Sort by probability descending
    species_distribution.sort(key=lambda x: x["probability"], reverse=True)

    # Dominant and least species
    dominant_species = species_distribution[0]["species"] if species_distribution else "Unknown"
    least_species = species_distribution[-1]["species"] if species_distribution else "Unknown"

    # Detected species
    detected_species = [s for s in species_distribution if s["detected"]]
    total_detected = len(detected_species)

    # Confidence statistics
    all_probs = [s["probability"] for s in species_distribution]
    confidence_stats = {
        "average": float(np.mean(all_probs)),
        "minimum": float(np.min(all_probs)),
        "maximum": float(np.max(all_probs)),
        "std_dev": float(np.std(all_probs))
    }

    # Biodiversity metrics
    biodiversity = compute_biodiversity(species_probs, species_threshold)

    return {
        "density": density,
        "tree_count": round(estimated_trees, 1),
        "trees_per_hectare": round(trees_per_hectare, 1),
        "trees_per_sqkm": round(trees_per_sqkm, 1),
        "dominant_species": dominant_species,
        "least_species": least_species,
        "total_species_detected": total_detected,
        "species_distribution": species_distribution,
        "confidence_stats": confidence_stats,
        "biodiversity_metrics": biodiversity,
        "patch_area_hectares": patch_area_hectares
    }


def run_inference_paired(
    s2_path: str,
    s1_path: str,
    species_threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Run inference on paired S2 + S1 files.
    """
    model, device = ModelSingleton.get_model()
    tensor = preprocess_paired_s1_s2(s2_path, s1_path)
    tensor = tensor.to(device)

    with torch.no_grad():
        density_out, species_out = model(tensor)

    density_raw = float(density_out.squeeze().cpu())
    density_mode = getattr(model, "density_mode", "normalized")
    density, trees_per_hectare = postprocess_density_output(density_raw, density_mode=density_mode)
    species_probs = species_out.squeeze().cpu().numpy()

    return _build_result(
        density,
        species_probs,
        species_threshold,
        filename=os.path.basename(s2_path),
        trees_per_hectare_override=trees_per_hectare,
    )


def _build_result(
    density: float,
    species_probs: np.ndarray,
    species_threshold: float,
    filename: str = "",
    trees_per_hectare_override: Optional[float] = None,
) -> Dict[str, Any]:
    """Build a result dict from raw model outputs."""
    patch_area_hectares = (200 * 200) / 10000
    trees_per_hectare = (
        float(trees_per_hectare_override)
        if trees_per_hectare_override is not None
        else float(density) * MAX_TPH_REFERENCE
    )
    estimated_trees = trees_per_hectare * patch_area_hectares
    trees_per_sqkm = trees_per_hectare * 100

    species_distribution = []
    for name, prob in zip(SPECIES_LABELS, species_probs):
        species_distribution.append({
            "species": name,
            "probability": float(prob),
            "detected": bool(prob >= species_threshold)
        })
    species_distribution.sort(key=lambda x: x["probability"], reverse=True)

    dominant_species = species_distribution[0]["species"] if species_distribution else "Unknown"
    dominant_confidence = species_distribution[0]["probability"] if species_distribution else 0.0
    least_species = species_distribution[-1]["species"] if species_distribution else "Unknown"

    detected_species = [s for s in species_distribution if s["detected"]]
    total_detected = len(detected_species)

    all_probs = [s["probability"] for s in species_distribution]
    confidence_stats = {
        "average": float(np.mean(all_probs)),
        "minimum": float(np.min(all_probs)),
        "maximum": float(np.max(all_probs)),
        "std_dev": float(np.std(all_probs))
    }

    biodiversity = compute_biodiversity(species_probs, species_threshold)

    return {
        "filename": filename,
        "density": density,
        "tree_count": round(estimated_trees, 1),
        "trees_per_hectare": round(trees_per_hectare, 1),
        "trees_per_sqkm": round(trees_per_sqkm, 1),
        "dominant_species": dominant_species,
        "dominant_confidence": round(dominant_confidence, 4),
        "least_species": least_species,
        "total_species_detected": total_detected,
        "species_distribution": species_distribution,
        "confidence_stats": confidence_stats,
        "biodiversity_metrics": biodiversity,
        "patch_area_hectares": patch_area_hectares
    }


def discover_dataset(dataset_path: str) -> List[Tuple[str, str]]:
    """
    Discover paired S1/S2 files from a dataset directory.

    Expects structure:
        dataset_path/
            s1/file1.tif, file2.tif, ...
            s2/file1.tif, file2.tif, ...

    Returns list of (s2_path, s1_path) tuples for matched filenames.
    """
    s1_dir = os.path.join(dataset_path, "s1")
    s2_dir = os.path.join(dataset_path, "s2")

    if not os.path.isdir(s1_dir) or not os.path.isdir(s2_dir):
        raise FileNotFoundError(
            f"Dataset must contain 's1/' and 's2/' subdirectories. "
            f"Checked: {dataset_path}"
        )

    s1_files = {f for f in os.listdir(s1_dir) if f.lower().endswith(('.tif', '.tiff'))}
    s2_files = {f for f in os.listdir(s2_dir) if f.lower().endswith(('.tif', '.tiff'))}

    # Find matched filenames
    common = sorted(s1_files & s2_files)

    if not common:
        raise FileNotFoundError(
            f"No matching .tif files found between s1/ ({len(s1_files)} files) "
            f"and s2/ ({len(s2_files)} files). Filenames must match."
        )

    pairs = [
        (os.path.join(s2_dir, f), os.path.join(s1_dir, f))
        for f in common
    ]

    return pairs


def batch_inference(
    dataset_path: str,
    species_threshold: float = 0.5,
    batch_size: int = 32,
    progress_callback=None
) -> List[Dict[str, Any]]:
    """
    Run batch inference on a paired S1/S2 dataset.

    Args:
        dataset_path: Path to dataset containing s1/ and s2/ subdirectories
        species_threshold: Detection threshold
        batch_size: Number of images to process at once
        progress_callback: Optional callable(processed, total) for progress updates

    Returns:
        List of result dictionaries, one per file pair
    """
    model, device = ModelSingleton.get_model()
    pairs = discover_dataset(dataset_path)
    total = len(pairs)
    all_results = []

    for batch_start in range(0, total, batch_size):
        batch_pairs = pairs[batch_start:batch_start + batch_size]
        batch_tensors = []
        batch_filenames = []

        for s2_path, s1_path in batch_pairs:
            try:
                tensor = preprocess_paired_s1_s2(s2_path, s1_path)
                batch_tensors.append(tensor)
                batch_filenames.append(os.path.basename(s2_path))
            except Exception as e:
                print(f"[BatchInference] Skipping {os.path.basename(s2_path)}: {e}")
                continue

        if not batch_tensors:
            continue

        # Stack into a single batch tensor
        batch_tensor = torch.cat(batch_tensors, dim=0).to(device)  # (B, 15, 64, 64)

        with torch.no_grad():
            density_out, species_out = model(batch_tensor)

        # Process each result in the batch
        for i in range(len(batch_filenames)):
            density_raw = float(density_out[i].squeeze().cpu())
            density_mode = getattr(model, "density_mode", "normalized")
            density, trees_per_hectare = postprocess_density_output(density_raw, density_mode=density_mode)
            species_probs = species_out[i].cpu().numpy()

            result = _build_result(
                density,
                species_probs,
                species_threshold,
                filename=batch_filenames[i],
                trees_per_hectare_override=trees_per_hectare,
            )
            all_results.append(result)

        if progress_callback:
            progress_callback(len(all_results), total)

    return all_results


def save_results_csv(results: List[Dict[str, Any]], output_path: str):
    """Save inference results to a CSV file."""
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    fieldnames = [
        "filename", "density", "tree_count", "trees_per_hectare", "trees_per_sqkm",
        "dominant_species", "dominant_confidence", "total_species_detected",
        "shannon_index", "evenness", "species_richness", "biodiversity_score"
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row = {
                "filename": r.get("filename", ""),
                "density": r["density"],
                "tree_count": r["tree_count"],
                "trees_per_hectare": r["trees_per_hectare"],
                "trees_per_sqkm": r["trees_per_sqkm"],
                "dominant_species": r["dominant_species"],
                "dominant_confidence": r.get("dominant_confidence", ""),
                "total_species_detected": r["total_species_detected"],
                "shannon_index": r["biodiversity_metrics"]["shannon_index"],
                "evenness": r["biodiversity_metrics"]["evenness"],
                "species_richness": r["biodiversity_metrics"]["species_richness"],
                "biodiversity_score": r["biodiversity_metrics"]["biodiversity_score"],
            }
            writer.writerow(row)

    print(f"[CSV] Saved {len(results)} results to {output_path}")


def compute_biodiversity(
    species_probs: np.ndarray, threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Compute biodiversity metrics from species probabilities.

    Shannon Index: H = -Σ(p_i * log(p_i))
    Evenness:      E = H / log(S)
    """
    # Species richness: number of species above threshold
    detected_mask = species_probs >= threshold
    species_richness = int(np.sum(detected_mask))

    # Normalize probabilities for Shannon computation
    probs = species_probs.copy()
    probs = np.clip(probs, 1e-10, 1.0)
    prob_sum = probs.sum()

    if prob_sum > 0:
        p_normalized = probs / prob_sum
    else:
        p_normalized = np.ones_like(probs) / len(probs)

    # Shannon Index
    shannon_index = -np.sum(p_normalized * np.log(p_normalized + 1e-10))

    # Shannon Evenness
    if species_richness > 1:
        max_diversity = np.log(species_richness)
        evenness = float(shannon_index / max_diversity) if max_diversity > 0 else 0.0
    else:
        evenness = 0.0

    # Simpson's Diversity Index (1 - D)
    simpsons_index = 1.0 - np.sum(p_normalized ** 2)

    # Biodiversity score (composite)
    bio_score = (shannon_index / np.log(len(species_probs))) * 100

    return {
        "species_richness": species_richness,
        "shannon_index": round(float(shannon_index), 4),
        "evenness": round(float(evenness), 4),
        "simpsons_index": round(float(simpsons_index), 4),
        "biodiversity_score": round(float(bio_score), 2)
    }


def get_tif_metadata(file_path: str) -> Dict[str, Any]:
    """Extract metadata from a .tif file, including WGS84 geographic bounds."""
    import math
    from rasterio.crs import CRS
    from rasterio.warp import transform_bounds

    with rasterio.open(file_path) as src:
        raw_bounds = {
            "left": src.bounds.left,
            "bottom": src.bounds.bottom,
            "right": src.bounds.right,
            "top": src.bounds.top
        } if src.bounds else None

        # Compute WGS84 bounds for map display
        geo_bounds = None
        if src.bounds and src.crs:
            try:
                if src.crs != CRS.from_epsg(4326):
                    west, south, east, north = transform_bounds(
                        src.crs, CRS.from_epsg(4326),
                        src.bounds.left, src.bounds.bottom,
                        src.bounds.right, src.bounds.top
                    )
                else:
                    west, south, east, north = (
                        src.bounds.left, src.bounds.bottom,
                        src.bounds.right, src.bounds.top
                    )

                center_lat = (south + north) / 2
                center_lon = (west + east) / 2
                lat_km = abs(north - south) * 111.32
                lon_km = abs(east - west) * 111.32 * math.cos(math.radians(center_lat))
                area_km2 = round(lat_km * lon_km, 4)

                geo_bounds = {
                    "west": round(west, 6),
                    "south": round(south, 6),
                    "east": round(east, 6),
                    "north": round(north, 6),
                    "center_lat": round(center_lat, 6),
                    "center_lon": round(center_lon, 6),
                    "area_km2": area_km2,
                    "total_tiles": 1,
                }
            except Exception:
                pass

        return {
            "width": src.width,
            "height": src.height,
            "bands": src.count,
            "crs": str(src.crs) if src.crs else "Unknown",
            "resolution": list(src.res) if src.res else [200, 200],
            "bounds": raw_bounds,
            "geographic_bounds": geo_bounds,
            "dtype": str(src.dtypes[0]) if src.dtypes else "unknown",
            "nodata": src.nodata
        }
