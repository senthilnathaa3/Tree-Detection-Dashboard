"""
Ecological metrics computation module.
Computes advanced biodiversity, density, and statistical metrics.
"""

import numpy as np
from typing import Dict, Any, List


def compute_density_statistics(densities: List[float]) -> Dict[str, Any]:
    """Compute statistical metrics from multiple density predictions."""
    arr = np.array(densities)
    return {
        "mean": round(float(np.mean(arr)), 4),
        "median": round(float(np.median(arr)), 4),
        "std": round(float(np.std(arr)), 4),
        "min": round(float(np.min(arr)), 4),
        "max": round(float(np.max(arr)), 4),
        "variance": round(float(np.var(arr)), 6),
        "q25": round(float(np.percentile(arr, 25)), 4),
        "q75": round(float(np.percentile(arr, 75)), 4),
        "count": len(densities)
    }


def compute_species_summary(all_distributions: List[List[Dict]]) -> Dict[str, Any]:
    """
    Aggregate species distributions across multiple patches.
    
    Args:
        all_distributions: List of species_distribution lists from inference
    """
    if not all_distributions:
        return {}

    species_names = [s["species"] for s in all_distributions[0]]
    num_species = len(species_names)

    # Aggregate probabilities
    prob_matrix = np.zeros((len(all_distributions), num_species))
    for i, dist in enumerate(all_distributions):
        for j, sp in enumerate(dist):
            prob_matrix[i, j] = sp["probability"]

    # Mean probabilities per species
    mean_probs = prob_matrix.mean(axis=0)
    std_probs = prob_matrix.std(axis=0)

    species_summary = []
    for k in range(num_species):
        species_summary.append({
            "species": species_names[k],
            "mean_probability": round(float(mean_probs[k]), 4),
            "std_probability": round(float(std_probs[k]), 4),
            "detection_rate": round(
                float(np.mean(prob_matrix[:, k] >= 0.5)), 4
            )
        })

    species_summary.sort(key=lambda x: x["mean_probability"], reverse=True)

    return {
        "species_summary": species_summary,
        "total_patches": len(all_distributions),
        "most_common_species": species_summary[0]["species"],
        "least_common_species": species_summary[-1]["species"]
    }


def compute_batch_biodiversity(
    all_results: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Compute aggregated biodiversity metrics across a batch of results.
    """
    if not all_results:
        return {}

    shannon_values = [r["biodiversity_metrics"]["shannon_index"] for r in all_results]
    evenness_values = [r["biodiversity_metrics"]["evenness"] for r in all_results]
    richness_values = [r["biodiversity_metrics"]["species_richness"] for r in all_results]
    bio_scores = [r["biodiversity_metrics"]["biodiversity_score"] for r in all_results]

    return {
        "mean_shannon_index": round(float(np.mean(shannon_values)), 4),
        "mean_evenness": round(float(np.mean(evenness_values)), 4),
        "mean_richness": round(float(np.mean(richness_values)), 2),
        "mean_biodiversity_score": round(float(np.mean(bio_scores)), 2),
        "shannon_std": round(float(np.std(shannon_values)), 4),
        "evenness_std": round(float(np.std(evenness_values)), 4),
        "richness_range": [int(np.min(richness_values)), int(np.max(richness_values))],
        "total_patches_analyzed": len(all_results)
    }


def generate_heatmap_data(
    all_results: List[Dict[str, Any]],
    grid_size: int = 5
) -> Dict[str, Any]:
    """
    Generate heatmap grid data from batch predictions.
    Arranges results in a grid for spatial visualization.
    """
    n = len(all_results)

    # Determine grid dimensions
    cols = min(grid_size, n)
    rows = (n + cols - 1) // cols

    density_grid = np.zeros((rows, cols))
    richness_grid = np.zeros((rows, cols))
    shannon_grid = np.zeros((rows, cols))

    for i, result in enumerate(all_results):
        r, c = divmod(i, cols)
        if r < rows:
            density_grid[r, c] = result["density"]
            richness_grid[r, c] = result["biodiversity_metrics"]["species_richness"]
            shannon_grid[r, c] = result["biodiversity_metrics"]["shannon_index"]

    return {
        "density_heatmap": density_grid.tolist(),
        "richness_heatmap": richness_grid.tolist(),
        "shannon_heatmap": shannon_grid.tolist(),
        "grid_rows": rows,
        "grid_cols": cols,
        "patch_count": n
    }
