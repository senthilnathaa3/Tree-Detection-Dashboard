#!/usr/bin/env python3
"""
Export a founder-ready pitch bundle from the location validation endpoint.

Outputs:
- pitch_response.json
- pitch_summary.csv
- representative_patches.csv
- high_density_patch.png / medium_density_patch.png / low_density_patch.png
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import os
from typing import Any, Dict, List

import requests


def _mkdir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _decode_data_url(data_url: str) -> bytes:
    if "," not in data_url:
        raise ValueError("Annotated image is not a valid data URL.")
    _, encoded = data_url.split(",", 1)
    return base64.b64decode(encoded)


def _write_json(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _write_summary_csv(path: str, response: Dict[str, Any]) -> None:
    validation = response.get("validation", {})
    comparison = validation.get("comparison", {})
    calibrated = comparison.get("density_agreement_calibrated", {}) or {}
    pitch_regions = response.get("pitch_regions", {}) or {}
    summary = pitch_regions.get("summary", {}) or {}

    rows = [
        {
            "section": "aoi",
            "label": "agreement_score",
            "value": (
                round(100.0 - abs(float(calibrated["percent_difference"])), 4)
                if calibrated.get("percent_difference") is not None
                else None
            ),
        },
        {
            "section": "aoi",
            "label": "model_tph_calibrated",
            "value": calibrated.get("model_tph_calibrated"),
        },
        {
            "section": "aoi",
            "label": "fia_tph",
            "value": calibrated.get("fia_mean_trees_per_hectare"),
        },
        {
            "section": "aoi",
            "label": "percent_difference_calibrated",
            "value": calibrated.get("percent_difference"),
        },
    ]

    for bucket in ("high", "medium", "low"):
        bucket_summary = summary.get(bucket, {}) or {}
        rows.append(
            {
                "section": f"bucket:{bucket}",
                "label": "patch_count",
                "value": bucket_summary.get("count"),
            }
        )
        rows.append(
            {
                "section": f"bucket:{bucket}",
                "label": "mean_calibrated_tph",
                "value": bucket_summary.get("mean_calibrated_tph"),
            }
        )
        rows.append(
            {
                "section": f"bucket:{bucket}",
                "label": "mean_patch_tree_count",
                "value": bucket_summary.get("mean_patch_tree_count"),
            }
        )

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["section", "label", "value"])
        writer.writeheader()
        writer.writerows(rows)


def _write_representatives_csv(path: str, representatives: List[Dict[str, Any]]) -> None:
    fieldnames = [
        "density_bucket",
        "patch_id",
        "calibrated_tph",
        "patch_tree_count_calibrated",
        "dominant_species",
        "crown_candidates",
        "fia_local_tph",
        "fia_local_plots",
        "center_lat",
        "center_lon",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for patch in representatives:
            patch_aoi = patch.get("patch_aoi", {}) or {}
            crown = patch.get("crown_annotation", {}) or {}
            fia_local = patch.get("fia_local", {}) or {}
            writer.writerow(
                {
                    "density_bucket": patch.get("density_bucket"),
                    "patch_id": patch.get("patch_id"),
                    "calibrated_tph": patch.get("calibrated_tph"),
                    "patch_tree_count_calibrated": patch.get("patch_tree_count_calibrated"),
                    "dominant_species": patch.get("dominant_species"),
                    "crown_candidates": crown.get("candidate_count"),
                    "fia_local_tph": fia_local.get("trees_per_hectare"),
                    "fia_local_plots": fia_local.get("plots_in_patch"),
                    "center_lat": patch_aoi.get("center_lat"),
                    "center_lon": patch_aoi.get("center_lon"),
                }
            )


def _save_representative_images(output_dir: str, representatives: List[Dict[str, Any]]) -> List[str]:
    written = []
    for patch in representatives:
        bucket = patch.get("density_bucket") or "patch"
        crown = patch.get("crown_annotation", {}) or {}
        data_url = crown.get("annotated_image_data_url")
        if not data_url:
            continue
        image_path = os.path.join(output_dir, f"{bucket}_density_patch.png")
        with open(image_path, "wb") as f:
            f.write(_decode_data_url(data_url))
        written.append(image_path)
    return written


def build_payload(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "lat": args.lat,
        "lon": args.lon,
        "radius_km": args.radius_km,
        "provider": args.provider,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "cloud_cover_max": args.cloud_cover_max,
        "threshold": args.threshold,
        "validation_source": "fia",
        "fia_csv_path": args.fia_csv_path,
        "year_start": args.year_start,
        "year_end": args.year_end,
        "sample_grid_size": args.sample_grid_size,
        "calibration_profile_path": args.calibration_profile_path,
        "calibration_region": args.calibration_region,
        "crown_radius_km": args.crown_radius_km,
        "crown_ndvi_threshold": args.crown_ndvi_threshold,
        "crown_min_area_px": args.crown_min_area_px,
        "crown_max_candidates": args.crown_max_candidates,
        "crown_align_with_model": args.crown_align_with_model,
        "include_pitch_visuals": args.include_pitch_visuals,
        "representative_imagery_source": args.representative_imagery_source,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a pitch bundle from validate-location-crowns.")
    parser.add_argument("--api-base", default="http://127.0.0.1:8001/api")
    parser.add_argument("--output-dir", default="./pitch_bundle")
    parser.add_argument("--lat", type=float, default=38.7)
    parser.add_argument("--lon", type=float, default=-79.9)
    parser.add_argument("--radius-km", type=float, default=10.0)
    parser.add_argument("--provider", default="planetary_computer")
    parser.add_argument("--start-date", default="2024-01-01")
    parser.add_argument("--end-date", default="2024-12-31")
    parser.add_argument("--cloud-cover-max", type=float, default=80.0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--fia-csv-path", required=True)
    parser.add_argument("--year-start", type=int, default=2018)
    parser.add_argument("--year-end", type=int, default=2024)
    parser.add_argument("--sample-grid-size", type=int, default=5)
    parser.add_argument("--calibration-profile-path", required=True)
    parser.add_argument("--calibration-region", default="WV")
    parser.add_argument("--crown-radius-km", type=float, default=0.2)
    parser.add_argument("--crown-ndvi-threshold", type=float, default=0.45)
    parser.add_argument("--crown-min-area-px", type=int, default=12)
    parser.add_argument("--crown-max-candidates", type=int, default=5000)
    parser.add_argument("--crown-align-with-model", dest="crown_align_with_model", action="store_true")
    parser.add_argument("--no-crown-align-with-model", dest="crown_align_with_model", action="store_false")
    parser.set_defaults(crown_align_with_model=True)
    parser.add_argument("--include-pitch-visuals", action="store_true", default=False)
    parser.add_argument("--representative-imagery-source", default="naip", choices=["auto", "naip", "sentinel"])
    parser.add_argument("--timeout", type=int, default=1800)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = _mkdir(os.path.abspath(os.path.expanduser(args.output_dir)))
    payload = build_payload(args)

    response = requests.post(
        f"{args.api_base.rstrip('/')}/validate-location-crowns",
        json=payload,
        timeout=args.timeout,
    )
    response.raise_for_status()
    data = response.json()

    _write_json(os.path.join(output_dir, "pitch_response.json"), data)
    _write_summary_csv(os.path.join(output_dir, "pitch_summary.csv"), data)

    representatives = (data.get("pitch_regions", {}) or {}).get("representatives", []) or []
    _write_representatives_csv(os.path.join(output_dir, "representative_patches.csv"), representatives)
    written_images = _save_representative_images(output_dir, representatives)

    print(json.dumps(
        {
            "status": "success",
            "output_dir": output_dir,
            "files": [
                "pitch_response.json",
                "pitch_summary.csv",
                "representative_patches.csv",
                *[os.path.basename(p) for p in written_images],
            ],
        },
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
