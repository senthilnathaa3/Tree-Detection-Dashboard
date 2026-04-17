#!/usr/bin/env python3
"""
Build a multi-state FIA regional calibration profile for /api/validate-location.

Workflow:
1) Discover FIA state zip pairs in --fia-zips-dir (e.g., WV_PLOT.zip + WV_TREE.zip).
2) Extract and normalize to PLOT.csv / TREE.csv per state.
3) Convert each state to simplified FIA CSV schema.
4) Merge all states into one CSV with a `region` column.
5) Sample calibration anchors by region and call /api/validate-location.
6) Fit region-aware linear calibration profile and save JSON.

Colab example:
python build_regional_calibration.py \
  --fia-zips-dir /content/drive/MyDrive/fia_zips \
  --work-dir /content/fia_work \
  --api-base http://127.0.0.1:8001/api \
  --start-date 2024-05-01 \
  --end-date 2024-10-31 \
  --radius-km 10 \
  --sample-grid-size 5 \
  --samples-per-region 40 \
  --min-samples-per-region-fit 10
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re
import time
import zipfile
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests

from backend.calibration import (
    fit_regional_linear_tph_calibration,
    load_regional_calibration_samples_csv,
    save_calibration_profile,
)
from backend.fia_datamart import build_fia_csv_from_datamart


STATE_RE = re.compile(r"^([A-Za-z]{2})_(PLOT|TREE)\.zip$", re.IGNORECASE)


@dataclass
class StateZipPair:
    state: str
    plot_zip: str
    tree_zip: Optional[str]


def _discover_state_zip_pairs(fia_zips_dir: str) -> List[StateZipPair]:
    found: Dict[str, Dict[str, str]] = {}
    for path in glob.glob(os.path.join(fia_zips_dir, "*.zip")):
        name = os.path.basename(path)
        m = STATE_RE.match(name)
        if not m:
            continue
        state = m.group(1).upper()
        kind = m.group(2).upper()
        found.setdefault(state, {})
        found[state][kind] = path

    pairs: List[StateZipPair] = []
    for state, d in sorted(found.items()):
        if "PLOT" not in d:
            continue
        pairs.append(StateZipPair(state=state, plot_zip=d["PLOT"], tree_zip=d.get("TREE")))
    return pairs


def _download_file(url: str, out_path: str, timeout_sec: int = 120):
    r = requests.get(url, stream=True, timeout=timeout_sec)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code} for {url}")
    with open(out_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=1 << 20):
            if chunk:
                f.write(chunk)


def _auto_download_state_zips(
    states: List[str],
    fia_zips_dir: str,
    datamart_base_url: str,
    skip_existing: bool = True,
):
    os.makedirs(fia_zips_dir, exist_ok=True)
    base = datamart_base_url.rstrip("/")
    for state in states:
        st = state.strip().upper()
        if not st:
            continue
        if len(st) != 2 or not st.isalpha():
            raise ValueError(f"Invalid state abbreviation: {state}")

        for table in ("PLOT", "TREE"):
            fname = f"{st}_{table}.zip"
            out_path = os.path.join(fia_zips_dir, fname)
            if skip_existing and os.path.isfile(out_path):
                print(f"[download] skip existing {fname}")
                continue
            url = f"{base}/{fname}"
            print(f"[download] {url}")
            _download_file(url, out_path)
            print(f"[download] saved {out_path}")


def _extract_zip_member_by_suffix(zip_path: str, wanted_suffix: str, output_path: str):
    wanted = wanted_suffix.lower()
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        candidate = None
        for n in names:
            if n.lower().endswith(wanted):
                candidate = n
                break
        if candidate is None:
            raise FileNotFoundError(f"{os.path.basename(zip_path)} missing *{wanted_suffix}")
        with zf.open(candidate, "r") as src, open(output_path, "wb") as dst:
            dst.write(src.read())


def _prepare_state_datamart_dir(pair: StateZipPair, root_dir: str) -> str:
    state_dir = os.path.join(root_dir, pair.state)
    os.makedirs(state_dir, exist_ok=True)

    plot_out = os.path.join(state_dir, "PLOT.csv")
    _extract_zip_member_by_suffix(pair.plot_zip, "PLOT.csv", plot_out)

    if pair.tree_zip:
        tree_out = os.path.join(state_dir, "TREE.csv")
        _extract_zip_member_by_suffix(pair.tree_zip, "TREE.csv", tree_out)

    return state_dir


def _convert_states_to_simplified_csv(pairs: List[StateZipPair], work_dir: str) -> List[Tuple[str, str]]:
    converted: List[Tuple[str, str]] = []
    datamart_root = os.path.join(work_dir, "datamart_states")
    os.makedirs(datamart_root, exist_ok=True)

    for pair in pairs:
        state_dir = _prepare_state_datamart_dir(pair, datamart_root)
        out_csv = os.path.join(work_dir, f"{pair.state}_fia.csv")
        result = build_fia_csv_from_datamart(state_dir, out_csv)
        print(
            f"[convert] {pair.state}: rows_written={result.get('rows_written')} "
            f"plots_valid={result.get('plots_valid')}/{result.get('plots_total')} "
            f"tree_rows_total={result.get('tree_rows_total')}"
        )
        converted.append((pair.state, out_csv))
    return converted


def _merge_state_csvs_with_region(converted: List[Tuple[str, str]], output_csv: str) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for state, path in converted:
        df = pd.read_csv(path)
        if df.empty:
            continue
        df["region"] = state
        frames.append(df)

    if not frames:
        raise ValueError("No converted state CSV rows available to merge.")

    merged = pd.concat(frames, ignore_index=True)
    merged.to_csv(output_csv, index=False)
    return merged


def _sample_anchors(
    merged: pd.DataFrame,
    samples_per_region: int,
    random_seed: int,
) -> pd.DataFrame:
    needed = ["lat", "lon", "trees_per_hectare", "region"]
    for col in needed:
        if col not in merged.columns:
            raise ValueError(f"Merged FIA CSV missing required column: {col}")

    clean = merged.dropna(subset=needed).copy()
    if clean.empty:
        raise ValueError("Merged FIA CSV has no usable rows after dropping missing lat/lon/TPH/region.")

    random.seed(random_seed)
    out_parts: List[pd.DataFrame] = []
    for region, grp in clean.groupby("region"):
        n = min(samples_per_region, len(grp))
        out_parts.append(grp.sample(n=n, random_state=random_seed))
        print(f"[sample] {region}: selected={n} total={len(grp)}")

    out = pd.concat(out_parts, ignore_index=True)
    out["lat"] = out["lat"].astype(float)
    out["lon"] = out["lon"].astype(float)
    return out


def _run_validate_location(
    api_base: str,
    payload: dict,
    timeout_sec: int,
) -> dict:
    url = api_base.rstrip("/") + "/validate-location"
    r = requests.post(url, json=payload, timeout=timeout_sec)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:500]}")
    return r.json()


def _collect_calibration_pairs(
    anchors: pd.DataFrame,
    fia_csv_path: str,
    api_base: str,
    start_date: str,
    end_date: str,
    radius_km: float,
    sample_grid_size: int,
    provider: str,
    cloud_cover_max: float,
    threshold: float,
    timeout_sec: int,
    sleep_sec: float,
    year_start: Optional[int],
    year_end: Optional[int],
) -> pd.DataFrame:
    rows: List[dict] = []
    total = len(anchors)
    for i, (_, r) in enumerate(anchors.iterrows(), start=1):
        payload = {
            "lat": float(r["lat"]),
            "lon": float(r["lon"]),
            "radius_km": float(radius_km),
            "validation_source": "fia",
            "fia_csv_path": fia_csv_path,
            "provider": provider,
            "start_date": start_date,
            "end_date": end_date,
            "sample_grid_size": int(sample_grid_size),
            "cloud_cover_max": float(cloud_cover_max),
            "threshold": float(threshold),
        }
        if year_start is not None:
            payload["year_start"] = int(year_start)
        if year_end is not None:
            payload["year_end"] = int(year_end)

        rec = {
            "idx": i,
            "region": str(r["region"]),
            "lat": float(r["lat"]),
            "lon": float(r["lon"]),
            "status": "error",
            "model_tph": None,
            "fia_tph": None,
            "error": None,
        }

        try:
            data = _run_validate_location(api_base=api_base, payload=payload, timeout_sec=timeout_sec)
            da = data.get("comparison", {}).get("density_agreement", {})
            m = da.get("model_mean_trees_per_hectare")
            f = da.get("fia_mean_trees_per_hectare")
            if m is None or f is None:
                rec["error"] = "missing model/fia tph in response"
            else:
                rec["status"] = "ok"
                rec["model_tph"] = float(m)
                rec["fia_tph"] = float(f)
                rec["error"] = ""
        except Exception as e:
            rec["error"] = str(e)

        rows.append(rec)
        if i % 10 == 0 or i == total:
            ok = sum(1 for x in rows if x["status"] == "ok")
            print(f"[validate] {i}/{total} complete, ok={ok}, fail={i-ok}")

        if sleep_sec > 0:
            time.sleep(sleep_sec)

    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build multi-state FIA regional calibration profile.")
    p.add_argument("--fia-zips-dir", required=True, help="Directory containing <STATE>_PLOT.zip and <STATE>_TREE.zip files.")
    p.add_argument("--auto-download-fia", action="store_true", help="Auto-download state PLOT/TREE zips from FIA DataMart.")
    p.add_argument("--states", default="", help="Comma-separated state abbreviations, e.g. WV,VA,PA. Required for --auto-download-fia.")
    p.add_argument(
        "--datamart-base-url",
        default="https://apps.fs.usda.gov/fia/datamart/CSV",
        help="FIA DataMart CSV base URL.",
    )
    p.add_argument("--download-skip-existing", action="store_true", help="Skip downloading files that already exist.")
    p.add_argument("--work-dir", required=True, help="Writable output working directory.")
    p.add_argument("--api-base", default="http://127.0.0.1:8001/api", help="Base API URL.")

    p.add_argument("--start-date", required=True, help="Remote imagery start date YYYY-MM-DD.")
    p.add_argument("--end-date", required=True, help="Remote imagery end date YYYY-MM-DD.")
    p.add_argument("--provider", default="planetary_computer")
    p.add_argument("--radius-km", type=float, default=10.0)
    p.add_argument("--sample-grid-size", type=int, default=5)
    p.add_argument("--cloud-cover-max", type=float, default=40.0)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--timeout-sec", type=int, default=240)
    p.add_argument("--sleep-sec", type=float, default=0.0)

    p.add_argument("--year-start", type=int, default=None)
    p.add_argument("--year-end", type=int, default=None)

    p.add_argument("--samples-per-region", type=int, default=40)
    p.add_argument("--random-seed", type=int, default=42)
    p.add_argument("--min-samples-per-region-fit", type=int, default=10)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.work_dir, exist_ok=True)

    if args.auto_download_fia:
        states = [s.strip().upper() for s in args.states.split(",") if s.strip()]
        if not states:
            raise ValueError("--states is required when --auto-download-fia is set.")
        print(f"[step] auto-downloading FIA zips for states={states} ...")
        _auto_download_state_zips(
            states=states,
            fia_zips_dir=args.fia_zips_dir,
            datamart_base_url=args.datamart_base_url,
            skip_existing=args.download_skip_existing,
        )

    print("[step] discovering FIA zip pairs...")
    pairs = _discover_state_zip_pairs(args.fia_zips_dir)
    if not pairs:
        raise ValueError(
            "No valid state zip pairs found. Expect files like WV_PLOT.zip and WV_TREE.zip."
        )
    print(f"[discover] states={len(pairs)} -> {[p.state for p in pairs]}")

    print("[step] converting FIA state zips...")
    converted = _convert_states_to_simplified_csv(pairs=pairs, work_dir=args.work_dir)

    merged_fia_csv = os.path.join(args.work_dir, "fia_multi_state.csv")
    print("[step] merging converted FIA csvs...")
    merged = _merge_state_csvs_with_region(converted, merged_fia_csv)
    print(f"[merge] rows={len(merged)} output={merged_fia_csv}")

    print("[step] sampling calibration anchors...")
    anchors = _sample_anchors(
        merged=merged,
        samples_per_region=args.samples_per_region,
        random_seed=args.random_seed,
    )
    anchors_csv = os.path.join(args.work_dir, "calibration_anchor_points.csv")
    anchors.to_csv(anchors_csv, index=False)
    print(f"[anchors] rows={len(anchors)} output={anchors_csv}")

    print("[step] collecting model-vs-fia calibration pairs from validate-location...")
    pair_df = _collect_calibration_pairs(
        anchors=anchors,
        fia_csv_path=merged_fia_csv,
        api_base=args.api_base,
        start_date=args.start_date,
        end_date=args.end_date,
        radius_km=args.radius_km,
        sample_grid_size=args.sample_grid_size,
        provider=args.provider,
        cloud_cover_max=args.cloud_cover_max,
        threshold=args.threshold,
        timeout_sec=args.timeout_sec,
        sleep_sec=args.sleep_sec,
        year_start=args.year_start,
        year_end=args.year_end,
    )
    pair_csv_all = os.path.join(args.work_dir, "calibration_pairs_raw.csv")
    pair_df.to_csv(pair_csv_all, index=False)

    ok_df = pair_df[pair_df["status"] == "ok"][["model_tph", "fia_tph", "region", "lat", "lon"]].copy()
    pair_csv_ok = os.path.join(args.work_dir, "calibration_samples_with_region.csv")
    ok_df.to_csv(pair_csv_ok, index=False)
    print(f"[pairs] total={len(pair_df)} ok={len(ok_df)} raw={pair_csv_all} samples={pair_csv_ok}")

    if len(ok_df) < 2:
        raise ValueError("Not enough successful validation pairs to fit calibration.")

    print("[step] fitting regional calibration profile...")
    regional_samples = load_regional_calibration_samples_csv(pair_csv_ok, region_column="region")
    profile = fit_regional_linear_tph_calibration(
        regional_samples,
        min_samples_per_region=args.min_samples_per_region_fit,
    )
    profile_path = os.path.join(args.work_dir, "regional_calibration.json")
    save_calibration_profile(profile, profile_path)
    print(f"[fit] saved profile={profile_path}")

    summary = {
        "status": "success",
        "states": [p.state for p in pairs],
        "merged_fia_csv": merged_fia_csv,
        "anchors_csv": anchors_csv,
        "calibration_pairs_raw_csv": pair_csv_all,
        "calibration_samples_csv": pair_csv_ok,
        "regional_profile_json": profile_path,
        "ok_pairs": int(len(ok_df)),
        "total_pairs": int(len(pair_df)),
    }
    summary_path = os.path.join(args.work_dir, "build_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[done] summary={summary_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
