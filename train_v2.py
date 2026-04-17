#!/usr/bin/env python3
"""
Train TreeSatMultiHeadModelV2.

Generalized training modes:
1) local: paired S1/S2 files from dataset_path + labels CSV keyed by filename
2) aoi:   lat/lon sample CSV (WV or any state/region) with date ranges + labels

Outputs:
- best_v2.pth
- last_v2.pth
- train_history.csv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

from backend.inference import discover_dataset, preprocess_paired_s1_s2
from backend.model_loader import SPECIES_LABELS, TreeSatMultiHeadModelV2
from backend.remote_inference import fetch_remote_tensor_planetary_computer


DENSITY_KEYS = ["density", "tree_density", "density_target", "tree_density_target", "target_density"]


def _norm_key(text: str) -> str:
    return "".join(ch.lower() for ch in str(text) if ch.isalnum())


def _parse_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    s = str(v).strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _parse_binary(v: Any) -> int:
    if v is None:
        return 0
    s = str(v).strip().lower()
    if s in {"1", "true", "t", "yes", "y"}:
        return 1
    try:
        return 1 if float(s) >= 0.5 else 0
    except ValueError:
        return 0


def _build_species_column_map(headers: List[str]) -> Dict[str, str]:
    normalized_headers = {_norm_key(h): h for h in headers}
    mapping: Dict[str, str] = {}
    for sp in SPECIES_LABELS:
        for cand in (f"species_{sp}", sp):
            nk = _norm_key(cand)
            if nk in normalized_headers:
                mapping[sp] = normalized_headers[nk]
                break
    return mapping


def _find_density(row: Dict[str, Any]) -> Optional[float]:
    for key in DENSITY_KEYS:
        if key in row:
            return _parse_float(row[key])
    return None


@dataclass
class LabelRow:
    density: Optional[float]
    species: np.ndarray


class LocalPairedDataset(Dataset):
    def __init__(self, dataset_path: str, labels_csv: str):
        self.pairs = discover_dataset(dataset_path)
        self.by_filename = {os.path.basename(s2): (s2, s1) for s2, s1 in self.pairs}
        self.labels = self._load_labels(labels_csv)
        self.keys = sorted(set(self.by_filename.keys()) & set(self.labels.keys()))

        if not self.keys:
            raise ValueError("No matching filenames between local dataset and labels CSV")

    def _load_labels(self, labels_csv: str) -> Dict[str, LabelRow]:
        if not os.path.isfile(labels_csv):
            raise FileNotFoundError(f"Labels CSV not found: {labels_csv}")

        out: Dict[str, LabelRow] = {}
        with open(labels_csv, "r", newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                raise ValueError("Labels CSV has no header row")

            headers = [h.strip() for h in reader.fieldnames if h is not None]
            lower_map = {h.lower().strip(): h for h in headers}
            if "filename" not in lower_map:
                raise ValueError("Labels CSV must include filename column")

            filename_col = lower_map["filename"]
            species_map = _build_species_column_map(headers)
            if not species_map:
                raise ValueError("No species columns found in labels CSV")

            for raw in reader:
                row = {(k.strip() if isinstance(k, str) else k): v for k, v in raw.items()}
                fname = str(row.get(filename_col, "")).strip()
                if not fname:
                    continue

                density = _find_density({k.lower(): v for k, v in row.items() if isinstance(k, str)})
                if density is None:
                    density = 0.0

                species = np.zeros(len(SPECIES_LABELS), dtype=np.float32)
                for i, sp in enumerate(SPECIES_LABELS):
                    col = species_map.get(sp)
                    if col is not None:
                        species[i] = float(_parse_binary(row.get(col)))

                out[fname] = LabelRow(density=float(density), species=species)

        return out

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx: int):
        fname = self.keys[idx]
        s2_path, s1_path = self.by_filename[fname]
        tensor = preprocess_paired_s1_s2(s2_path, s1_path).squeeze(0).float()  # (15,64,64)
        label = self.labels[fname]
        density = torch.tensor([label.density], dtype=torch.float32)
        species = torch.tensor(label.species, dtype=torch.float32)
        return tensor, density, species


class AOISampleDataset(Dataset):
    """
    AOI-mode generalized dataset.
    CSV required columns:
    - lat
    - lon
    - start_date
    - end_date

    Optional:
    - radius_km (default from CLI)
    - cloud_cover_max (default from CLI)
    - density/tree_density target
    - species columns: species_<label> or direct label columns

    This supports WV or any other state/region based on lat/lon samples.
    """

    def __init__(
        self,
        samples_csv: str,
        default_radius_km: float,
        default_cloud_cover_max: float,
        cache_dir: Optional[str] = None,
    ):
        if not os.path.isfile(samples_csv):
            raise FileNotFoundError(f"AOI samples CSV not found: {samples_csv}")

        self.default_radius_km = float(default_radius_km)
        self.default_cloud_cover_max = float(default_cloud_cover_max)
        self.cache_dir = cache_dir
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)

        self.rows = self._load_rows(samples_csv)
        if not self.rows:
            raise ValueError("No valid AOI sample rows found")

    def _load_rows(self, samples_csv: str) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        with open(samples_csv, "r", newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                raise ValueError("AOI samples CSV has no header row")

            headers = [h.strip() for h in reader.fieldnames if h is not None]
            low = {h.lower().strip(): h for h in headers}
            required = ["lat", "lon", "start_date", "end_date"]
            for req in required:
                if req not in low:
                    raise ValueError(f"AOI samples CSV missing required column: {req}")

            species_map = _build_species_column_map(headers)

            for raw in reader:
                row = {(k.strip() if isinstance(k, str) else k): v for k, v in raw.items()}
                lat = _parse_float(row.get(low["lat"]))
                lon = _parse_float(row.get(low["lon"]))
                start_date = str(row.get(low["start_date"], "")).strip()
                end_date = str(row.get(low["end_date"], "")).strip()
                if lat is None or lon is None or not start_date or not end_date:
                    continue

                radius_km = _parse_float(row.get(low.get("radius_km", "")))
                cloud_cover_max = _parse_float(row.get(low.get("cloud_cover_max", "")))

                density = _find_density({k.lower(): v for k, v in row.items() if isinstance(k, str)})
                if density is None:
                    density = 0.0

                species = np.zeros(len(SPECIES_LABELS), dtype=np.float32)
                for i, sp in enumerate(SPECIES_LABELS):
                    col = species_map.get(sp)
                    if col is not None:
                        species[i] = float(_parse_binary(row.get(col)))

                out.append(
                    {
                        "lat": float(lat),
                        "lon": float(lon),
                        "start_date": start_date,
                        "end_date": end_date,
                        "radius_km": float(radius_km) if radius_km is not None else self.default_radius_km,
                        "cloud_cover_max": float(cloud_cover_max) if cloud_cover_max is not None else self.default_cloud_cover_max,
                        "density": float(density),
                        "species": species,
                    }
                )

        return out

    def __len__(self):
        return len(self.rows)

    def _cache_key(self, row: Dict[str, Any]) -> str:
        raw = f"{row['lat']}|{row['lon']}|{row['start_date']}|{row['end_date']}|{row['radius_km']}|{row['cloud_cover_max']}"
        return hashlib.md5(raw.encode("utf-8")).hexdigest()

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        tensor = None

        if self.cache_dir:
            key = self._cache_key(row)
            npy_path = os.path.join(self.cache_dir, f"{key}.npy")
            if os.path.isfile(npy_path):
                arr = np.load(npy_path)
                tensor = torch.from_numpy(arr).float()

        if tensor is None:
            t, _meta = fetch_remote_tensor_planetary_computer(
                lat=row["lat"],
                lon=row["lon"],
                start_date=row["start_date"],
                end_date=row["end_date"],
                radius_km=row["radius_km"],
                cloud_cover_max=row["cloud_cover_max"],
            )
            tensor = t.squeeze(0).float()
            if self.cache_dir:
                np.save(npy_path, tensor.numpy())

        density = torch.tensor([row["density"]], dtype=torch.float32)
        species = torch.tensor(row["species"], dtype=torch.float32)
        return tensor, density, species


def collate_batch(batch):
    xs, ds, ys = zip(*batch)
    x = torch.stack(xs, dim=0)
    d = torch.stack(ds, dim=0)
    y = torch.stack(ys, dim=0)
    return x, d, y


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer],
    density_loss_weight: float,
    species_loss_weight: float,
):
    train_mode = optimizer is not None
    model.train(train_mode)

    mse = nn.MSELoss()
    bce = nn.BCELoss()

    total_loss = 0.0
    total_density = 0.0
    total_species = 0.0
    n = 0

    for x, d_true, y_true in loader:
        x = x.to(device)
        d_true = d_true.to(device)
        y_true = y_true.to(device)

        if train_mode:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train_mode):
            d_pred, y_pred = model(x)
            loss_d = mse(d_pred, d_true)
            loss_s = bce(y_pred, y_true)
            loss = density_loss_weight * loss_d + species_loss_weight * loss_s
            if train_mode:
                loss.backward()
                optimizer.step()

        bs = x.size(0)
        total_loss += float(loss.item()) * bs
        total_density += float(loss_d.item()) * bs
        total_species += float(loss_s.item()) * bs
        n += bs

    if n == 0:
        return {"loss": 0.0, "density_loss": 0.0, "species_loss": 0.0}

    return {
        "loss": total_loss / n,
        "density_loss": total_density / n,
        "species_loss": total_species / n,
    }


def save_checkpoint(path: str, payload: Dict[str, Any]):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(payload, path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train TreeSatMultiHeadModelV2")
    p.add_argument("--mode", choices=["local", "aoi"], required=True)

    p.add_argument("--dataset-path", default="", help="Local paired dataset path containing s1/ and s2/")
    p.add_argument("--labels-csv", default="", help="CSV labels for local mode (filename + targets)")

    p.add_argument("--aoi-samples-csv", default="", help="CSV samples for AOI mode (lat/lon/date + targets)")
    p.add_argument("--aoi-default-radius-km", type=float, default=0.2)
    p.add_argument("--aoi-default-cloud-cover-max", type=float, default=40.0)
    p.add_argument("--aoi-cache-dir", default="")

    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--density-loss-weight", type=float, default=1.0)
    p.add_argument("--species-loss-weight", type=float, default=1.0)

    p.add_argument("--output-dir", default="backend/checkpoints/v2")
    p.add_argument("--resume-checkpoint", default="")
    p.add_argument("--num-workers", type=int, default=2)

    return p.parse_args()


def main() -> int:
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.mode == "local":
        if not args.dataset_path or not args.labels_csv:
            raise ValueError("local mode requires --dataset-path and --labels-csv")
        dataset = LocalPairedDataset(args.dataset_path, args.labels_csv)
    else:
        if not args.aoi_samples_csv:
            raise ValueError("aoi mode requires --aoi-samples-csv")
        dataset = AOISampleDataset(
            samples_csv=args.aoi_samples_csv,
            default_radius_km=args.aoi_default_radius_km,
            default_cloud_cover_max=args.aoi_default_cloud_cover_max,
            cache_dir=args.aoi_cache_dir or None,
        )

    total = len(dataset)
    if total < 2:
        raise ValueError("Need at least 2 samples for train/val split")

    val_size = max(1, int(total * args.val_ratio))
    train_size = total - val_size
    train_ds, val_ds = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TreeSatMultiHeadModelV2().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    start_epoch = 1
    best_val = float("inf")

    if args.resume_checkpoint:
        ckpt = torch.load(args.resume_checkpoint, map_location=device, weights_only=False)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"], strict=False)
            if "optimizer_state_dict" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            start_epoch = int(ckpt.get("epoch", 0)) + 1
            best_val = float(ckpt.get("best_val_loss", best_val))
        else:
            model.load_state_dict(ckpt, strict=False)

    os.makedirs(args.output_dir, exist_ok=True)
    history_path = os.path.join(args.output_dir, "train_history.csv")
    best_path = os.path.join(args.output_dir, "best_v2.pth")
    last_path = os.path.join(args.output_dir, "last_v2.pth")
    cfg_path = os.path.join(args.output_dir, "train_config.json")

    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    history_rows = []

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()

        train_metrics = run_epoch(
            model,
            train_loader,
            device,
            optimizer,
            density_loss_weight=args.density_loss_weight,
            species_loss_weight=args.species_loss_weight,
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            device,
            optimizer=None,
            density_loss_weight=args.density_loss_weight,
            species_loss_weight=args.species_loss_weight,
        )

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_density_loss": train_metrics["density_loss"],
            "train_species_loss": train_metrics["species_loss"],
            "val_loss": val_metrics["loss"],
            "val_density_loss": val_metrics["density_loss"],
            "val_species_loss": val_metrics["species_loss"],
            "seconds": round(time.time() - t0, 2),
        }
        history_rows.append(row)

        print(
            f"[Epoch {epoch:03d}] "
            f"train={row['train_loss']:.6f} "
            f"val={row['val_loss']:.6f} "
            f"(density={row['val_density_loss']:.6f}, species={row['val_species_loss']:.6f})"
        )

        checkpoint_payload = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": best_val,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "timestamp_utc": datetime.utcnow().isoformat() + "Z",
            "model_variant": "v2",
            "train_config": vars(args),
        }
        save_checkpoint(last_path, checkpoint_payload)

        if row["val_loss"] < best_val:
            best_val = row["val_loss"]
            checkpoint_payload["best_val_loss"] = best_val
            save_checkpoint(best_path, checkpoint_payload)
            print(f"  -> new best checkpoint: {best_path} (val={best_val:.6f})")

        with open(history_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(history_rows[0].keys()))
            writer.writeheader()
            writer.writerows(history_rows)

    print("Training complete")
    print(f"Best checkpoint: {best_path}")
    print(f"Last checkpoint: {last_path}")
    print(f"History CSV: {history_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
