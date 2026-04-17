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
TPH_KEYS = [
    "trees_per_hectare",
    "tph",
    "fia_tph",
    "target_tph",
    "tree_tph",
    "model_tph",
]


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


def _find_tph(row: Dict[str, Any]) -> Optional[float]:
    for key in TPH_KEYS:
        if key in row:
            return _parse_float(row[key])
    return None


@dataclass
class LabelRow:
    density: Optional[float]
    tph: Optional[float]
    species: np.ndarray


class LocalPairedDataset(Dataset):
    def __init__(
        self,
        dataset_path: str,
        labels_csv: str,
        density_target_mode: str = "tph",
        tph_reference: float = 1000.0,
    ):
        self.pairs = discover_dataset(dataset_path)
        self.by_filename = {os.path.basename(s2): (s2, s1) for s2, s1 in self.pairs}
        self.density_target_mode = (density_target_mode or "tph").strip().lower()
        self.tph_reference = float(tph_reference)
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

                lowered = {k.lower(): v for k, v in row.items() if isinstance(k, str)}
                density = _find_density(lowered)
                tph = _find_tph(lowered)

                species = np.zeros(len(SPECIES_LABELS), dtype=np.float32)
                for i, sp in enumerate(SPECIES_LABELS):
                    col = species_map.get(sp)
                    if col is not None:
                        species[i] = float(_parse_binary(row.get(col)))

                out[fname] = LabelRow(
                    density=float(density) if density is not None else None,
                    tph=float(tph) if tph is not None else None,
                    species=species,
                )

        return out

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx: int):
        fname = self.keys[idx]
        s2_path, s1_path = self.by_filename[fname]
        tensor = preprocess_paired_s1_s2(s2_path, s1_path).squeeze(0).float()  # (15,64,64)
        label = self.labels[fname]
        if self.density_target_mode == "tph":
            target = (
                float(label.tph)
                if label.tph is not None
                else float(label.density or 0.0) * self.tph_reference
            )
        else:
            target = (
                float(label.density)
                if label.density is not None
                else float(label.tph or 0.0) / max(1e-6, self.tph_reference)
            )
        density = torch.tensor([target], dtype=torch.float32)
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
        density_target_mode: str = "tph",
        tph_reference: float = 1000.0,
    ):
        if not os.path.isfile(samples_csv):
            raise FileNotFoundError(f"AOI samples CSV not found: {samples_csv}")

        self.default_radius_km = float(default_radius_km)
        self.default_cloud_cover_max = float(default_cloud_cover_max)
        self.density_target_mode = (density_target_mode or "tph").strip().lower()
        self.tph_reference = float(tph_reference)
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

                lowered = {k.lower(): v for k, v in row.items() if isinstance(k, str)}
                density = _find_density(lowered)
                tph = _find_tph(lowered)

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
                        "density": float(density) if density is not None else None,
                        "tph": float(tph) if tph is not None else None,
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

        if self.density_target_mode == "tph":
            target = (
                float(row["tph"])
                if row.get("tph") is not None
                else float(row.get("density") or 0.0) * self.tph_reference
            )
        else:
            target = (
                float(row["density"])
                if row.get("density") is not None
                else float(row.get("tph") or 0.0) / max(1e-6, self.tph_reference)
            )
        density = torch.tensor([target], dtype=torch.float32)
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
    density_loss_kind: str = "huber",
):
    train_mode = optimizer is not None
    model.train(train_mode)

    mse = nn.MSELoss()
    l1 = nn.L1Loss()
    huber = nn.SmoothL1Loss(beta=1.0)
    bce = nn.BCELoss()

    total_loss = 0.0
    total_density = 0.0
    total_species = 0.0
    n = 0
    preds: List[float] = []
    trues: List[float] = []

    for x, d_true, y_true in loader:
        x = x.to(device)
        d_true = d_true.to(device)
        y_true = y_true.to(device)

        if train_mode:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train_mode):
            d_pred, y_pred = model(x)
            if density_loss_kind == "l1":
                loss_d = l1(d_pred, d_true)
            elif density_loss_kind == "mse":
                loss_d = mse(d_pred, d_true)
            else:
                loss_d = huber(d_pred, d_true)
            loss_s = bce(y_pred, y_true)
            loss = density_loss_weight * loss_d + species_loss_weight * loss_s
            if train_mode:
                loss.backward()
                optimizer.step()

        preds.extend(d_pred.detach().cpu().numpy().reshape(-1).tolist())
        trues.extend(d_true.detach().cpu().numpy().reshape(-1).tolist())

        bs = x.size(0)
        total_loss += float(loss.item()) * bs
        total_density += float(loss_d.item()) * bs
        total_species += float(loss_s.item()) * bs
        n += bs

    if n == 0:
        return {"loss": 0.0, "density_loss": 0.0, "species_loss": 0.0, "mae": 0.0, "rmse": 0.0, "mape": 0.0}

    pred_arr = np.asarray(preds, dtype=np.float64)
    true_arr = np.asarray(trues, dtype=np.float64)
    abs_err = np.abs(pred_arr - true_arr)
    mae = float(np.mean(abs_err))
    rmse = float(np.sqrt(np.mean((pred_arr - true_arr) ** 2)))
    nz = np.abs(true_arr) > 1e-6
    mape = float(np.mean(abs_err[nz] / np.abs(true_arr[nz])) * 100.0) if np.any(nz) else 0.0

    return {
        "loss": total_loss / n,
        "density_loss": total_density / n,
        "species_loss": total_species / n,
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
    }


def evaluate_regression_on_loader(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    preds: List[float] = []
    trues: List[float] = []
    with torch.no_grad():
        for x, d_true, _y_true in loader:
            x = x.to(device)
            d_true = d_true.to(device)
            d_pred, _ = model(x)
            preds.extend(d_pred.detach().cpu().numpy().reshape(-1).tolist())
            trues.extend(d_true.detach().cpu().numpy().reshape(-1).tolist())

    if not trues:
        return {"count": 0, "mae": 0.0, "rmse": 0.0, "mape": 0.0, "r2": 0.0}

    pred_arr = np.asarray(preds, dtype=np.float64)
    true_arr = np.asarray(trues, dtype=np.float64)
    abs_err = np.abs(pred_arr - true_arr)
    mae = float(np.mean(abs_err))
    rmse = float(np.sqrt(np.mean((pred_arr - true_arr) ** 2)))
    nz = np.abs(true_arr) > 1e-6
    mape = float(np.mean(abs_err[nz] / np.abs(true_arr[nz])) * 100.0) if np.any(nz) else 0.0
    sst = float(np.sum((true_arr - np.mean(true_arr)) ** 2))
    sse = float(np.sum((pred_arr - true_arr) ** 2))
    r2 = float(1.0 - (sse / sst)) if sst > 1e-9 else 0.0
    return {"count": int(len(true_arr)), "mae": mae, "rmse": rmse, "mape": mape, "r2": r2}


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
    p.add_argument("--density-target-mode", choices=["normalized", "tph"], default="tph")
    p.add_argument("--tph-reference", type=float, default=1000.0)
    p.add_argument("--density-loss-kind", choices=["huber", "mse", "l1"], default="huber")
    p.add_argument("--holdout-csv", default="", help="Optional holdout CSV (same schema as training mode) for final raw metric report.")

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
        dataset = LocalPairedDataset(
            args.dataset_path,
            args.labels_csv,
            density_target_mode=args.density_target_mode,
            tph_reference=args.tph_reference,
        )
    else:
        if not args.aoi_samples_csv:
            raise ValueError("aoi mode requires --aoi-samples-csv")
        dataset = AOISampleDataset(
            samples_csv=args.aoi_samples_csv,
            default_radius_km=args.aoi_default_radius_km,
            default_cloud_cover_max=args.aoi_default_cloud_cover_max,
            cache_dir=args.aoi_cache_dir or None,
            density_target_mode=args.density_target_mode,
            tph_reference=args.tph_reference,
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
    model = TreeSatMultiHeadModelV2(density_mode=args.density_target_mode).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    holdout_loader = None
    if args.holdout_csv:
        if args.mode == "local":
            holdout_ds = LocalPairedDataset(
                args.dataset_path,
                args.holdout_csv,
                density_target_mode=args.density_target_mode,
                tph_reference=args.tph_reference,
            )
        else:
            holdout_ds = AOISampleDataset(
                samples_csv=args.holdout_csv,
                default_radius_km=args.aoi_default_radius_km,
                default_cloud_cover_max=args.aoi_default_cloud_cover_max,
                cache_dir=args.aoi_cache_dir or None,
                density_target_mode=args.density_target_mode,
                tph_reference=args.tph_reference,
            )
        holdout_loader = DataLoader(
            holdout_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_batch,
        )

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
            density_loss_kind=args.density_loss_kind,
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            device,
            optimizer=None,
            density_loss_weight=args.density_loss_weight,
            species_loss_weight=args.species_loss_weight,
            density_loss_kind=args.density_loss_kind,
        )

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_density_loss": train_metrics["density_loss"],
            "train_species_loss": train_metrics["species_loss"],
            "train_mae": train_metrics["mae"],
            "train_rmse": train_metrics["rmse"],
            "train_mape": train_metrics["mape"],
            "val_loss": val_metrics["loss"],
            "val_density_loss": val_metrics["density_loss"],
            "val_species_loss": val_metrics["species_loss"],
            "val_mae": val_metrics["mae"],
            "val_rmse": val_metrics["rmse"],
            "val_mape": val_metrics["mape"],
            "seconds": round(time.time() - t0, 2),
        }
        history_rows.append(row)

        print(
            f"[Epoch {epoch:03d}] "
            f"train={row['train_loss']:.6f} "
            f"val={row['val_loss']:.6f} "
            f"(density={row['val_density_loss']:.6f}, species={row['val_species_loss']:.6f}, "
            f"val_mae={row['val_mae']:.4f}, val_mape={row['val_mape']:.2f}%)"
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
            "density_target_mode": args.density_target_mode,
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

    holdout_summary = None
    if holdout_loader is not None:
        best_ckpt = torch.load(best_path, map_location=device, weights_only=False)
        if isinstance(best_ckpt, dict) and "model_state_dict" in best_ckpt:
            model.load_state_dict(best_ckpt["model_state_dict"], strict=False)
        holdout_summary = evaluate_regression_on_loader(model, holdout_loader, device)
        holdout_summary["target_mode"] = args.density_target_mode
        holdout_summary["metric_units"] = "tph" if args.density_target_mode == "tph" else "normalized_density"
        holdout_path = os.path.join(args.output_dir, "holdout_metrics.json")
        with open(holdout_path, "w", encoding="utf-8") as f:
            json.dump(holdout_summary, f, indent=2)
        print(f"Holdout metrics: {holdout_path}")
        print(json.dumps(holdout_summary, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
