"""
Offline model evaluation and threshold tuning utilities.

Expected ground-truth CSV schema (minimum):
- filename
- density (or tree_density) [optional but recommended]
- species labels as either:
    - species_<label> columns (e.g., species_Abies), or
    - direct label columns matching SPECIES_LABELS (e.g., Abies)

Species columns are binary (0/1).
"""

import csv
import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .inference import discover_dataset, run_inference_paired
from .model_loader import SPECIES_LABELS


DENSITY_KEYS = ["density", "tree_density", "density_target", "tree_density_target"]


@dataclass
class GroundTruthRow:
    filename: str
    density: Optional[float]
    species: np.ndarray


def _norm_key(text: str) -> str:
    return "".join(ch.lower() for ch in str(text) if ch.isalnum())


def _parse_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    sval = str(value).strip()
    if sval == "":
        return None
    try:
        return float(sval)
    except ValueError:
        return None


def _parse_binary(value: Any) -> int:
    if value is None:
        return 0
    sval = str(value).strip().lower()
    if sval in {"1", "true", "t", "yes", "y"}:
        return 1
    try:
        return 1 if float(sval) >= 0.5 else 0
    except ValueError:
        return 0


def _find_density_value(row: Dict[str, Any]) -> Optional[float]:
    for key in DENSITY_KEYS:
        if key in row:
            return _parse_float(row[key])
    return None


def _build_species_column_map(headers: List[str]) -> Dict[str, str]:
    """
    Map each species label to its corresponding CSV column if present.
    Accepts both `species_<label>` and `<label>` variants.
    """
    normalized_headers = {_norm_key(h): h for h in headers}

    mapping: Dict[str, str] = {}
    for sp in SPECIES_LABELS:
        candidates = [f"species_{sp}", sp]
        for cand in candidates:
            nk = _norm_key(cand)
            if nk in normalized_headers:
                mapping[sp] = normalized_headers[nk]
                break
    return mapping


def load_ground_truth_csv(csv_path: str) -> Dict[str, GroundTruthRow]:
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Ground-truth CSV not found: {csv_path}")

    rows: Dict[str, GroundTruthRow] = {}

    with open(csv_path, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("Ground-truth CSV has no header row")

        headers = [h.strip() for h in reader.fieldnames if h is not None]
        lower_map = {h.lower().strip(): h for h in headers}

        if "filename" not in lower_map:
            raise ValueError("Ground-truth CSV must contain a 'filename' column")

        filename_col = lower_map["filename"]
        species_col_map = _build_species_column_map(headers)
        if not species_col_map:
            raise ValueError(
                "No species columns found. Provide either species_<label> columns "
                "or direct label columns matching model species names."
            )

        for raw in reader:
            normalized = {(k.strip() if isinstance(k, str) else k): v for k, v in raw.items()}
            filename = str(normalized.get(filename_col, "")).strip()
            if not filename:
                continue

            density = _find_density_value({k.lower(): v for k, v in normalized.items() if isinstance(k, str)})

            species_vec = np.zeros(len(SPECIES_LABELS), dtype=np.int64)
            for i, sp in enumerate(SPECIES_LABELS):
                col = species_col_map.get(sp)
                if col is not None:
                    species_vec[i] = _parse_binary(normalized.get(col))

            rows[filename] = GroundTruthRow(filename=filename, density=density, species=species_vec)

    if not rows:
        raise ValueError("Ground-truth CSV has no usable rows")

    return rows


def _density_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    if y_true.size == 0:
        return {}

    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))

    sst = float(np.sum((y_true - np.mean(y_true)) ** 2))
    sse = float(np.sum(err ** 2))
    r2 = 1.0 - (sse / sst) if sst > 0 else None

    mape_mask = np.abs(y_true) > 1e-8
    mape = float(np.mean(np.abs(err[mape_mask] / y_true[mape_mask])) * 100.0) if np.any(mape_mask) else None

    return {
        "count": int(y_true.size),
        "mae": round(mae, 6),
        "rmse": round(rmse, 6),
        "r2": round(r2, 6) if r2 is not None else None,
        "mape_percent": round(mape, 4) if mape is not None else None,
    }


def _prf_counts(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[int, int, int, int]:
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    return tp, fp, fn, tn


def _precision_recall_f1(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def _average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Computes average precision (area under precision-recall curve) using ranking.
    """
    positives = int(np.sum(y_true == 1))
    if positives == 0:
        return 0.0

    order = np.argsort(-y_score)
    y_sorted = y_true[order]

    tp = 0
    fp = 0
    precisions = []

    for val in y_sorted:
        if val == 1:
            tp += 1
            precisions.append(tp / (tp + fp))
        else:
            fp += 1

    return float(np.sum(precisions) / positives) if precisions else 0.0


def _evaluate_species(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: np.ndarray,
) -> Dict[str, Any]:
    n_classes = y_true.shape[1]

    per_class = []
    macro_precisions = []
    macro_recalls = []
    macro_f1s = []
    aps = []

    total_tp = total_fp = total_fn = 0

    for k in range(n_classes):
        thr = float(thresholds[k])
        yt = y_true[:, k]
        yp = (y_prob[:, k] >= thr).astype(np.int64)
        tp, fp, fn, tn = _prf_counts(yt, yp)
        precision, recall, f1 = _precision_recall_f1(tp, fp, fn)
        ap = _average_precision(yt, y_prob[:, k])

        macro_precisions.append(precision)
        macro_recalls.append(recall)
        macro_f1s.append(f1)
        aps.append(ap)

        total_tp += tp
        total_fp += fp
        total_fn += fn

        per_class.append(
            {
                "species": SPECIES_LABELS[k],
                "threshold": round(thr, 4),
                "support": int(np.sum(yt == 1)),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "precision": round(precision, 6),
                "recall": round(recall, 6),
                "f1": round(f1, 6),
                "average_precision": round(ap, 6),
            }
        )

    micro_precision, micro_recall, micro_f1 = _precision_recall_f1(total_tp, total_fp, total_fn)

    return {
        "per_class": per_class,
        "macro": {
            "precision": round(float(np.mean(macro_precisions)), 6),
            "recall": round(float(np.mean(macro_recalls)), 6),
            "f1": round(float(np.mean(macro_f1s)), 6),
            "average_precision": round(float(np.mean(aps)), 6),
        },
        "micro": {
            "precision": round(micro_precision, 6),
            "recall": round(micro_recall, 6),
            "f1": round(micro_f1, 6),
        },
    }


def _tune_thresholds(y_true: np.ndarray, y_prob: np.ndarray, grid: np.ndarray) -> np.ndarray:
    n_classes = y_true.shape[1]
    best = np.full(n_classes, 0.5, dtype=np.float64)

    for k in range(n_classes):
        yt = y_true[:, k]
        best_f1 = -1.0
        best_thr = 0.5

        for thr in grid:
            yp = (y_prob[:, k] >= thr).astype(np.int64)
            tp, fp, fn, _ = _prf_counts(yt, yp)
            _, _, f1 = _precision_recall_f1(tp, fp, fn)
            if f1 > best_f1:
                best_f1 = f1
                best_thr = float(thr)

        best[k] = best_thr

    return best


def evaluate_offline(
    dataset_path: str,
    ground_truth_csv: str,
    species_threshold: float = 0.5,
    threshold_grid: Optional[List[float]] = None,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run full offline evaluation against ground-truth labels.
    Produces report + CSV artifacts under output_dir.
    """
    if threshold_grid is None:
        threshold_grid = [round(x, 2) for x in np.arange(0.1, 0.95, 0.05)]

    grid = np.array(threshold_grid, dtype=np.float64)

    gt = load_ground_truth_csv(ground_truth_csv)
    pairs = discover_dataset(dataset_path)

    by_filename = {os.path.basename(s2): (s2, s1) for s2, s1 in pairs}

    matched_files = sorted(set(gt.keys()) & set(by_filename.keys()))
    missing_in_dataset = sorted(set(gt.keys()) - set(by_filename.keys()))

    if not matched_files:
        raise ValueError(
            "No overlap between ground-truth filenames and dataset S2 filenames. "
            "Ensure GT 'filename' values match tile names (e.g., tile_0000.tif)."
        )

    density_true: List[float] = []
    density_pred: List[float] = []
    y_true_species: List[np.ndarray] = []
    y_prob_species: List[np.ndarray] = []
    prediction_rows: List[Dict[str, Any]] = []

    for fname in matched_files:
        s2_path, s1_path = by_filename[fname]
        pred = run_inference_paired(s2_path, s1_path, species_threshold=species_threshold)

        prob_by_species = {d["species"]: float(d["probability"]) for d in pred["species_distribution"]}
        prob_vec = np.array([prob_by_species.get(sp, 0.0) for sp in SPECIES_LABELS], dtype=np.float64)

        truth = gt[fname]
        y_true_species.append(truth.species.astype(np.int64))
        y_prob_species.append(prob_vec)

        if truth.density is not None:
            density_true.append(float(truth.density))
            density_pred.append(float(pred["density"]))

        row: Dict[str, Any] = {
            "filename": fname,
            "pred_density": float(pred["density"]),
            "true_density": truth.density,
        }

        for i, sp in enumerate(SPECIES_LABELS):
            row[f"pred_prob_{sp}"] = round(float(prob_vec[i]), 6)
            row[f"true_{sp}"] = int(truth.species[i])

        prediction_rows.append(row)

    y_true = np.vstack(y_true_species)
    y_prob = np.vstack(y_prob_species)

    base_thresholds = np.full(len(SPECIES_LABELS), float(species_threshold), dtype=np.float64)
    tuned_thresholds = _tune_thresholds(y_true, y_prob, grid)

    species_metrics_default = _evaluate_species(y_true, y_prob, base_thresholds)
    species_metrics_tuned = _evaluate_species(y_true, y_prob, tuned_thresholds)

    density_metrics = _density_metrics(np.array(density_true), np.array(density_pred))

    now = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), "outputs", f"evaluation_{now}")
    os.makedirs(output_dir, exist_ok=True)

    report = {
        "dataset_path": os.path.abspath(dataset_path),
        "ground_truth_csv": os.path.abspath(ground_truth_csv),
        "evaluated_at_utc": datetime.utcnow().isoformat() + "Z",
        "tile_matching": {
            "dataset_pairs": len(pairs),
            "ground_truth_rows": len(gt),
            "matched": len(matched_files),
            "missing_ground_truth_in_dataset": missing_in_dataset,
        },
        "density_metrics": density_metrics,
        "species_metrics_default_threshold": species_metrics_default,
        "species_metrics_tuned_threshold": species_metrics_tuned,
        "tuned_thresholds": {
            sp: round(float(th), 4) for sp, th in zip(SPECIES_LABELS, tuned_thresholds)
        },
    }

    report_path = os.path.join(output_dir, "evaluation_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    predictions_csv = os.path.join(output_dir, "per_tile_predictions.csv")
    with open(predictions_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = list(prediction_rows[0].keys()) if prediction_rows else []
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(prediction_rows)

    thresholds_csv = os.path.join(output_dir, "tuned_thresholds.csv")
    with open(thresholds_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["species", "threshold"])
        for sp, th in zip(SPECIES_LABELS, tuned_thresholds):
            writer.writerow([sp, round(float(th), 4)])

    return {
        "status": "success",
        "output_dir": output_dir,
        "report_path": report_path,
        "predictions_csv": predictions_csv,
        "thresholds_csv": thresholds_csv,
        "summary": {
            "matched_tiles": len(matched_files),
            "density_rmse": density_metrics.get("rmse"),
            "density_mae": density_metrics.get("mae"),
            "species_macro_f1_default": species_metrics_default["macro"]["f1"],
            "species_macro_f1_tuned": species_metrics_tuned["macro"]["f1"],
        },
    }
