#!/usr/bin/env python3
"""
Adaptive Runtime Safeguard — Stress Test Runner
================================================

Validates the zero-crash inference guarantee of DualTower-FT's Adaptive
Runtime Safeguard against four classes of malformed inputs:

    C0  baseline             — untouched preprocessed test tensor
    C1  cat_oor              — categorical out-of-range index (age = 200)
    C2  num_extreme          — numerical extreme value (lab × 10× train max)
    C3  nan_inject           — NaN injected after preprocessing
    C4  sentinel             — eICU missing-value sentinel (-1) in a numeric column

For each condition we run the trained DualTower-FT in two states:
    safeguard=True   — the model's inference-time Adaptive Runtime Safeguard
                       is active (three stages: NaN/Inf guard, categorical
                       clamp, ±10 numerical z-score clamp). Implemented in
                       DualTowerFT._compute_towers; gated by the attribute
                       ``model.safeguard_enabled``.
    safeguard=False  — all three stages are bypassed. Cat OOR triggers
                       IndexError on the Embedding lookup; NaN propagates
                       to the output; extreme numerical values are not
                       capped.

Recorded per (condition, safeguard) combination:
    crashes_pct       — % of batches that raised IndexError/RuntimeError
    nan_pred_pct      — % of patients whose final probability is NaN
    auprc             — average_precision on patients with a valid prediction
    latency_ms_mean   — mean per-batch forward latency (first 20 batches)
    latency_ms_std    — SD of the same

Usage
-----
    python run_safeguard_stress_test.py \\
        --model_dir ../../outputs/final_model/DUALTOWER_FT_patient_stratified_seed42_auprc0626 \\
        --data_path ../../data/train_ready/test_patient_stratified.csv \\
        --output ../../outputs/safeguard_validation/seed42.json \\
        --rate 0.01

Limitation (mentioned in paper §3.1.3):
    The safeguard does not detect inputs that are within bounds but
    semantically wrong (e.g., a glucose value entered in the wrong unit
    but still within plausible numeric range). That is a data-validation
    concern, not a runtime-safety concern.

Conventions:
    - Model loaded from ``model_dir/model_weights.pkl`` (full pickled model)
    - Scaler loaded from ``model_dir/scaler.pkl`` (sklearn StandardScaler)
    - Feature names from ``model_dir/feature_list.json``
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score


# ─── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─── Constants ──────────────────────────────────────────────────────────────
CONDITIONS = ["baseline", "cat_oor", "num_extreme", "nan_inject", "sentinel"]
SAFEGUARD_STATES = [True, False]

# Anomaly target columns (must exist in feature_list.json — script falls back
# to the first available column of matching type if any of these are absent).
TARGET_COLS = {
    "cat_oor":     "age",
    "num_extreme": "lab_glucose_mean",
    "nan_inject":  "sao2_mean",
    "sentinel":    "lab_BUN_mean",
}


# ─── Model loading ──────────────────────────────────────────────────────────
def load_model(model_dir: Path, device: str) -> torch.nn.Module:
    """Load the pickled DualTowerFT model."""
    pkl_path = model_dir / "model_weights.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(f"model_weights.pkl not found in {model_dir}")
    logger.info(f"Loading model: {pkl_path}")
    with open(pkl_path, "rb") as f:
        model = pickle.load(f)
    model = model.to(device)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  {type(model).__name__}: {n_params:,} params")
    return model


def load_data(
    data_path: Path,
    model_dir: Path,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load eICU test set and apply the saved scaler — returns (X, y, names)."""
    logger.info(f"Loading test data: {data_path}")
    df = pd.read_csv(data_path)

    with open(model_dir / "feature_list.json") as f:
        feature_info = json.load(f)
    feature_names = feature_info["features"]

    target_col = "mortality"
    y = df[target_col].values.astype(np.float32)

    drop_cols = ["patient_id", "hospital_id", "unit_id", target_col]
    for col in drop_cols:
        if col in df.columns:
            df = df.drop(columns=[col])

    missing = set(feature_names) - set(df.columns)
    if missing:
        logger.warning(f"  Missing features (zero-filled): {missing}")
        for col in missing:
            df[col] = 0.0

    X_raw = df[feature_names].values.astype(np.float32)

    scaler_path = model_dir / "scaler.pkl"
    if scaler_path.exists():
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        X = scaler.transform(X_raw).astype(np.float32)
        logger.info(f"  Applied {type(scaler).__name__}")
    else:
        X = X_raw
        logger.warning("  No scaler.pkl — using raw features")

    logger.info(f"  X={X.shape}, y prevalence={y.mean():.3f}")
    return X, y, feature_names


# ─── Safeguard toggle (attribute flip) ──────────────────────────────────────
def patch_safeguard(model: torch.nn.Module, enabled: bool) -> torch.nn.Module:
    """
    Toggle the inference-time Adaptive Runtime Safeguard.

    The safeguard logic lives inside ``DualTowerFT._compute_towers`` and is
    controlled by ``model.safeguard_enabled`` (see dualtower_ft.py for the
    three stages: NaN/Inf guard, categorical clamp, numerical clamp).

    Setting ``enabled=False`` bypasses all three stages, which lets the
    stress test observe what fails when the safeguard is absent.
    """
    model.safeguard_enabled = bool(enabled)
    return model


# ─── Stress injection ───────────────────────────────────────────────────────
def find_col_index(feature_names: List[str], col_name: str) -> int:
    """Locate column index; fall back to first column starting with prefix."""
    if col_name in feature_names:
        return feature_names.index(col_name)
    # Fallback: prefix match (e.g., 'lab_BUN' might appear as 'lab_BUN_mean')
    for i, name in enumerate(feature_names):
        if name.startswith(col_name.split("_")[0]):
            logger.warning(
                f"  Column '{col_name}' not found; using fallback "
                f"'{name}' (index {i})"
            )
            return i
    raise KeyError(f"No suitable column for '{col_name}' in feature_list")


def inject(
    X: np.ndarray,
    condition: str,
    rate: float,
    rng: np.random.Generator,
    col_idx: Dict[str, int],
) -> np.ndarray:
    """Return a copy of X with the requested anomaly applied to ``rate`` of rows."""
    if condition == "baseline":
        return X.copy()

    X_out = X.copy()
    n_rows = int(len(X) * rate)
    if n_rows == 0:
        logger.warning(f"  rate {rate} → 0 rows; condition '{condition}' is no-op")
        return X_out

    idx = rng.choice(len(X), n_rows, replace=False)

    if condition == "cat_oor":
        # Age vocabulary is [0, 149]; 200 is OOR
        X_out[idx, col_idx["age"]] = 200.0
    elif condition == "num_extreme":
        col = col_idx["lab_glucose_mean"]
        max_val = float(np.nanmax(X[:, col]))
        # 10× the observed maximum
        X_out[idx, col] = max_val * 10.0 if max_val != 0 else 1e6
    elif condition == "nan_inject":
        X_out[idx, col_idx["sao2_mean"]] = np.nan
    elif condition == "sentinel":
        # eICU missing-value sentinel (after scaling this won't be exactly -1,
        # but we inject -1 to simulate raw inputs bypassing preprocessing)
        X_out[idx, col_idx["lab_BUN_mean"]] = -1.0
    else:
        raise ValueError(f"Unknown condition: {condition}")

    return X_out


# ─── Evaluation ─────────────────────────────────────────────────────────────
def evaluate(
    model: torch.nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    device: str,
    batch_size: int = 512,
    n_latency_batches: int = 20,
) -> Dict[str, float]:
    """Run forward pass; record crashes / NaN / AUPRC / latency.

    IMPORTANT: when the safeguard is disabled (``model.safeguard_enabled is
    False``), an out-of-range categorical index triggers a device-side
    CUDA assertion that poisons the entire CUDA context. Subsequent CUDA
    operations all fail and the process becomes unusable. To collect the
    "without-safeguard" data point cleanly we transparently fall back to
    CPU for that case; CPU's ``IndexError`` is synchronous and recoverable.
    """
    model.eval()
    n_total = len(X)
    n_batches = (n_total + batch_size - 1) // batch_size

    # Switch to CPU when the safeguard is off, to keep exceptions catchable.
    safeguard_on = bool(getattr(model, "safeguard_enabled", True))
    requested_cuda = device.startswith("cuda") and torch.cuda.is_available()
    actual_device = device if (safeguard_on and requested_cuda) else "cpu"
    if actual_device != device:
        logger.info(
            f"  safeguard=False -> running this cell on CPU "
            f"(GPU CUDA-asserts would poison the context for later cells)"
        )
    model = model.to(actual_device)

    crashes = 0
    preds = np.full(n_total, np.nan, dtype=np.float32)
    latencies: List[float] = []

    X_tensor = torch.from_numpy(X).float()
    use_cuda = actual_device.startswith("cuda")

    for batch_idx in range(n_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, n_total)
        batch_X = X_tensor[start:end].to(actual_device)

        try:
            with torch.no_grad():
                measure = batch_idx < n_latency_batches
                if measure and use_cuda:
                    torch.cuda.synchronize()
                t0 = time.perf_counter()

                logits = model(batch_X)

                if measure and use_cuda:
                    torch.cuda.synchronize()
                t1 = time.perf_counter()
                if measure:
                    latencies.append((t1 - t0) * 1000.0)

                probs = torch.sigmoid(logits).cpu().numpy()
                preds[start:end] = probs

        except (IndexError, RuntimeError, ValueError) as e:
            crashes += 1
            # Preds for this batch stay as NaN

    crashes_pct = 100.0 * crashes / n_batches
    nan_pct = 100.0 * np.isnan(preds).mean()
    valid_mask = ~np.isnan(preds)

    if valid_mask.sum() > 0 and len(set(y[valid_mask].astype(int))) > 1:
        auprc = float(average_precision_score(y[valid_mask], preds[valid_mask]))
    else:
        auprc = float("nan")

    return {
        "crashes_pct":      float(crashes_pct),
        "nan_pred_pct":     float(nan_pct),
        "auprc":            auprc if not np.isnan(auprc) else None,
        "latency_ms_mean":  float(np.mean(latencies)) if latencies else None,
        "latency_ms_std":   float(np.std(latencies)) if latencies else None,
        "n_total":          int(n_total),
        "n_valid_preds":    int(valid_mask.sum()),
        "n_batches":        int(n_batches),
        "n_crashed_batches": int(crashes),
        "device_used":      actual_device,
        "safeguard_state":  safeguard_on,
    }


# ─── Pretty printer ─────────────────────────────────────────────────────────
def print_table(results: Dict[str, Dict[str, float]]) -> None:
    """Console table of the 5 × 2 matrix for at-a-glance reading."""
    print()
    print("=" * 90)
    print(f"{'Condition':<16} {'Safeguard':<10} {'Crashes %':>10} "
          f"{'NaN preds %':>12} {'AUPRC':>8} {'Latency ms':>14}")
    print("-" * 90)
    for cond in CONDITIONS:
        for sg in SAFEGUARD_STATES:
            key = f"{cond}_sg_{sg}"
            if key not in results:
                continue
            r = results[key]
            auprc_str = f"{r['auprc']:.4f}" if r["auprc"] is not None else "  N/A "
            lat = r["latency_ms_mean"]
            lat_str = f"{lat:.2f} ± {r['latency_ms_std']:.2f}" if lat is not None else "  N/A"
            print(f"{cond:<16} {'ON' if sg else 'OFF':<10} "
                  f"{r['crashes_pct']:>10.1f} {r['nan_pred_pct']:>12.2f} "
                  f"{auprc_str:>8} {lat_str:>14}")
    print("=" * 90)


# ─── Main ───────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model_dir", type=Path, required=True,
                        help="Trained model directory (contains model_weights.pkl)")
    parser.add_argument("--data_path", type=Path, required=True,
                        help="Test set CSV (e.g., test_patient_stratified.csv)")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output JSON path")
    parser.add_argument("--rate", type=float, default=0.01,
                        help="Anomaly injection rate (fraction of test rows). Default 0.01")
    parser.add_argument("--injection_seed", type=int, default=42,
                        help="RNG seed for anomaly placement (not the model seed). Default 42")
    parser.add_argument("--batch_size", type=int, default=512,
                        help="Inference batch size. Default 512")
    parser.add_argument("--device", default="cuda",
                        help="Torch device. Default 'cuda'; falls back to CPU if unavailable")
    args = parser.parse_args()

    # Resolve device
    device = args.device if (
        args.device == "cpu" or torch.cuda.is_available()
    ) else "cpu"
    if device != args.device:
        logger.warning(f"CUDA unavailable; falling back to CPU")

    logger.info(f"Run config: rate={args.rate}, injection_seed={args.injection_seed}, "
                f"batch_size={args.batch_size}, device={device}")

    # Load
    model = load_model(args.model_dir, device)
    X, y, feature_names = load_data(args.data_path, args.model_dir)

    # Resolve target column indices
    col_idx = {
        name: find_col_index(feature_names, name)
        for name in set(TARGET_COLS.values())
    }
    logger.info(f"Target columns: {col_idx}")

    # Run the 5 × 2 matrix
    rng = np.random.default_rng(args.injection_seed)
    results: Dict[str, Dict[str, float]] = {}

    for cond in CONDITIONS:
        X_stress = inject(X, cond, args.rate, rng, col_idx)
        for sg in SAFEGUARD_STATES:
            key = f"{cond}_sg_{sg}"
            logger.info(f"==> {key}")
            patch_safeguard(model, enabled=sg)
            results[key] = evaluate(
                model, X_stress, y, device,
                batch_size=args.batch_size,
            )
            r = results[key]
            logger.info(
                f"    crashes={r['crashes_pct']:.1f}% nan={r['nan_pred_pct']:.2f}% "
                f"auprc={r['auprc']} latency={r['latency_ms_mean']}"
            )

    # Save + summary
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_payload = {
        "config": {
            "model_dir": str(args.model_dir),
            "data_path": str(args.data_path),
            "rate": args.rate,
            "injection_seed": args.injection_seed,
            "batch_size": args.batch_size,
            "device": device,
        },
        "feature_columns_used": col_idx,
        "results": results,
    }
    with open(args.output, "w") as f:
        json.dump(output_payload, f, indent=2)
    logger.info(f"Saved: {args.output}")
    print_table(results)


if __name__ == "__main__":
    main()
