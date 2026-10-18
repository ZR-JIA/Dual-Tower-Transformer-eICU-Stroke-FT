# DualTower-FT with an Adaptive Runtime Safeguard: A Deep Tabular Approach for ICU Stroke Mortality

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)

> [!NOTE]
> Some descriptions in the PRICAI 2026 paper differ from the experiments as run. See [Clarifications to the paper](#clarifications-to-the-paper). All reported numbers are unaffected.

This repository extends [Dual-Tower-Transformer-eICU-Stroke](https://github.com/ZR-JIA/Dual-Tower-Transformer-eICU-Stroke), the code release of our CCAI 2026 paper, and is built on that codebase. The original Dual-Tower model and the MLP and Transformer baselines are unchanged (`dualtower.py`, `mlp.py`, `transformer.py`, `base.py`). The main additions are DualTower-FT with the extended runtime safeguard (`dualtower_ft.py`), the safeguard stress test (`run_safeguard_stress_test.py`), two-stage MIMIC-IV transfer (`run_two_stage_train.py`), and the DT-CA, DT-AS and DT-DC ablation variants. The data pipeline, trainers, multi-seed runner and Optuna search spaces were also modified. The explainability module of the earlier repository is not included.

> **⚠️ IMPORTANT: Medical AI Disclaimer**
> This model is for **Academic Research Use Only**. It is **NOT** an FDA-approved medical device and must **NOT** be used for clinical diagnosis, treatment decisions, or patient care. Commercial use is strictly prohibited. See [LICENSE](LICENSE) for details.

---

## Abstract

Multi-center ICU electronic health records are noisy, and deep tabular models trained on such data often underperform tree ensembles while crashing at inference on out-of-range inputs. We propose DualTower-FT, a dual-tower architecture with FT-Transformer-style [CLS] aggregation on the numerical tower, paired with an Adaptive Runtime Safeguard that clamps categorical indices, bounds numerical extremes, and neutralises NaN/±∞ inside the forward pass. We evaluate on 11,187 patients from 176 eICU hospitals with five-seed reproducibility and a MIMIC-IV transfer test. DualTower-FT reaches AUPRC 0.6247 ± 0.0015 on the eICU test cohort (N = 2,238): the best deep learning result, with four-fold lower cross-seed variance than the prior dual-tower baseline, while XGBoost retains the overall lead (0.6467). No inference failure occurs in any stress batch across four injection conditions; the safeguard keeps AUPRC within ±0.0012 of baseline, whereas every batch raises an exception when it is disabled, including the unperturbed baseline. None of the four MIMIC-IV transfer strategies improved on eICU-only training; missing feature coverage and MIMIC-IV's single-centre nature likely explain the gap. We contribute a runtime-safe deep tabular pipeline for noisy multi-center ICU data and empirical evidence on the limits of naive cross-database transfer.

*The abstract is reproduced as published. See [Clarifications to the paper](#clarifications-to-the-paper) for points where the paper's description differs from the experiments as run.*

**Paper**: Z. Jia and K.-C. Wong. DualTower-FT with an Adaptive Runtime Safeguard: A Deep Tabular Approach for ICU Stroke Mortality. PRICAI 2026, Springer LNAI.

---

## Key Features

- **Dual-Tower Architecture** — Decoupled processing of categorical demographics (Left Tower: learnable embeddings) and numerical vitals (Right Tower: configurable MLP or Transformer encoder)
- **NumericalFeatureTokenizer** — Converts scalar physiological features into dense token embeddings for self-attention, enabling high-order feature interaction capture
- **Adaptive Runtime Safeguard** — Inference-time stability mechanism against physiological outliers
- **Hierarchical Configuration System** — YAML-based config with `_base/` global defaults and per-model overrides, supporting `${variable}` resolution
- **Unified Training Engine** — Single entry point supporting multiple model types: `DualTower-FT` (proposed), `DualTower (Transformer)`, `DualTower (MLP)`, `DualTower-CA`, `Transformer`, `MLP`, `NN`, `XGBoost`, `RandomForest`
- **Two-Stage Transfer Learning** — Combined (eICU+MIMIC) pretraining followed by eICU-only fine-tuning, the standard cross-dataset transfer paradigm in medical AI
- **Cross-Platform Device Support** — Automatic device detection with priority `CUDA → MPS → CPU`; runs natively on NVIDIA GPUs, Apple Silicon (M1/M2/M3), and CPU
- **Multi-Seed Reproducibility** — Academic-grade multi-seed experiment runner with aggregated mean ± std reporting
- **Optuna Hyperparameter Optimization** — Integrated Bayesian search with configurable search spaces
- **Comprehensive Evaluation** — AUROC, AUPRC, Brier score, ECE, bootstrap confidence intervals, Youden threshold selection, and isotonic calibration

---

## Adaptive Runtime Safeguard

The **Adaptive Runtime Safeguard** is the engineering pipeline that lets DualTower-FT survive noisy multi-center ICU inputs without crashing. It has **two defensive layers** (preprocessing-time and inference-time) plus a **training-time complement**, providing a *zero-crash inference guarantee* under malformed inputs (out-of-range categorical indices, NaN/Inf, numerical extremes).

### Layer 1 — Preprocessing-time (one-shot, in `TabularPreprocessor.transform`)

| Stage | Mechanism | Implementation |
|---|---|---|
| NaN imputation | `X.fillna(self.fill_values)` with training median (default) | `preprocessor.py` `_fit_impute_missing` / `_transform_impute_missing` |
| Categorical validation | clamp age ∈ [0, 149], gender ∈ [0, 9], ethnicity ∈ [0, 49] | `preprocessor.py` `_validate_categoricals` |
| NaN runtime check | raise `RuntimeError` if any NaN remains | `preprocessor.py` `_verify_no_nans` |
| Optional percentile cap | 0.1th / 99.9th percentile on training distribution | `preprocessor.py` `_fit_clip_outliers` / `_transform_clip_outliers` |

The optional cap is controlled by `src/train/config/_base/01_data.yaml`:

```yaml
data:
  features:
    clip:
      enabled: false      # disabled by default; inference layer also bounds numerics
      p_low: 0.001        # lower percentile
      p_high: 0.999       # upper percentile
```

### Layer 2 — Inference-time (every forward pass, in `DualTowerFT._compute_towers`)

| Stage | Mechanism | Behaviour |
|---|---|---|
| A. NaN / Inf guard | `torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)` | replaces NaN / ±∞ with 0 (post-scaling neutral); catches downstream NaN that bypasses preprocessing |
| B. Categorical clamp | `torch.clamp(x, min=0, max=num_classes-1)` | bounds each categorical column to its embedding vocabulary range |
| C. Numerical clamp | `torch.clamp(x_num, min=-10, max=10)` | bounds numerical columns to ±10 z-score units (legitimate physiological values fall well inside ±5) |

All three stages are gated by `model.safeguard_enabled` (default `True`). Set it to `False` to bypass — used by the stress-test harness below.

```python
# Toggle the inference-time safeguard
model.safeguard_enabled = False   # bypass all three stages (used in stress tests only)
model.safeguard_enabled = True    # default: all three active
```

**Implementation**: `src/train/architectures/models/dualtower_ft.py` → `DualTowerFT._compute_towers` (the toggle attribute is set in `__init__`).

### Training-time complement

Gradient norms are clipped to `1.0` via `torch.nn.utils.clip_grad_norm_`, enabled by default. This bounds single-batch parameter updates and prevents extreme inputs from destabilising training. Configuration in `src/train/config/_base/02_train.yaml`:

```yaml
training:
  gradient_clip:
    enabled: true
    norm: 1.0
    algorithm: norm
```

**Implementation**: `src/train/engine/trainers.py` — called between `loss.backward()` and `optimizer.step()`.

### Stress Test — empirical validation

The safeguard's *zero-crash inference* claim is validated by an injection-based stress test. The harness lives at `src/train/run_safeguard_stress_test.py` and produces a 5 × 2 matrix (5 conditions × {safeguard ON, safeguard OFF}) reporting per-batch crash rate, NaN prediction rate, AUPRC, and per-batch latency.

**Stress conditions** (1 % of test rows perturbed per condition by default):

| Condition | Injection | Without safeguard | With safeguard |
|---|---|---|---|
| `baseline` | none | normal AUPRC | normal AUPRC |
| `cat_oor` | `age` ← 200 (a numerical column; labelled "categorical OOR" in the paper) | AUPRC degraded | clamp C bounds it |
| `num_extreme` | `lab_glucose_mean` ← train max × 10 | AUPRC degraded | clamp C saves it |
| `nan_inject` | `sao2_mean` ← NaN (post-preprocessing) | ~1 % NaN predictions | guard A replaces with 0 |
| `sentinel` | `lab_BUN_mean` ← −1 after scaling (an in-range z-score) | none expected | no stage triggered |

All four injections target numerical columns; no injection targets the categorical clamp. See [Clarifications to the paper](#clarifications-to-the-paper).

#### Run on a single seed (debug)

```bash
# From the project root, with the training environment activated
python src/train/run_safeguard_stress_test.py \
    --model_dir outputs/final_model/DUALTOWER_FT_patient_stratified_seed42_auprc0626 \
    --data_path data/train_ready/test_patient_stratified.csv \
    --output    outputs/safeguard_validation/seed42.json \
    --rate 0.01 \
    --device cuda
```

#### Run on the 5 paper-canonical seeds + aggregate

```bash
# From the project root, with the training environment activated
SEEDS="42 777 2024 9999 12345"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUT_DIR="outputs/safeguard_validation/run_${TIMESTAMP}"
mkdir -p "${OUT_DIR}"

for SEED in ${SEEDS}; do
    # Pick the highest-AUPRC checkpoint for this seed
    CKPT=$(ls -d outputs/final_model/DUALTOWER_FT_patient_stratified_seed${SEED}_auprc* 2>/dev/null \
           | sort -t_ -k7 -r | head -1)
    [ -z "${CKPT}" ] && { echo "no ckpt for seed ${SEED}, skipping"; continue; }

    echo "--- seed ${SEED}: ${CKPT}"
    python src/train/run_safeguard_stress_test.py \
        --model_dir "${CKPT}" \
        --data_path data/train_ready/test_patient_stratified.csv \
        --output    "${OUT_DIR}/seed${SEED}.json" \
        --rate 0.01 \
        --device cuda
done

# Aggregate across seeds → mean ± SD per (condition, safeguard) cell
python - "${OUT_DIR}" ${SEEDS} <<'PYEOF'
import json, sys, numpy as np
from pathlib import Path

out_dir, *seeds = sys.argv[1:]
out_dir = Path(out_dir)

per_seed = {s: json.load(open(out_dir / f"seed{s}.json")) for s in seeds}
keys = list(per_seed[seeds[0]]["results"].keys())

def safe(vals, fn):
    arr = [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
    return float(fn(arr)) if arr else None

agg = {}
for k in keys:
    cells = [per_seed[s]["results"][k] for s in seeds]
    agg[k] = {
        "auprc_mean":            safe([c["auprc"] for c in cells], np.mean),
        "auprc_std":             safe([c["auprc"] for c in cells], np.std),
        "crashes_pct_mean":      safe([c["crashes_pct"] for c in cells], np.mean),
        "crashes_pct_std":       safe([c["crashes_pct"] for c in cells], np.std),
        "nan_pred_pct_mean":     safe([c["nan_pred_pct"] for c in cells], np.mean),
        "nan_pred_pct_std":      safe([c["nan_pred_pct"] for c in cells], np.std),
        "latency_ms_mean_mean":  safe([c["latency_ms_mean"] for c in cells], np.mean),
        "latency_ms_mean_std":   safe([c["latency_ms_mean"] for c in cells], np.std),
        "n_seeds":               len(cells),
    }

json.dump({"seeds": seeds, "aggregated": agg},
          open(out_dir / "aggregated.json", "w"), indent=2)
print(f"\n=== Aggregated 5-seed stress test (rate=0.01) ===")
def f(m, s, d=2):
    if m is None: return "N/A"
    return f"{m:.{d}f} ± {s:.{d}f}" if s and s > 0 else f"{m:.{d}f}"
hdr = ("Condition","Safeguard","Crashes %","NaN preds %","AUPRC","Latency ms")
print(f"{hdr[0]:<14}{hdr[1]:<10}{hdr[2]:>16}{hdr[3]:>18}{hdr[4]:>16}{hdr[5]:>22}")
print("-" * 96)
for cond in ["baseline","cat_oor","num_extreme","nan_inject","sentinel"]:
    for sg in (True, False):
        k = f"{cond}_sg_{sg}"
        if k not in agg: continue
        a = agg[k]
        print(f"{cond:<14}{('ON' if sg else 'OFF'):<10}"
              f"{f(a['crashes_pct_mean'], a['crashes_pct_std'], 1):>16}"
              f"{f(a['nan_pred_pct_mean'], a['nan_pred_pct_std'], 2):>18}"
              f"{f(a['auprc_mean'], a['auprc_std'], 4):>16}"
              f"{f(a['latency_ms_mean_mean'], a['latency_ms_mean_std'], 2):>22}")
PYEOF

echo
echo "Done. Per-seed JSONs and aggregated.json are in: ${OUT_DIR}"
```

Outputs are written to `outputs/safeguard_validation/run_${TIMESTAMP}/` (gitignored). The aggregated `aggregated.json` is the table used in the paper.

#### Observed results — run `run_20260601_195630`

Aggregated over 5 seeds (42, 777, 2024, 9999, 12345) at injection rate 0.01, exactly as the paper table reports. Per-seed JSONs are in the same `run_…/` directory.

| Condition | Safeguard | Device | Crashes (%) | NaN preds (%) | AUPRC | Latency (ms/batch) |
|---|---|---|---|---|---|---|
| baseline | ON | cuda | 0.0 | 0.00 | 0.6246 ± 0.0014 | 35.66 ± 0.11 *(warm-up)* |
| baseline | OFF | cpu | 100.0 | 100.00 | — | — |
| cat_oor | ON | cuda | 0.0 | 0.00 | 0.6242 ± 0.0015 | 18.89 ± 0.04 |
| cat_oor | OFF | cpu | 100.0 | 100.00 | — | — |
| num_extreme | ON | cuda | 0.0 | 0.00 | 0.6249 ± 0.0012 | 18.88 ± 0.05 |
| num_extreme | OFF | cpu | 100.0 | 100.00 | — | — |
| nan_inject | ON | cuda | 0.0 | 0.00 | 0.6234 ± 0.0013 | 18.86 ± 0.09 |
| nan_inject | OFF | cpu | 100.0 | 100.00 | — | — |
| sentinel | ON | cuda | 0.0 | 0.00 | 0.6253 ± 0.0016 | 18.70 ± 0.10 |
| sentinel | OFF | cpu | 100.0 | 100.00 | — | — |

**Source files** (provenance for every number above):

- `results/safeguard_validation/run_20260601_195630/aggregated.json` — 5-seed mean ± SD per cell
- `results/safeguard_validation/run_20260601_195630/seed{42,777,2024,9999,12345}.json` — per-seed raw values, each with its `config` block recording the checkpoint, scaler, and test CSV used
- Generator: `src/train/run_safeguard_stress_test.py`
- Component implementing the safeguard: `src/train/architectures/models/dualtower_ft.py` → `DualTowerFT._compute_towers` (NaN/Inf guard, categorical clamp, ±10 numerical clamp), gated by `model.safeguard_enabled`

**Interpretation**:

- *Safeguard ON*: every condition runs cleanly. AUPRC stays within ±0.0012 of the 0.6246 baseline across the four stress conditions. The 35.66 ms latency on the first cell is one-time CUDA kernel-launch warm-up; subsequent ON cells settle at ~18.8 ms.
- *Safeguard OFF*: every cell raises an `IndexError` on the categorical embedding lookup, **including the unperturbed baseline**. This is because the deployed `scaler.pkl` is fit on all 94 features including the three categorical embedding indices (gender, ethnicity, site_id), so post-scaling categorical values are real-valued and produce negative integers under a direct `.long()` cast. The forward-pass categorical clamp (Stage B) is therefore the sole mechanism that maps these values back into the embedding vocabulary range. The OFF cells are run on CPU because a CUDA device-side assertion on the first OFF batch poisons the context for the rest of the process; the CPU re-run reproduces the failure synchronously.

**Limitation.** The safeguard does not detect inputs that are within bounds but semantically wrong (e.g., a glucose value entered in the wrong unit but inside the plausible numeric range). That is a data-validation concern, not a runtime-safety concern.

---

## Project Structure

```
Dual-Tower-Transformer-eICU-Stroke-FT/
├── README.md                           # This file
├── LICENSE                             # CC BY-NC 4.0 + Medical AI Disclaimer
├── data/
│   └── train_ready/                    # Place preprocessed CSV splits here
│       └── DATA_README.md              # Data format instructions
├── outputs/                            # All experiment outputs (git-ignored)
│   ├── experiments/                    # Per-run experiment artifacts
│   ├── final/                          # Promoted best models
│   └── optuna/                         # Hyperparameter search results
├── results/                            # Aggregated results behind the reported numbers (no patient-level data)
└── src/
    └── train/
        ├── Makefile                    # Convenience commands (install, train, test)
        ├── requirements.txt            # Python dependencies
        ├── run_unified_train.py        # Main training entry point
        ├── run_two_stage_train.py      # Two-stage transfer learning (combined → eICU-only)
        ├── run_multiple_seeds.py       # Multi-seed experiment runner
        ├── architectures/
        │   ├── model_factory.py        # Factory pattern for model instantiation
        │   └── models/
        │       ├── base.py             # NumericalFeatureTokenizer
        │       ├── dualtower.py        # DualTower model (Transformer / MLP right tower)
        │       ├── dualtower_ft.py     # ★ DualTower-FT (FT-Transformer CLS right tower)
        │       ├── dualtower_dcn.py    # DualTower-DC (Demographic-Conditioned Numerical Attn, ablation)
        │       ├── dualtower_as.py     # DualTower-AS (DT + Learnable Soft Norm M3, ablation)
        │       ├── dualtower_ca.py     # DualTower-CA (Cross-Attention fusion, ablation)
        │       ├── mlp.py              # MLP baseline
        │       └── transformer.py      # Vanilla Transformer baseline
        ├── config/
        │   ├── _base/                  # Shared base configurations
        │   │   ├── 00_global.yaml      # Paths, logging, reproducibility, resources
        │   │   ├── 01_data.yaml        # Data splits, preprocessing, loader settings
        │   │   ├── 02_train.yaml       # Training loop, early stopping, checkpointing
        │   │   └── 03_transfer.yaml    # Two-stage transfer learning defaults
        │   └── models/                 # Per-model configuration overrides
        │       ├── dualtower_ft.yaml      # ★ DT-FT (proposed method)
        │       ├── dualtower_ft_ma.yaml   # DT-FT-MA (MIMIC-Adapted: source_mimic domain embedding)
        │       ├── dualtower_dcn.yaml     # DT-DC (ablation: demographic cross-attn)
        │       ├── dualtower_ft_optuna.yaml # DT-FT with Optuna-tuned params (ablation)
        │       ├── dualtower_as.yaml      # DT-AS (ablation: DT + Mechanism 3)
        │       ├── dualtower.yaml         # DT-Transformer (conference baseline)
        │       ├── dualtower_mlp.yaml     # DT-MLP (ablation: no self-attention)
        │       ├── dualtower_ca.yaml      # DT-CA (ablation: cross-attention fusion)
        │       ├── transformer.yaml    # Vanilla Transformer baseline
        │       ├── mlp.yaml            # MLP baseline
        │       ├── nn.yaml             # Simple neural network
        │       ├── xgboost.yaml        # XGBoost baseline
        │       └── random_forest.yaml  # RandomForest baseline
        ├── data_pipeline/
        │   ├── loader.py               # DataModule, DataLoader factory, trainer builder
        │   └── preprocessor.py         # Feature standardization, imputation, clipping
        ├── engine/
        │   ├── trainers.py             # NeuralTrainer & TreeTrainer
        │   ├── evaluator.py            # UnifiedEvaluator (metrics, CI, plots)
        │   ├── losses.py               # BCE, Focal, Label Smoothing
        │   ├── metrics.py              # AUROC, AUPRC, Brier, ECE, etc.
        │   ├── optimizers.py           # AdamW, SGD, etc.
        │   ├── schedulers.py           # Cosine, StepLR, ReduceOnPlateau, warmup
        │   ├── calibration.py          # Isotonic / Platt calibration
        │   ├── callbacks.py            # Early stopping, checkpointing
        │   └── utils_cast.py           # Mixed precision utilities
        ├── optuna/
        │   ├── run_optuna.py           # Optuna optimization entry point
        │   └── optuna_modules/
        │       ├── objective.py        # Trial objective function
        │       └── spaces.py           # Per-model search spaces
        ├── tests/
        │   └── smoke_test_cli.py       # CLI smoke tests
        └── utils/
            ├── config_manager.py       # Hierarchical YAML config with variable resolution
            ├── validator.py            # Data & config validation
            ├── exporter.py             # Model export, versioning, leaderboard
            ├── logger.py               # Structured logging (JSONL + console)
            ├── device.py               # Cross-platform device selection (CUDA / MPS / CPU)
            ├── seed.py                 # Reproducibility (seed all RNGs)
            ├── inference.py            # Batch prediction & threshold selection
            ├── visualization.py        # Publication-style plots
            ├── statistical_tests.py    # DeLong test, McNemar, etc.
            ├── calibration.py          # CalibratorWrapper
            ├── registry.py             # ModelRegistry
            └── system_check.py         # Environment & dependency validation
```

---

## Data Preprocessing

> **This repository covers the model training pipeline only.**
> The data preprocessing pipeline (eICU raw data → train-ready CSV splits) is maintained in a separate repository:
> **[Data-Preprocessing-for-eICU](https://github.com/ZR-JIA/Data-Preprocessing-for-eICU)**.
>
> The MIMIC-IV integration pipeline is maintained in `mimic_tools`
> (**[Data-Preprocessing-for-MIMIC](https://github.com/ZR-JIA/Data-Preprocessing-for-MIMIC)**), which extracts stroke ICU patients from MIMIC-IV, harmonizes features to the eICU schema, z-scores numerical features (Stage 4), and produces a combined training set (Stage 5).
>
> The prior conference paper this work builds on (CCAI 2026) is released at
> **[Dual-Tower-Transformer-eICU-Stroke](https://github.com/ZR-JIA/Dual-Tower-Transformer-eICU-Stroke)**.

**Data access**: eICU-CRD and MIMIC-IV are credentialed datasets on PhysioNet. You need your own PhysioNet credentials and signed Data Use Agreements to obtain them. This repository contains no patient-level data; see [`data/train_ready/DATA_README.md`](data/train_ready/DATA_README.md).

The preprocessed output files should be placed in `data/train_ready/`:

```
data/train_ready/
├── train_patient_stratified.csv          eICU-only train (main experiments)
├── train_patient_stratified_combined.csv  eICU + MIMIC-IV combined train (transfer experiments)
├── val_patient_stratified.csv             eICU-only val
└── test_patient_stratified.csv            eICU-only test
```

**Format**: CSV with `patient_id` as the first column. Target column: `mortality` (binary 0/1). All numerical features are z-scored; categorical features (`gender`, `ethnicity`, `site_id`) are integer-encoded for embedding lookup. The model selects them by position (`index` 1–3 in `model_config.architecture.cat_features`, `config/models/dualtower_ft.yaml`); the `name` fields there are labels only.

### MIMIC-IV Combined Training

The combined training set (`train_patient_stratified_combined.csv`) contains a `source_mimic` column (0 = eICU, 1 = MIMIC) that gives the model explicit domain awareness:

| Split | Rows | `source_mimic` | Purpose |
|-------|------|----------------|---------|
| `train_patient_stratified.csv` | 7,830 | 0 (all) | Main experiments (eICU-only) |
| `train_patient_stratified_combined.csv` | 14,955 | 0 or 1 | MIMIC-IV transfer experiments |
| `val_patient_stratified.csv` | 1,119 | 0 (all) | Evaluation (always eICU-only) |
| `test_patient_stratified.csv` | 2,238 | 0 (all) | Evaluation (always eICU-only) |

**Scale alignment**: The `mimic_tools` Stage 4 (`adapt_to_dual_tower.py`) z-scores MIMIC numerical features before merging. Without this step, eICU features are ~N(0,1) while MIMIC features are raw physiological values (e.g., heartrate=75 vs heartrate≈0), causing catastrophic scale mismatch and negative transfer.

**Leakage-free preprocessing**: `loader.py` fits the `TabularPreprocessor` exclusively on eICU-only rows from the combined train (via `source_mimic == 0` mask) to prevent MIMIC's distribution from contaminating imputation and scaling statistics. MIMIC rows are then transformed with eICU-fitted parameters. See `src/train/data_pipeline/loader.py` → `_preprocess()`.

---

## Installation

All commands should be run from the `src/train/` directory:

```bash
cd src/train
```

### Option 1: Using Make (Recommended)

```bash
# Install all dependencies (CPU)
make install

# Verify installation
make validate
```

### Option 2: Using pip directly

```bash
pip install -r requirements.txt
```

### Option 3: GPU Support

```bash
# NVIDIA CUDA 11.8
make install-gpu

# Apple Silicon (MPS) — included by default
pip install -r requirements.txt

# Verify GPU availability
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, MPS: {torch.backends.mps.is_available()}')"
```

### System Requirements

- **Python** >= 3.10
- **PyTorch** >= 2.0
- **RAM** >= 16 GB recommended
- **GPU** optional (CUDA 11.8+ or Apple MPS)

---

## Usage

> **Important**: All training commands must be executed from the `src/train/` directory.
> All output paths in the codebase are relative to this working directory (e.g., `../../outputs/` resolves to the project-root `outputs/` folder).

### Quick Start

```bash
cd src/train

# Run a quick smoke test
make test

# Train the proposed DualTower-FT
python run_unified_train.py --model dualtower_ft

# Train with a specific seed
python run_unified_train.py --model dualtower_ft --seed 42
```

### Training All Models

```bash
# ★ DualTower-FT (proposed method: FT-Transformer CLS right tower)
python run_unified_train.py --model dualtower_ft

# Ablation studies
python run_unified_train.py --model dualtower        # conference baseline (mean pooling)
python run_unified_train.py --model dualtower_as     # DT + Adaptive Safeguard M3 (ablation)
python run_unified_train.py --model dualtower_mlp    # no self-attention in right tower
python run_unified_train.py --model dualtower_ca     # cross-attention fusion (ablation)
python run_unified_train.py --model dualtower_dcn    # demographic-conditioned cross-attn (ablation)

# Baselines
python run_unified_train.py --model transformer
python run_unified_train.py --model mlp
python run_unified_train.py --model nn
python run_unified_train.py --model xgboost
python run_unified_train.py --model random_forest
```

### Two-Stage Transfer Learning

For cross-dataset generalisation experiments, we adopt the two-stage fine-tuning paradigm commonly used in medical AI transfer studies:

1. **Stage 1** — Pretrain on combined eICU + MIMIC-IV training set
2. **Stage 2** — Fine-tune on eICU-only with reduced learning rate

Val/test remain eICU-only throughout both stages.

```bash
# Single seed
python run_two_stage_train.py --model dualtower_ft

# 5-seed run for mean ± std reporting
python run_two_stage_train.py --model dualtower_ft --seeds 5

# Override Stage 2 hyperparameters
python run_two_stage_train.py --model dualtower_ft --lr-factor 0.05 --s2-epochs 50
```

Stage 2 defaults (LR factor, patience, warmup) are configured in `config/_base/03_transfer.yaml`.

### Multi-Seed Experiments

```bash
# Run DT-Transformer with 5 academic seeds → reports mean ± std
python run_multiple_seeds.py --model dualtower --seeds 5

# Run all 7 models × 5 seeds
python run_multiple_seeds.py --all --seeds 5

# Use custom seeds
python run_multiple_seeds.py --model dualtower --seeds 42,2024,777,9999,12345

# Multi-seed with two-stage transfer
python run_multiple_seeds.py --model dualtower_ft --seeds 5 --two-stage
```

### Hyperparameter Optimization

```bash
# Quick Optuna search (5 trials)
make optuna

# Full search (50 trials)
cd optuna && python run_optuna.py --model dualtower --n_trials 50
```

### Makefile Shortcuts

```bash
make help              # Show all available commands
make install           # Install dependencies
make test              # Run smoke tests
make validate          # System health check
make train             # Example MLP training
make train-xgb         # XGBoost training
make seeds             # Multi-seed experiment (3 seeds)
make clean             # Clean __pycache__ and .DS_Store
make clean-all         # Deep clean (cache + all outputs)
```

---

## Configuration

The project uses a **hierarchical YAML configuration system**:

```
config/
├── _base/
│   ├── 00_global.yaml    # Inherited by ALL models
│   ├── 01_data.yaml      # Data loading & preprocessing
│   ├── 02_train.yaml     # Training loop defaults
│   └── 03_transfer.yaml  # Two-stage transfer learning defaults
└── models/
    └── dualtower.yaml     # Model-specific overrides
```

Base configs are automatically merged with model-specific configs via `ConfigManager`. Variable references like `${common.paths.project_root}` are resolved at load time.

To customize a run, either:
1. Edit the relevant YAML file in `config/models/`
2. Or pass a custom config: `python run_unified_train.py --model dualtower --config path/to/custom.yaml`

---

## Results

### Clarifications to the paper

The code in this repository is the version that produced all results
reported in the PRICAI 2026 paper; the camera-ready revision changed only
the text. The following points clarify where the paper's description
differs from the experiments as run. No reported number is affected.

**1. Stress-test conditions (Section 4.3, Table 2).**
All four injection conditions target numerical columns
(`age` = column 0, `sao2_mean` = 8, `lab_BUN_mean` = 39,
`lab_glucose_mean` = 79).
- The condition labelled "categorical OOR" injects `age = 200`. Age is
  processed by the numerical tower, so this value is bounded by the
  ±10 z-score clamp, not by the categorical clamp.
- The sentinel condition injects `lab_BUN_mean = −1` after scaling, where
  it is an ordinary in-range z-score, so it tests no sentinel handling.
  Section 3.1 also describes recoding −1 sentinels to NaN during
  preprocessing; the pipeline has no such step. Out-of-range values are set
  to NaN only for four vital signs (heart rate, MAP, SpO2, temperature), and
  laboratory values such as BUN are not range-checked.
- No injection targets the categorical clamp. That clamp is exercised on
  every input, which is why all batches fail on clean data when the
  safeguard is disabled.

**2. Prior-work baselines (Fig. 2, Table 1).**
The Dual-Tower Transformer and DT-MLP rows in Table 1, and all baselines
in Fig. 2, are taken from our CCAI 2026 paper and were not retrained with
this codebase. Those experiments used the same split sizes (test N = 2,238,
314 deaths) and the same 94 input features. The CCAI paper reported 97
features by counting the three categorical columns twice, so the statements
in this paper that the prior cohort had 97 features (Fig. 2 caption) or
differed by three features (Section 3.1) are incorrect. The differences
against DualTower-FT (e.g., −0.76 pp AUPRC and the cross-seed SD comparison)
come from separate training runs, not from a controlled ablation.

**3. MIMIC-IV transfer (Section 4.4, Table 3).**
DT-FT-MA trained on eICU only is an architecture-matched control: it uses
the DT-FT-MA architecture (source-domain embedding and revised categorical
indices; see the DT-FT-MA section below) but sees no MIMIC-IV data. The table therefore
contains three transfer strategies and one control. The observed difference
between DT-FT-MA (2-stage) and this control is −0.14 pp AUPRC, smaller
than the cross-seed SD; it is not a confirmed transfer effect.
The MIMIC-IV rows also have encoding defects from the preprocessing
pipeline: the gender, ethnicity, age and `site_id` codes differ from eICU
(see [Known issues](https://github.com/ZR-JIA/Data-Preprocessing-for-MIMIC#known-issues)).
These may contribute to the negative result; their effect has not been
quantified.

**4. Decision threshold for F1.**
Not reported in the paper. For each seed, the threshold is selected on the
validation set by maximising Youden's J (sensitivity + specificity − 1)
over 1,000 evenly spaced values in [0, 1], then applied unchanged to the
test set (`config/_base/02_train.yaml`, `engine/metrics.py`). Across the
five seeds the selected thresholds range from 0.369 to 0.759. This
threshold is not F1-optimised.

**5. Minor corrections.**
- "Four-fold lower cross-seed variance" (Abstract, Introduction) should
  read "four-fold lower cross-seed standard deviation".
- Section 4.1 compares against four deep learning baselines, not five.
- Fig. 1 labels the prior model as [1]; the correct reference is [13].
- The "best F1" statement in Section 4.1 is not supported: DualTower-FT
  (0.5417 ± 0.0156) and the prior DT-Transformer (0.5401 ± 0.0172) differ
  by less than one seed SD.
- DT-DC (Section 4.2, Table 1) is not a deep cross-network: it adds
  cross-attention from numerical tokens to the demographic embeddings
  inside each right-tower layer (`dualtower_dcn.py`). DT-AS and DT-DC
  modify the right tower, not the fusion layer.
- In the DT-FT configs, the `cat_features` names read age/gender/ethnicity,
  but the model selects columns by `index`; indices 1–3 are gender,
  ethnicity and site_id, as Section 3.1 states. The names are labels only.
- Section 3.1 states that the ±10 clamp replaces the fixed 99th-percentile
  cap. The earlier cap was implemented but disabled, so no cap was active
  before.
- "Age 0–75" in Section 3.1 is the range of the label-encoded age index,
  not age in years.

### Ablation Study (5-seed mean ± std, AUPRC primary metric)

| Model | AUPRC | Std | vs DT-FT | Status | Device |
|-------|-------|-----|----------|--------|--------|
| **DT-FT** (proposed) | **0.6247** | 0.0015 | — | Best confirmed DL | CUDA |
| DT-FT-FL (focal loss) | 0.6237 | 0.0050 | −0.10pp | Within noise (2026-06-04 rerun) | CUDA |
| DT-FT-LTE (large emb) | 0.6182 | 0.0108 | −0.65pp | Slight regression (2026-06-04 rerun) | CUDA |
| DT-FT-FL-LTE (combined)† | 0.6182 | 0.0108 | −0.65pp | Bit-identical to LTE; config bug | CUDA |
| DT-FT-Optuna (d_model=256, 1L) | 0.6108 | 0.0084 | −1.39pp | Overfits (2026-06-04 rerun) | CUDA |
| DT-DC (cross-attn; small and large emb) | 0.6192 | 0.0067 | −0.56pp | Negative ablation; v1 and v2 bit-identical per seed | CUDA |
| DT-CA (cross-attention fusion) | 0.6108 | 0.0099 | −1.39pp | Negative ablation | CUDA |
| DT-AS (LearnableSoftNorm M3) | 0.6096 | 0.0065 | −1.52pp | Negative ablation | CUDA |
| DT (mean pool, conference) | 0.6171 | 0.0058 | −0.76pp | Baseline | CUDA |

> † DT-FT-FL-LTE and DT-FT-LTE produce bit-identical per-seed AUPRC (config bug — trainer reads `loss.type: focal` regardless of YAML).

### MIMIC-IV Data Migration (5-seed, AUPRC, CUDA)

> All rows below are CUDA results (authoritative). MPS runs exhibited non-deterministic
> variance (seed 777: 0.5609 MPS vs 0.6314 CUDA); see Exploration Log for details.

| Strategy | AUPRC | Std | vs DT-FT | Status |
|----------|-------|-----|----------|--------|
| **DT-FT eICU-only** (reference) | **0.6247** | 0.0015 | — | Best |
| DT-FT-MA eICU-only (architecture-matched control) | 0.6240 | 0.0057 | −0.07pp | Neutral ✓ |
| DT-FT-MA 2-Stage (domain embedding) | 0.6226 | 0.0048 | −0.21pp | Mild negative |
| DT-FT 2-Stage (no domain embedding) | 0.6169 | 0.0066 | −0.78pp | Negative |
| DT-FT Direct combined (eICU+MIMIC) | 0.6188 | 0.0080 | −0.59pp | Negative (2026-06-04 rerun) |

> **Summary**: The corrected architecture (DT-FT-MA eICU-only, 0.6240) matches the
> DT-FT baseline within noise (−0.07pp), confirming the index fix and `source_mimic`
> embedding introduce no regression. MIMIC-IV transfer still does not yield positive
> transfer: 2-stage with domain embedding narrows the gap to −0.21pp (vs −0.78pp
> without it), but eICU-only training remains the best strategy at this dataset scale.

> **Key finding (updated 2026-06-04)**: CLS token (DT-FT) is the primary improvement
> (+0.76pp, std reduced 4× from 0.0058 to 0.0015). After per-variant 5-seed reruns,
> none of the four auxiliary modifications improved over the proposed baseline:
> focal loss alone fell within cross-seed variance (−0.10pp), larger embeddings
> regressed by 0.65pp, FL-LTE produced bit-identical results to LTE (config-loading
> issue — see DT-FT-FL-LTE section below), and
> Optuna-tuned hyperparameters overfit (−1.39pp).

### Comparison with Tree Baselines (Conference Results, 5-seed)

| Model | AUROC | AUPRC | Device |
|-------|-------|-------|--------|
| XGBoost (Optuna-tuned) | 0.8908 | 0.6467 | CUDA |
| RandomForest | 0.8806 | 0.6236 | CUDA |
| **DT-FT** (proposed) | 0.8848 ± 0.0024 | **0.6247 ± 0.0015** | CUDA |
| DT-Transformer (conference) | 0.8848 ± 0.0034 | 0.6171 ± 0.0058 | CUDA |
| DT-MLP | 0.8645 ± 0.0027 | 0.5394 ± 0.0054 | CUDA |
| Standard Transformer | 0.8457 ± 0.0129 | 0.5279 ± 0.0195 | CUDA |
| Standard MLP | 0.8534 ± 0.0058 | 0.5170 ± 0.0081 | CUDA |

---

## Architecture Exploration Log

Chronological record of architectural experiments with root-cause analysis.

> ⚠️ **Note on baseline values in 2026-03 and 2026-04 dated sections below.**
> These sections reference the DT-FT baseline as **0.6245 ± 0.0015** — the
> 5-seed value originally recorded. The canonical
> reference, used in the result tables above and in the paper, is
> **0.6247 ± 0.0015** (the five unique seeds in
> `aggregated_dualtower_ft_20260327_173130.json`, whose 31 records include
> duplicates). Values for DT-CA, DT-AS and DT-DC in these sections were likewise
> computed with a duplicated record included; the tables above use the five
> unique seeds, as reported in the paper.
> Historical sections preserve the original values to keep their internal
> analysis self-consistent at the time of writing; the differences do not change
> any qualitative conclusion (positive / negative / neutral verdict).
>
> Sections with **SUPERSEDED 2026-06-04** markers (DT-FT-FL, DT-FT-LTE, DT-FT-FL-LTE)
> additionally report the per-variant 5-seed reruns that replaced the
> earlier pooled-aggregation values.

### ✅ DT-FT — CLS Token Aggregation (2026-03, ACCEPTED)

**Change**: Replace mean pooling in right tower with a learnable `[CLS]` token
(FT-Transformer style, Gorishniy et al. 2021).

**Result**: AUPRC 0.6171 → **0.6247** (+0.76pp), std 0.0058 → **0.0015** (4× more stable).

**Why it works**: CLS token aggregates global interactions across all numerical
features via attention, vs. mean pooling which equally weights all positions.
The reduction in std is arguably as important as the mean gain for clinical use.

---

### ❌ DT-CA — Cross-Attention Fusion (2026-03, REJECTED)

**Change**: Replace MLP fusion layer with cross-attention between left-tower
outputs (categorical) and right-tower outputs (numerical).

**Result**: AUPRC 0.6118 (−1.27pp vs DT-FT). Fusion cross-attention too late in
the pipeline; by the time left/right outputs are vectors, cross-attention
cannot recover per-feature conditional interactions.

---

### ❌ DT-AS — Adaptive Safeguard LearnableSoftNorm (2026-03, REJECTED)

**Change**: Add learnable soft normalization (M3) to the right tower input,
intended to provide adaptive outlier correction at inference time.

**Result**: AUPRC 0.6100 ± 0.0059 (−1.45pp vs DT-FT, std no improvement).
Data was already cleaned (zero missing values, outlier-capped at 99.9th pct);
M3 adds parameters without a meaningful problem to solve.

---

### ❌ DT-FT-Optuna — Width Scaling (d_model=256, 2026-03, REJECTED)

**Change**: Optuna 30-trial TPE search → best trial at 50 epochs: d_model=256,
nhead=8, 1 layer, dropout=0.075, fusion_hidden=128, AUPRC=0.6938 (trial).

**Result**: Full 5-seed training: AUPRC 0.6246 ± 0.0014 (no gain).

**Root cause**: 50-epoch Optuna trials are unreliable for large models (d_model=256
LR is ~75% of max at epoch 50 on a T_max=100 cosine schedule). The wider model
appears better mid-training but converges to the same point with weaker
regularization (dropout 0.15 → 0.075). Width does not help on this dataset size.

---

### ❌ DT-DC — Demographic-Conditioned Numerical Attention (2026-03, REJECTED)

**Change**: Add cross-attention from right-tower numerical tokens to left-tower
demographic embeddings within each Transformer layer (DT-FT + DCNA).
Motivation: allow numerical features to be interpreted in the context of the
demographic embeddings, replicating tree-model conditional splits in continuous
attention form. (NIHSS and GCS are not in the feature set.)

**Result**: AUPRC 0.6196 ± 0.0061 (−0.49pp vs DT-FT, std regressed to 0.0058).

**Root cause**: Demographic embeddings are too low-dimensional (gender=2-dim,
ethnicity=4-dim) for meaningful cross-attention keys when projected to d_model=64.
The cross-attention over 3 demographic tokens degenerates to near-uniform attention,
adding noise rather than signal. The std regression confirms training instability
from uninformative gradient pathways through the cross-attention.

**Config**: `config/models/dualtower_dcn.yaml` | **Model**: `models/dualtower_dcn.py`

---

### DT-DC-v2 — Demographic Cross-Attention with Large Embeddings (2026-03, REJECTED)

**Change**: Same cross-attention architecture as DT-DC-v1, but with LTE-scale embedding
dimensions (age 64, gender 16, ethnicity 32, fusion_hidden 128).
Config: `config/models/dualtower_dcn_lte.yaml`

**Result**: AUPRC 0.6194 +/- 0.0060 — marginally worse than DT-DC-v1 (0.6196),
both significantly below DT-FT (0.6245). Std remains at 0.0060 (same as v1).

**Interpretation**: Larger embeddings alone (DT-FT-LTE) gave +0.02pp. Adding
cross-attention to those larger embeddings erases that gain and adds instability.
The problem is not the embedding size — cross-attention within the right tower
is structurally harmful regardless of key dimensionality. With only 3 demographic
tokens, the attention mechanism cannot learn meaningful conditional patterns and
introduces gradient noise that disrupts the CLS token's aggregation.

---

### DT-FT-FL — Focal Loss (2026-03 POSITIVE, **SUPERSEDED 2026-06-04**)

> **2026-06-04 rerun (n=5, clean per-variant aggregation)**: AUPRC **0.6237 ± 0.0050**
> (Δ vs DT-FT 0.6247: **−0.10pp, within cross-seed variance**). The 2026-03 "+0.02pp"
> claim was an artefact of pooled aggregation across multiple variant runs
> (`aggregate_metrics_from_experiments` substring-match bug).
> Source: `results/aggregated_dualtower_ft_fl_20260604_184159.json`.

**Change**: `loss.type: bce` → `loss.type: focal` (α=0.25, γ=2.0). All other
hyperparameters identical to DT-FT. Config: `config/models/dualtower_ft_fl.yaml`

**Original 2026-03 result (now corrected above)**: AUPRC 0.6247 ± 0.0014 (claimed +0.02pp).

**Why we expected it to help**: Focal Loss down-weights easy negatives via (1−p)^γ factor,
approximating XGBoost's gradient boosting curriculum where each tree focuses on
hard examples the current model still misclassifies.

---

### DT-FT-LTE — Large Tower Embeddings (2026-03 POSITIVE, **SUPERSEDED 2026-06-04**)

> **2026-06-04 rerun (n=5)**: AUPRC **0.6182 ± 0.0108** (Δ vs DT-FT 0.6247: **−0.65pp**).
> The earlier "+0.02pp" claim came from the same pooled-aggregation bug as DT-FT-FL.
> Source: `results/aggregated_dualtower_ft_lte_20260604_184159.json`.

**Change**: Increased left tower categorical embedding dimensions to match d_model:
- age: 16 → 64, gender: 2 → 16, ethnicity: 4 → 32
- Total left_out: 22 → 112 dims; fusion_hidden_dim: 64 → 128
- Config: `config/models/dualtower_ft_lte.yaml`

**Original 2026-03 result (now corrected above)**: AUPRC 0.6247 ± 0.0014 (claimed +0.02pp).

**Why it helps**: The original 2-dim gender and 4-dim ethnicity embeddings could
barely encode any signal. Expanding to 16/32 dims allows the left tower to carry
richer demographic representations into the fusion layer. Demographics now occupy
64% of the fusion input (112/176) vs 26% (22/86) previously.

**Side effect**: With meaningful embedding dims, DT-DC-v2 cross-attention is now
viable — original DT-DC failed because gender=2-dim keys degenerated to near-uniform
cross-attention.

---

### DT-FT-FL-LTE — Combined Focal Loss + Large Embeddings (2026-03, **SUPERSEDED 2026-06-04**)

> **2026-06-04 rerun (n=5)**: AUPRC **0.6182 ± 0.0108** — **bit-identical** to DT-FT-LTE
> (per-seed AUPRC matches to 16 decimal places). Root cause: in
> `src/train/engine/losses.py:212`, `get_loss_fn` reads `loss_config.get('name',
> 'bce_with_logits')` but all YAML configs use `loss.type:` (not `loss.name:`),
> so the `.get('name', ...)` lookup falls back to the default `'bce_with_logits'`
> and BCE is used regardless of what `type:` says. The bce-vs-focal contrast
> between LTE (`type: bce`) and FL-LTE (`type: focal`) was therefore never
> realised at run-time. `config_snapshot.yaml` correctly records the intended
> `type:` value for each variant; only the runtime loss-fn selection ignored it.
> Source: `results/aggregated_dualtower_ft_fl_lte_20260604_184159.json`.
> Reported in Section 4.2 of the paper.

**Change**: Apply both FL and LTE simultaneously.
Config: `config/models/dualtower_ft_fl_lte.yaml`

**Original 2026-03 result (now corrected above)**: AUPRC 0.6247 ± 0.0014 — claimed
identical to each individual variant; we now know this was driven by the
same pooled-aggregation bug.

---

### ❌ MIMIC-IV Direct Combined Training (2026-03, NEGATIVE TRANSFER)

**Change**: Train DT-FT on combined eICU + MIMIC-IV (14,955 rows) instead of
eICU-only (7,830 rows). MIMIC features harmonised via z-score alignment
(`mimic_tools` Stage 4); preprocessor fitted on eICU-only rows to prevent
statistic leakage.

**Result**: AUPRC 0.6171 ± 0.0058 — regressed to the conference DT baseline,
erasing the entire CLS token improvement.

**Root cause**: Despite feature-level harmonisation, residual distributional
differences between eICU and MIMIC-IV (coding conventions, site-level
variability, population demographics) introduce noise that the model cannot
disentangle from signal. The MIMIC rows effectively act as label noise for the
eICU evaluation distribution, pulling decision boundaries away from the
eICU-optimal frontier. The std regression (0.0015 → 0.0058) confirms increased
training instability.

---

### ❌ MIMIC-IV 2-Stage Transfer Learning (2026-03, NEGATIVE TRANSFER)

**Change**: Stage 1 pretrain on combined eICU + MIMIC-IV (100 epochs), then
Stage 2 fine-tune on eICU-only with LR × 0.1 (30 epochs, patience 10).
Standard cross-dataset transfer paradigm in medical AI.

**Result**: AUPRC 0.6169 ± 0.0066 — no improvement over direct combined
training, and worse than eICU-only DT-FT (0.6245).

**Root cause**: Two contributing factors. (1) The distributional gap between
eICU and MIMIC-IV is large enough that Stage 1 representations are biased
toward a compromise between the two domains rather than learning generalisable
ICU features. Stage 2 fine-tuning cannot fully recover from this bias within
the early stopping budget. (2) The dataset size ratio (eICU 7.8k : MIMIC 7.1k)
means MIMIC is not a minor augmentation but a near-equal partner, amplifying
its distributional influence. The higher std (0.0066) suggests the quality of
Stage 1 representations varies significantly across seeds, further indicating
that combined pretraining does not converge to a stable shared representation.

---

### ⬆️ DT-FT-MA — MIMIC-Adapted Domain Embedding (2026-04, CUDA VERIFIED)

**Change**: Three fixes to the Dual Tower's MIMIC adaptation:
1. Added `source_mimic` as a categorical embedding in the left tower (2 classes,
   8-dim), giving the model explicit domain awareness during combined training.
2. Corrected categorical feature indices (gender=1, ethnicity=2, source_mimic=94)
   which were previously misaligned after data preprocessing.
3. Kept age in the right tower as a numerical feature (ordinal, strongest predictor)
   rather than converting it to a categorical embedding.

Config: `config/models/dualtower_ft_ma.yaml` | Reuses `DualTowerFT` architecture.

**CUDA results (5 seeds, authoritative):**

| Seed | eICU-only MA | 2-Stage MA | Delta |
|------|-------------|------------|-------|
| 42 | 0.6159 | 0.6141 | −0.18pp |
| 2024 | 0.6244 | 0.6260 | +0.16pp |
| 777 | 0.6314 | 0.6247 | −0.67pp |
| 9999 | 0.6260 | 0.6232 | −0.28pp |
| 12345 | 0.6225 | 0.6250 | +0.25pp |
| **Mean** | **0.6240 ± 0.0057** | **0.6226 ± 0.0048** | −0.14pp |

vs DT-FT eICU-only reference: **0.6245 ± 0.0015** (CUDA)

**MPS results (for reference, same seeds):** eICU-only mean 0.6109 (seed 777
produced 0.5609, a non-determinism artefact confirmed by the CUDA run returning
0.6314 for the same seed). 2-Stage MPS mean: 0.6187 ± 0.0038.

**Key findings**:
- DT-FT-MA eICU-only on CUDA (0.6240) is statistically indistinguishable from
  DT-FT (0.6245), confirming that corrected indices and `source_mimic` embedding
  introduce no regression relative to the original (misconfigured) model.
- 2-Stage transfer yields a mild −0.19pp gap on CUDA, consistent with the pattern
  of absent positive transfer across all MIMIC experiments in this study.
- MPS non-determinism caused seed 777 to produce 0.5609 (vs 0.6314 on CUDA),
  artificially inflating apparent variance and degrading the MPS mean by >1pp.

**Conclusion**: Cross-dataset transfer from MIMIC-IV does not yield positive
transfer at this dataset scale (eICU 7.8k, MIMIC 7.1k), regardless of domain
embedding or two-stage training. The eICU-only DT-FT (0.6245 ± 0.0015)
remains the best configuration.

---

## Limitations

### Hardware Reproducibility (MPS vs CUDA)

All primary results reported in this repository were obtained on **NVIDIA CUDA** hardware. A subset of exploratory experiments was also run on **Apple MPS** (M1 Pro) for rapid local iteration.

MPS exhibited substantially higher run-to-run variance compared to CUDA, even with identical seeds and `torch.manual_seed` / `numpy.random.seed` fixed:

| Condition | Seed 777 AUPRC | Mean AUPRC (5-seed) | Std |
|-----------|---------------|---------------------|-----|
| CUDA | 0.6314 | 0.6240 | 0.0057 |
| MPS | 0.5609 | 0.6109 | ~0.025 |

The discrepancy stems from **non-deterministic floating-point reduction** in the MPS backend: operations such as `torch.sum` and attention softmax do not guarantee bit-identical results across runs on Apple Silicon, even when deterministic mode is requested. This is a known PyTorch/MPS limitation as of PyTorch 2.x and is not specific to this codebase.

**Practical implication**: MPS is suitable for rapid local debugging and smoke tests, but should not be used for final result reporting in academic work. All numbers cited in the Results section are CUDA-derived and should be treated as authoritative.

---

## License

This project is licensed under the [Creative Commons Attribution-NonCommercial 4.0 International License](https://creativecommons.org/licenses/by-nc/4.0/). See the [LICENSE](LICENSE) file for full terms, including the Medical AI Disclaimer.
