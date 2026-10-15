# Data Directory — Train-Ready Splits

All CSV files here are **gitignored** (DUA compliance). Only this README is tracked.

---

## Files in This Directory

| File | Rows | Source | Role |
|------|------|--------|------|
| `train_patient_stratified.csv` | 7,830 | eICU only | **Main experiments** train |
| `train_patient_stratified_combined.csv` | 14,955 | eICU + MIMIC-IV | MIMIC-IV transfer experiments train |
| `val_patient_stratified.csv` | 1,119 | eICU only | Val (all experiments) |
| `test_patient_stratified.csv` | 2,238 | eICU only | Test (all experiments) |

Val and test are always eICU-only to ensure a clean, unbiased evaluation regardless of which training set is used.

> **Scale note**: `train_patient_stratified_combined.csv` is produced by `mimic_tools` Stage 5 after Stage 4 (`adapt_to_dual_tower.py`) has z-scored the MIMIC numerical features using MIMIC's own distribution. Both eICU and MIMIC rows arrive at ~N(0,1) scale before concatenation. Without Stage 4, raw MIMIC values (heartrate≈75) and z-scored eICU values (heartrate≈0) would be concatenated directly, causing negative transfer.

---

## The `source_mimic` Column

Every file has a `source_mimic` column (last column):

| File | `source_mimic` value | Meaning |
|------|---------------------|---------|
| `train_patient_stratified.csv` | 0 (all rows) | eICU-only, no domain signal |
| `train_patient_stratified_combined.csv` | 0 = eICU row, 1 = MIMIC row | model sees domain origin |
| `val_patient_stratified.csv` | 0 (all rows) | eICU-only |
| `test_patient_stratified.csv` | 0 (all rows) | eICU-only |

The preprocessor treats `source_mimic` as a regular numeric feature — **do not add it to `drop_cols`**.

---

## Switching Experiments — Edit `01_data.yaml`

Located at `src/train/config/_base/01_data.yaml`:

```yaml
splits:
  train: train_patient_stratified.csv            # ← Default: eICU only (main experiments)
  val:   val_patient_stratified.csv
  test:  test_patient_stratified.csv
```

For the direct combined transfer experiment, change just the train line:
```yaml
splits:
  train: train_patient_stratified_combined.csv   # ← Transfer: eICU + MIMIC-IV
  val:   val_patient_stratified.csv
  test:  test_patient_stratified.csv
```

Val and test **never change** between experiments.

---

## Experiment Reference

| Experiment | `splits.train` | Train rows | Purpose |
|------------|---------------|-----------|---------|
| eICU-only (default) | `train_patient_stratified.csv` | 7,830 | Main result |
| eICU + MIMIC | `train_patient_stratified_combined.csv` | 14,955 | Transfer: direct combined |
| 2-Stage Transfer | combined → eICU-only | 14,955 → 7,830 | Transfer: pretrain, then fine-tune |

All splits: mortality rate = **14.0%** across all files (stratified sampling).

---

## 2-Stage Transfer Learning

Standard approach in medical AI cross-dataset papers (Gupta 2021, Wardi 2021, etc.):

1. **Stage 1 — Pretrain** on `train_patient_stratified_combined.csv` (eICU + MIMIC). The model learns general ICU representations from both distributions.
2. **Stage 2 — Fine-tune** on `train_patient_stratified.csv` (eICU-only) with a lower learning rate. The model adapts to the target eICU distribution, preventing MIMIC from acting as noise.

Val and test remain eICU-only throughout both stages.

In our experiments, neither 2-stage nor direct combined training improved on eICU-only training (PRICAI 2026, Section 4.4).

```bash
# Single seed
python run_two_stage_train.py --model dualtower_ft

# Multiple seeds for mean ± std reporting
python run_two_stage_train.py --model dualtower_ft --seeds 5

# Override Stage 2 hyperparameters
python run_two_stage_train.py --model dualtower_ft --lr-factor 0.05 --s2-epochs 50

# Via run_multiple_seeds.py
python run_multiple_seeds.py --model dualtower_ft --seeds 5 --two-stage
```

Stage 2 defaults are configured in `src/train/config/_base/03_transfer.yaml`.

---

## Column Format

- **Target**: `mortality` (binary: 0 = survived, 1 = died in hospital)
- **Demographics** (integer-encoded): in DT-FT, `gender`, `ethnicity` and `site_id` feed the left-tower embeddings (selected by column index); `age` is processed by the right tower as a numerical feature
- **Vitals** (z-scored, right tower): `sao2_*`, `heartrate_*`, `respiration_*`, `st1_*`, `st2_*`, `st3_*` — mean/std/slope over first 72h
- **Labs** (z-scored, right tower): `lab_*` — 60+ features, mean/std over first 72h ICU stay
- **Physical** (z-scored, right tower): `admissionheight`, `admissionweight`
- **Domain flag** (right tower numeric): `source_mimic` (0 = eICU, 1 = MIMIC) — do **not** add to `drop_cols`
- **Dropped at load time**: `patient_id`, `hospital_id`, `unit_id` (see `drop_cols` in `01_data.yaml`)

**Known all-zero features in MIMIC**: `st3_mean`, `st3_std`, `st3_slope`, `lab_-polys_mean`, `lab_-polys_std`, `lab_magnesium_std` are entirely absent in MIMIC-IV v3.1 and filled with 0 by Stage 4. eICU rows retain their original values for these features.

---

## Data Sources

- **eICU**: eICU Collaborative Research Database v2.0 (Pollard et al. 2018). Stroke ICU admissions, stratified split by patient.
- **MIMIC-IV**: MIMIC-IV v3.1 (Johnson et al. 2023). Stroke ICU cohort (ICD-10 I60–I69), harmonized to eICU schema via `mimic_tools` pipeline.

Both datasets require a PhysioNet Data Use Agreement. Do not share or redistribute these files.

---

## Preprocessing Pipelines

The code that produces these files is released separately:

| Pipeline | Repository |
|----------|------------|
| eICU extraction and feature engineering | https://github.com/ZR-JIA/Data-Preprocessing-for-eICU |
| MIMIC-IV extraction and harmonization (`mimic_tools`) | https://github.com/ZR-JIA/Data-Preprocessing-for-MIMIC |
| Prior conference paper (CCAI 2026), on which this work builds | https://github.com/ZR-JIA/Dual-Tower-Transformer-eICU-Stroke |

Neither pipeline repository contains patient-level data; both require your own
PhysioNet credentials to regenerate these files.
