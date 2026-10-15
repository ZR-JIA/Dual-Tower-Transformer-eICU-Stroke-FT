#!/usr/bin/env python3
"""
Two-Stage Transfer Learning Training Script
eICU Stroke Mortality Prediction

Standard approach used in most medical AI cross-dataset transfer papers:
  Stage 1 (Pretrain):   Train on combined eICU + MIMIC data
  Stage 2 (Fine-tune):  Fine-tune on eICU-only with lower LR

Val/Test are always eICU-only and never change between stages.

Usage:
    # Default: use model's own config + transfer defaults
    python run_two_stage_train.py --model dualtower_ft

    # Override Stage 2 hyperparameters
    python run_two_stage_train.py --model dualtower_ft --lr-factor 0.05 --s2-epochs 50

    # Multiple seeds
    python run_two_stage_train.py --model dualtower_ft --seeds 5

    # Freeze right tower encoder during Stage 2
    python run_two_stage_train.py --model dualtower_ft --freeze "right_tower.transformer"
"""

import os
import sys
import copy
import argparse
import yaml
import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from utils import (
    set_all_seeds,
    setup_logger,
    log_system_info,
    log_config_hash,
    build_datamodule,
    build_model,
    DataValidator,
    FinalModelExporter,
    CalibratorWrapper,
    update_leaderboard
)
from data_pipeline.loader import build_trainer
from engine.trainers import NeuralTrainer
from engine.evaluator import UnifiedEvaluator
from utils.config_manager import ConfigManager


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_config(model_name: str) -> Dict[str, Any]:
    """Load full merged config via ConfigManager."""
    ConfigManager.reset_singleton()
    config_mgr = ConfigManager()
    return config_mgr.load_config(model_name)


def load_transfer_defaults() -> Dict[str, Any]:
    """Load transfer config from 03_transfer.yaml."""
    transfer_path = Path(__file__).parent / 'config' / '_base' / '03_transfer.yaml'
    if transfer_path.exists():
        with open(transfer_path) as f:
            return yaml.safe_load(f).get('transfer', {})
    return {}


def swap_train_file(config: Dict[str, Any], train_file: str) -> Dict[str, Any]:
    """Return a deep-copied config with a different train split file."""
    cfg = copy.deepcopy(config)
    cfg['data']['splits']['train'] = train_file
    return cfg


# ---------------------------------------------------------------------------
# Single two-stage run
# ---------------------------------------------------------------------------

def run_two_stage(
    model_name: str,
    seed: int,
    transfer_cfg: Dict[str, Any],
    lr_factor: Optional[float] = None,
    s2_epochs: Optional[int] = None,
    s2_patience: Optional[int] = None,
    s2_warmup: Optional[int] = None,
    freeze_patterns: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Execute one full two-stage training run.

    Returns:
        dict with keys: seed, stage1_best_epoch, stage2_best_epoch, test_metrics
    """
    # ── 0. Load & prepare configs ──────────────────────────────────────
    base_config = load_config(model_name)

    # Merge transfer defaults with CLI overrides
    s1_cfg = transfer_cfg.get('stage1', {})
    s2_cfg = transfer_cfg.get('stage2', {})
    lr_factor = lr_factor if lr_factor is not None else s2_cfg.get('lr_factor', 0.1)
    s2_epochs = s2_epochs if s2_epochs is not None else s2_cfg.get('max_epochs', 30)
    s2_patience = s2_patience if s2_patience is not None else s2_cfg.get('patience', 10)
    s2_warmup = s2_warmup if s2_warmup is not None else s2_cfg.get('warmup_epochs', 3)
    wd_factor = s2_cfg.get('weight_decay_factor', 1.0)

    # Freeze config
    freeze_cfg = s2_cfg.get('freeze', {})
    if freeze_patterns is None and freeze_cfg.get('enabled', False):
        freeze_patterns = freeze_cfg.get('patterns', [])

    # Override seed
    base_config['common']['reproducibility']['seed'] = seed

    # Stage-specific configs (only differ in train file)
    s1_train_file = s1_cfg.get('train_file', 'train_patient_stratified_combined.csv')
    s2_train_file = s2_cfg.get('train_file', 'train_patient_stratified.csv')

    config_s1 = swap_train_file(base_config, s1_train_file)
    config_s2 = swap_train_file(base_config, s2_train_file)

    # ── 1. Experiment directory ────────────────────────────────────────
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_tag = base_config['model_config']['model'].upper()
    exp_name = f"{timestamp}_{model_tag}_2STAGE"
    exp_dir = Path('../../outputs/experiments') / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    for sub in ('logs', 'checkpoints', 'artifacts', 'plots'):
        (exp_dir / sub).mkdir(exist_ok=True)

    # ── 2. Logging ─────────────────────────────────────────────────────
    logger = setup_logger(
        name='train',
        log_dir=exp_dir / 'logs',
        level=base_config['common']['logging']['level'],
        save_jsonl=base_config['common']['logging']['save_jsonl']
    )
    logger.info("=" * 80)
    logger.info("TWO-STAGE TRANSFER LEARNING")
    logger.info("=" * 80)
    logger.info(f"Model       : {model_tag}")
    logger.info(f"Seed        : {seed}")
    logger.info(f"Stage 1 data: {s1_train_file}")
    logger.info(f"Stage 2 data: {s2_train_file}")
    logger.info(f"Stage 2 LR  : ×{lr_factor} of original")
    logger.info(f"Stage 2 epochs: {s2_epochs}  patience: {s2_patience}  warmup: {s2_warmup}")
    if freeze_patterns:
        logger.info(f"Freeze patterns: {freeze_patterns}")

    system_info = log_system_info(logger)

    # ── 3. Seed ────────────────────────────────────────────────────────
    seed_config = set_all_seeds(
        seed=seed,
        deterministic=base_config['common']['reproducibility']['cudnn_deterministic'],
        benchmark=base_config['common']['reproducibility']['cudnn_benchmark']
    )
    logger.info(f"Seed set: {seed}")

    # Save config snapshot
    snapshot = {
        'base_config': base_config,
        'transfer': transfer_cfg,
        'overrides': {
            'lr_factor': lr_factor,
            's2_epochs': s2_epochs,
            's2_patience': s2_patience,
            's2_warmup': s2_warmup,
            'freeze_patterns': freeze_patterns,
        }
    }
    with open(exp_dir / 'artifacts' / 'config_snapshot.yaml', 'w') as f:
        yaml.dump(snapshot, f, default_flow_style=False, sort_keys=False)
    config_hash = log_config_hash(base_config)

    # ══════════════════════════════════════════════════════════════════
    #  STAGE 1 — Pretrain on combined (eICU + MIMIC)
    # ══════════════════════════════════════════════════════════════════
    logger.info("=" * 80)
    logger.info("STAGE 1 — PRETRAIN ON COMBINED (eICU + MIMIC)")
    logger.info("=" * 80)

    data_s1 = build_datamodule(config_s1)
    input_dim = len(data_s1.feature_names)
    logger.info(f"Stage 1 train shape: {data_s1.X_train.shape}")

    model = build_model(config_s1, input_dim)
    if hasattr(model, 'parameters'):
        n_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model parameters: {n_params:,}")

    from utils.device import get_device, should_pin_memory
    device = get_device()
    logger.info(f"Device: {device}")

    trainer = build_trainer(model, config_s1, device)
    assert isinstance(trainer, NeuralTrainer), \
        "Two-stage fine-tuning is only supported for neural models."

    batch_size = config_s1['model_config'].get('train_loop', {}).get('batch_size', 512)
    s1_train_loader, s1_val_loader, _ = data_s1.get_dataloaders(
        batch_size=batch_size,
        num_workers=config_s1['common']['resources']['num_workers'],
        pin_memory=should_pin_memory(device)
    )

    s1_history = trainer.fit(s1_train_loader, s1_val_loader)
    s1_best_epoch = trainer.best_epoch
    logger.info(f"Stage 1 done — best epoch: {s1_best_epoch + 1}")

    # ══════════════════════════════════════════════════════════════════
    #  STAGE 2 — Fine-tune on eICU-only
    # ══════════════════════════════════════════════════════════════════
    logger.info("=" * 80)
    logger.info("STAGE 2 — FINE-TUNE ON eICU-ONLY")
    logger.info("=" * 80)

    # Build eICU-only data module (shares same scaler fitted in Stage 1?
    # No — we re-fit the scaler on eICU-only to keep val/test statistics
    # consistent with the eICU distribution that val/test come from.)
    data_s2 = build_datamodule(config_s2)
    logger.info(f"Stage 2 train shape: {data_s2.X_train.shape}")

    # Reset trainer for fine-tuning (lower LR, fresh optimizer/scheduler)
    trainer.reset_for_finetune(
        lr_factor=lr_factor,
        weight_decay_factor=wd_factor,
        max_epochs=s2_epochs,
        warmup_epochs=s2_warmup,
        patience=s2_patience,
        freeze_patterns=freeze_patterns
    )

    s2_train_loader, s2_val_loader, s2_test_loader = data_s2.get_dataloaders(
        batch_size=batch_size,
        num_workers=config_s2['common']['resources']['num_workers'],
        pin_memory=should_pin_memory(device)
    )

    s2_history = trainer.fit(s2_train_loader, s2_val_loader, epochs=s2_epochs)
    s2_best_epoch = trainer.best_epoch
    logger.info(f"Stage 2 done — best epoch: {s2_best_epoch + 1}")

    # ══════════════════════════════════════════════════════════════════
    #  EVALUATION
    # ══════════════════════════════════════════════════════════════════
    logger.info("=" * 80)
    logger.info("EVALUATION")
    logger.info("=" * 80)

    X_val, y_val = data_s2.X_val, data_s2.y_val
    X_test, y_test = data_s2.X_test, data_s2.y_test

    evaluator = UnifiedEvaluator(model, config_s2, device)
    threshold_method = config_s2['eval'].get('threshold_selection', {}).get('method', 'youden')

    val_metrics, threshold, threshold_method = evaluator.evaluate(
        X_val, y_val,
        threshold=None,
        threshold_method=threshold_method,
        compute_ci=False,
        save_plots=True,
        plot_dir=exp_dir / 'plots',
        prefix='val'
    )
    logger.info(f"Threshold: {threshold:.4f} ({threshold_method})")

    # Calibration
    calibrator = None
    if config_s2['model_config'].get('calibration_after_fit', False):
        logger.info("Calibrating on val set …")
        y_val_pred_uncal = trainer.predict(s2_val_loader)
        calibrator = CalibratorWrapper(
            method=config_s2['eval'].get('calibration', {}).get('method', 'isotonic')
        )
        calibrator.fit(y_val, y_val_pred_uncal)

    # Test
    y_test_pred = trainer.predict(s2_test_loader)
    if calibrator is not None:
        y_test_pred = calibrator.transform(y_test_pred)

    test_metrics, _, _ = evaluator.evaluate(
        X_test, y_test,
        threshold=threshold,
        threshold_method='fixed',
        compute_ci=True,
        save_plots=True,
        plot_dir=exp_dir / 'plots',
        prefix='test'
    )

    # Save metrics
    test_metrics_path = exp_dir / 'artifacts' / 'metrics_test.json'
    with open(test_metrics_path, 'w') as f:
        json.dump(test_metrics, f, indent=2)

    # ── Export & Leaderboard ──
    model_type = base_config['model_config']['model']
    run_meta = {
        'model_type': model_type.upper(),
        'experiment_name': exp_dir.name,
        'split_type': 'patient_stratified_2stage',
        'seed': seed,
        'stage1_best_epoch': s1_best_epoch,
        'stage2_best_epoch': s2_best_epoch,
        'config_hash': config_hash,
        'system_info': system_info,
        'transfer': {
            'lr_factor': lr_factor,
            's2_epochs': s2_epochs,
            's2_patience': s2_patience,
            'freeze_patterns': freeze_patterns,
        }
    }

    exporter = FinalModelExporter(config_s2, base_output_dir='../../outputs')
    latest_dir, archived_dir = exporter.export_final_model(
        model=model,
        scaler=data_s2.scaler,
        calibrator=calibrator,
        feature_names=data_s2.feature_names,
        metrics=test_metrics,
        threshold=threshold,
        threshold_method=threshold_method,
        best_epoch=s2_best_epoch,
        run_meta=run_meta
    )

    leaderboard_path = Path('../../outputs/leaderboard.csv')
    version_file = archived_dir / 'VERSION'
    version = version_file.read_text().strip() if version_file.exists() else '1.0.0'
    update_leaderboard(
        leaderboard_path=leaderboard_path,
        model_name=f"{model_type.upper()}_2STAGE",
        experiment_name=exp_dir.name,
        split_type=run_meta['split_type'],
        seed=seed,
        metrics=test_metrics,
        version=version,
        archived_path=str(archived_dir),
        timestamp=datetime.now().isoformat()
    )

    # ── Summary ──
    primary = config_s2['common']['promotion']['primary_metric']
    logger.info("=" * 80)
    logger.info("TWO-STAGE TRAINING SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Model          : {model_type.upper()}")
    logger.info(f"Seed           : {seed}")
    logger.info(f"Stage 1 best   : epoch {s1_best_epoch + 1}")
    logger.info(f"Stage 2 best   : epoch {s2_best_epoch + 1}")
    logger.info(f"{primary.upper():15s}: {test_metrics.get(primary, 0.0):.4f}")
    logger.info(f"Experiment     : {exp_dir.name}")
    logger.info("=" * 80)

    return {
        'seed': seed,
        'stage1_best_epoch': s1_best_epoch,
        'stage2_best_epoch': s2_best_epoch,
        'test_metrics': test_metrics,
        'experiment_dir': str(exp_dir),
    }


# ---------------------------------------------------------------------------
# Multi-seed wrapper
# ---------------------------------------------------------------------------

def aggregate_results(results: List[Dict]) -> Dict[str, Dict[str, float]]:
    """Compute mean ± std across seeds."""
    all_metrics = [r['test_metrics'] for r in results if r.get('test_metrics')]
    if not all_metrics:
        return {}

    metric_names = {k for m in all_metrics for k in m.keys()
                    if isinstance(m[k], (int, float)) and 'ci_' not in k
                    and k not in ('tp', 'fp', 'tn', 'fn')}

    agg = {}
    for name in sorted(metric_names):
        vals = [m[name] for m in all_metrics if name in m]
        agg[name] = {
            'mean': float(np.mean(vals)),
            'std': float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            'min': float(np.min(vals)),
            'max': float(np.max(vals)),
            'n': len(vals),
        }
    return agg


def print_aggregated(agg: Dict, model: str):
    key_metrics = ['auroc', 'auprc', 'f1', 'brier', 'ece',
                   'precision', 'recall', 'specificity', 'balanced_accuracy']
    print(f"\n{'=' * 80}")
    print(f"  AGGREGATED — {model.upper()} (2-Stage)")
    print(f"{'=' * 80}")
    print(f"{'Metric':<25} {'Mean ± Std':<22} {'Min':>8} {'Max':>8} {'N':>4}")
    print("-" * 80)
    for m in key_metrics:
        if m in agg:
            s = agg[m]
            print(f"{m:<25} {s['mean']:.4f} ± {s['std']:.4f}      "
                  f"{s['min']:8.4f} {s['max']:8.4f} {s['n']:4d}")
    print("=" * 80)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

ACADEMIC_SEEDS_POOL = [42, 2024, 777, 9999, 12345, 54321, 10086, 31415]


def parse_seeds(arg: str) -> List[int]:
    if ',' in arg:
        return [int(s.strip()) for s in arg.split(',')]
    n = int(arg)
    return ACADEMIC_SEEDS_POOL[:n] if n <= len(ACADEMIC_SEEDS_POOL) else \
        ACADEMIC_SEEDS_POOL + list(range(42 + len(ACADEMIC_SEEDS_POOL), 42 + n))


def main():
    parser = argparse.ArgumentParser(
        description='Two-Stage Transfer Learning (combined → eICU-only)'
    )
    parser.add_argument('--model', type=str, required=True,
                        help='Model name (must be a neural model, e.g. dualtower_ft)')
    parser.add_argument('--seeds', type=str, default='1',
                        help='Number of seeds or comma-separated list (e.g. "5" or "42,2024")')
    parser.add_argument('--lr-factor', type=float, default=None,
                        help='Stage 2 LR multiplier (default from 03_transfer.yaml)')
    parser.add_argument('--s2-epochs', type=int, default=None,
                        help='Stage 2 max epochs')
    parser.add_argument('--s2-patience', type=int, default=None,
                        help='Stage 2 early-stopping patience')
    parser.add_argument('--s2-warmup', type=int, default=None,
                        help='Stage 2 warmup epochs')
    parser.add_argument('--freeze', type=str, nargs='*', default=None,
                        help='Parameter name patterns to freeze in Stage 2')
    parser.add_argument('-y', '--yes', action='store_true',
                        help='Skip confirmation (for nohup / background)')
    args = parser.parse_args()

    seeds = parse_seeds(args.seeds)
    transfer_cfg = load_transfer_defaults()

    print(f"\n  Model  : {args.model}")
    print(f"  Seeds  : {seeds}")
    print(f"  LR×    : {args.lr_factor or transfer_cfg.get('stage2', {}).get('lr_factor', 0.1)}")
    print(f"  S2 ep  : {args.s2_epochs or transfer_cfg.get('stage2', {}).get('max_epochs', 30)}")
    print()

    if not args.yes and sys.stdin.isatty():
        resp = input("Start two-stage training? (y/n): ")
        if resp.lower() != 'y':
            print("Cancelled.")
            return 0

    results = []
    for i, seed in enumerate(seeds):
        print(f"\n{'#' * 80}")
        print(f"#  RUN {i + 1}/{len(seeds)}  —  seed={seed}")
        print(f"{'#' * 80}\n")
        try:
            r = run_two_stage(
                model_name=args.model,
                seed=seed,
                transfer_cfg=transfer_cfg,
                lr_factor=args.lr_factor,
                s2_epochs=args.s2_epochs,
                s2_patience=args.s2_patience,
                s2_warmup=args.s2_warmup,
                freeze_patterns=args.freeze,
            )
            results.append(r)
            print(f"✓ seed={seed} done")
        except Exception as e:
            print(f"✗ seed={seed} FAILED: {e}")
            import traceback; traceback.print_exc()
            results.append({'seed': seed, 'error': str(e)})

    # Aggregate if multi-seed
    if len(seeds) > 1:
        agg = aggregate_results(results)
        if agg:
            print_aggregated(agg, args.model)

            out_dir = Path('../../outputs/multi_seed_results')
            out_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            with open(out_dir / f'aggregated_{args.model}_2stage_{ts}.json', 'w') as f:
                json.dump({
                    'model': args.model,
                    'seeds': seeds,
                    'aggregated': agg,
                    'per_seed': results,
                }, f, indent=2, default=str)
            print(f"✓ Aggregated results saved to {out_dir}")

    print(f"\n{'=' * 80}")
    print("✓ TWO-STAGE TRAINING COMPLETED")
    print(f"{'=' * 80}\n")
    return 0


if __name__ == '__main__':
    sys.exit(main())
