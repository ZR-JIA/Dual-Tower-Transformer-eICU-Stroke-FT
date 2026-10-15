#!/usr/bin/env python3
"""
Unified Training Script for eICU Stroke Mortality Prediction

Supports all model types: MLP, NN, XGBoost, RandomForest, Transformer, DualTower

Usage:
    python run_unified_train.py --model mlp --config config/model_mlp.yaml
    python run_unified_train.py --model xgboost --config config/model_xgb.yaml
"""

import os
import sys
import argparse
import yaml
import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Import training modules
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
from engine.trainers import NeuralTrainer, TreeTrainer
from engine.evaluator import UnifiedEvaluator
from utils.config_manager import ConfigManager


def load_config_with_fallback(model_name: str, custom_config_path: Path = None) -> Dict[str, Any]:
    """
    Load configuration using new ConfigManager with fallback to old system.
    
    Args:
        model_name: Model name (e.g., 'mlp', 'xgboost')
        custom_config_path: Optional path to custom config file (old system)
    
    Returns:
        dict: Complete configuration
    """
    # Try new ConfigManager first
    try:
        config_mgr = ConfigManager()
        available_models = config_mgr.list_available_models()
        
        if model_name in available_models:
            print(f"✓ Using NEW config system: config/models/{model_name}.yaml")
            config = config_mgr.load_config(model_name)
            return config
        else:
            print(f"⚠️  Model '{model_name}' not found in new config system.")
            print(f"   Available models: {available_models}")
            if custom_config_path:
                print(f"   Falling back to old system with: {custom_config_path}")
            else:
                raise ValueError(
                    f"Model '{model_name}' not found in new config system and no custom config provided."
                )
    except FileNotFoundError as e:
        print(f"⚠️  New config system not fully set up: {e}")
        print(f"   Falling back to old config system")
    
    # Fallback to old system
    if custom_config_path:
        return load_config_old_system(custom_config_path, model_name)
    else:
        # Try default old config
        old_config_map = {
            'mlp': 'model_mlp.yaml',
            'nn': 'model_nn.yaml',
            'xgboost': 'model_xgb.yaml',
            'random_forest': 'model_rf.yaml',
            'transformer': 'model_transformer.yaml',
            'dualtower': 'model_transformer_dual_tower.yaml',
            'dualtower_mlp': 'model_mlp_dual_tower.yaml'
        }
        
        if model_name not in old_config_map:
            raise ValueError(f"Unknown model: {model_name}")
        
        config_path = Path(__file__).parent / 'config' / old_config_map[model_name]
        return load_config_old_system(config_path, model_name)


def load_config_old_system(config_path: Path, model_type: str) -> Dict[str, Any]:
    """
    OLD CONFIG SYSTEM - Load and merge configuration files manually.
    DEPRECATED: Use ConfigManager instead.
    
    Args:
        config_path: Path to model-specific config
        model_type: Model type
    
    Returns:
        dict: Merged configuration
    """
    config_dir = Path(__file__).parent / 'config'
    
    # Load common config
    common_path = config_dir / 'common.yaml'
    if common_path.exists():
        with open(common_path) as f:
            common_config = yaml.safe_load(f)
    else:
        common_config = {}
    
    # Load data config
    data_path = config_dir / 'data.yaml'
    if data_path.exists():
        with open(data_path) as f:
            data_config = yaml.safe_load(f)
    else:
        data_config = {}
    
    # Load train config
    train_path = config_dir / 'train.yaml'
    if train_path.exists():
        with open(train_path) as f:
            train_config = yaml.safe_load(f)
    else:
        train_config = {}
    
    # Load eval config
    eval_path = config_dir / 'eval.yaml'
    if eval_path.exists():
        with open(eval_path) as f:
            eval_config = yaml.safe_load(f)
    else:
        eval_config = {}
    
    # Load model-specific config
    with open(config_path) as f:
        model_config = yaml.safe_load(f)
    
    # Merge configurations
    config = {
        'common': common_config,
        'data': data_config,
        'train': train_config,
        'eval': eval_config,
        'model_config': model_config
    }
    
    # Resolve ${...} references using ConfigManager's resolver
    config_mgr = ConfigManager()
    config = config_mgr.resolve_variables(config)
    
    return config


def create_experiment_dir(config: Dict[str, Any]) -> Path:
    """Create experiment directory."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_name = config['model_config']['model'].upper()
    exp_name = f"{timestamp}_{model_name}"
    
    # Use project root outputs directory
    exp_dir = Path('../../outputs/experiments') / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (exp_dir / 'logs').mkdir(exist_ok=True)
    (exp_dir / 'checkpoints').mkdir(exist_ok=True)
    (exp_dir / 'artifacts').mkdir(exist_ok=True)
    (exp_dir / 'plots').mkdir(exist_ok=True)
    
    return exp_dir


def main():
    parser = argparse.ArgumentParser(description='Unified Training Script')
    parser.add_argument('--model', type=str, required=True,
                        choices=['mlp', 'nn', 'xgboost', 'random_forest', 'transformer', 'dualtower', 'dualtower_mlp', 'dualtower_ca', 'dualtower_ft', 'dualtower_ft_ma', 'dualtower_as', 'dualtower_dcn'],
                        help='Model type to train')
    parser.add_argument('--config', type=str, required=False,
                       help='Path to model config (optional, will use default if not provided)')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed (override config)')
    parser.add_argument('--experiment-name', type=str, default=None,
                       help='Experiment name (optional)')
    
    args = parser.parse_args()
    
    # ========================================
    # 1. Load Configuration (NEW SYSTEM)
    # ========================================
    print(f"Loading configuration for model: {args.model}")
    
    # Use new ConfigManager with fallback to old system
    custom_config_path = Path(args.config) if args.config else None
    config = load_config_with_fallback(args.model, custom_config_path)
    
    # Override seed if provided
    if args.seed is not None:
        config['common']['reproducibility']['seed'] = args.seed
    
    # ========================================
    # 2. Create Experiment Directory
    # ========================================
    exp_dir = create_experiment_dir(config)
    print(f"Experiment directory: {exp_dir}")
    
    # ========================================
    # 3. Setup Logging
    # ========================================
    logger = setup_logger(
        name='train',
        log_dir=exp_dir / 'logs',
        level=config['common']['logging']['level'],
        save_jsonl=config['common']['logging']['save_jsonl']
    )
    
    logger.info("="*80)
    logger.info("UNIFIED TRAINING SYSTEM - eICU Stroke Mortality Prediction")
    logger.info("="*80)
    logger.info(f"Model: {args.model.upper()}")
    logger.info(f"Experiment: {exp_dir.name}")
    
    # Log system information
    system_info = log_system_info(logger)
    
    # ========================================
    # 4. Set Random Seed
    # ========================================
    seed = config['common']['reproducibility']['seed']
    seed_config = set_all_seeds(
        seed=seed,
        deterministic=config['common']['reproducibility']['cudnn_deterministic'],
        benchmark=config['common']['reproducibility']['cudnn_benchmark']
    )
    logger.info(f"Random seed set: {seed}")
    logger.info(f"Seed configuration: {seed_config}")
    
    # ========================================
    # 5. Save Configuration Snapshot
    # ========================================
    config_snapshot_path = exp_dir / 'artifacts' / 'config_snapshot.yaml'
    with open(config_snapshot_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    logger.info(f"Configuration snapshot saved: {config_snapshot_path}")
    
    # Compute and log config hash
    config_hash = log_config_hash(config)
    logger.info(f"Configuration hash: {config_hash}")
    
    # ========================================
    # 6. Data Validation
    # ========================================
    logger.info("="*80)
    logger.info("DATA VALIDATION")
    logger.info("="*80)
    
    validator = DataValidator(config)
    config_valid, config_issues = validator.validate_config()
    
    if not config_valid:
        logger.error("Configuration validation failed:")
        for issue in config_issues:
            logger.error(f"  - {issue}")
        sys.exit(1)
    
    logger.info("✓ Configuration validation passed")
    
    # ========================================
    # 7. Load and Preprocess Data
    # ========================================
    logger.info("="*80)
    logger.info("DATA LOADING")
    logger.info("="*80)
    
    try:
        data_module = build_datamodule(config)
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        logger.exception(e)
        sys.exit(1)
    
    # Validate data splits
    from pandas import DataFrame
    train_df = DataFrame({config['common']['target_col']: data_module.y_train})
    val_df = DataFrame({config['common']['target_col']: data_module.y_val})
    test_df = DataFrame({config['common']['target_col']: data_module.y_test})
    
    data_valid, data_issues = validator.validate_splits(train_df, val_df, test_df)
    
    if not data_valid:
        logger.warning("Data validation found issues:")
        for issue in data_issues:
            logger.warning(f"  - {issue}")
        
        # Ask if we should continue
        response = input("Continue despite validation issues? (y/n): ")
        if response.lower() != 'y':
            logger.info("Training aborted by user")
            sys.exit(1)
    
    # ========================================
    # 8. Build Model
    # ========================================
    logger.info("="*80)
    logger.info("MODEL BUILDING")
    logger.info("="*80)
    
    input_dim = len(data_module.feature_names)
    logger.info(f"Input dimension: {input_dim}")
    
    try:
        model = build_model(config, input_dim)
    except Exception as e:
        logger.error(f"Failed to build model: {e}")
        logger.exception(e)
        sys.exit(1)
    
    # Log model info
    if hasattr(model, 'parameters'):
        n_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model parameters: {n_params:,}")
    
    # ========================================
    # 9. Training
    # ========================================
    logger.info("="*80)
    logger.info("TRAINING")
    logger.info("="*80)
    
    from utils.device import get_device, should_pin_memory
    device = get_device()
    logger.info(f"Training device: {device}")
    
    model_type = config['model_config']['model']
    
    try:
        # ========================================
        # Use Factory Pattern to Build Trainer
        # ========================================
        trainer = build_trainer(model, config, device)
        logger.info(f"Trainer type: {type(trainer).__name__}")
        
        # ========================================
        # Route Training Flow Based on Trainer Type
        # ========================================
        if isinstance(trainer, NeuralTrainer):
            # Neural network training requires DataLoaders
            logger.info("Neural network training flow")
            
            # Get data loaders
            batch_size = config['model_config'].get('train_loop', {}).get('batch_size', 512)
            train_loader, val_loader, test_loader = data_module.get_dataloaders(
                batch_size=batch_size,
                num_workers=config['common']['resources']['num_workers'],
                pin_memory=should_pin_memory(device)
            )
            
            # Train with DataLoaders
            history = trainer.fit(train_loader, val_loader)
            
            # Get numpy arrays for evaluation
            X_val, y_val = data_module.X_val, data_module.y_val
            X_test, y_test = data_module.X_test, data_module.y_test
            
        elif isinstance(trainer, TreeTrainer):
            # Tree-based training requires numpy arrays
            logger.info("Tree-based model training flow")
            
            # Get numpy arrays
            X_train, y_train, X_val, y_val, X_test, y_test = data_module.get_numpy_arrays()
            
            # Train with numpy arrays
            history = trainer.fit(X_train, y_train, X_val, y_val)
        
        else:
            raise ValueError(f"Unknown trainer type: {type(trainer).__name__}")
        
        logger.info("✓ Training completed")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        logger.exception(e)
        sys.exit(1)
    
    # ========================================
    # 10. Evaluation on Validation Set
    # ========================================
    logger.info("="*80)
    logger.info("VALIDATION EVALUATION")
    logger.info("="*80)
    
    evaluator = UnifiedEvaluator(model, config, device)
    
    # Evaluate on validation set to select threshold
    threshold_method = config['eval'].get('threshold_selection', {}).get('method', 'youden')
    
    val_metrics, threshold, threshold_method = evaluator.evaluate(
        X_val, y_val,
        threshold=None,
        threshold_method=threshold_method,
        compute_ci=False,
        save_plots=True,
        plot_dir=exp_dir / 'plots',
        prefix='val'
    )
    
    logger.info(f"Selected threshold: {threshold:.4f} (method: {threshold_method})")
    
    # ========================================
    # 11. Calibration
    # ========================================
    calibrator = None
    if config['model_config'].get('calibration_after_fit', False):
        logger.info("="*80)
        logger.info("PROBABILITY CALIBRATION")
        logger.info("="*80)
        
        calibration_method = config['eval'].get('calibration', {}).get('method', 'isotonic')
        logger.info(f"Calibration method: {calibration_method}")
        
        # Get uncalibrated predictions on validation set
        if isinstance(trainer, NeuralTrainer):
            y_val_pred_uncal = trainer.predict(val_loader)
        elif isinstance(trainer, TreeTrainer):
            y_val_pred_uncal = trainer.predict(X_val)
        else:
            raise ValueError(f"Unknown trainer type: {type(trainer).__name__}")
        
        # Fit calibrator
        calibrator = CalibratorWrapper(method=calibration_method)
        calibrator.fit(y_val, y_val_pred_uncal)
        
        logger.info("✓ Calibration completed")
    
    # ========================================
    # 12. Final Evaluation on Test Set
    # ========================================
    logger.info("="*80)
    logger.info("TEST EVALUATION")
    logger.info("="*80)
    
    # Get test predictions
    if isinstance(trainer, NeuralTrainer):
        y_test_pred = trainer.predict(test_loader)
    elif isinstance(trainer, TreeTrainer):
        y_test_pred = trainer.predict(X_test)
    else:
        raise ValueError(f"Unknown trainer type: {type(trainer).__name__}")
    
    # Apply calibration if available
    if calibrator is not None:
        logger.info("Applying calibration to test predictions")
        y_test_pred = calibrator.transform(y_test_pred)
    
    # Evaluate with fixed threshold
    test_metrics, _, _ = evaluator.evaluate(
        X_test, y_test,
        threshold=threshold,
        threshold_method='fixed',
        compute_ci=True,
        save_plots=True,
        plot_dir=exp_dir / 'plots',
        prefix='test'
    )
    
    # Save test metrics
    test_metrics_path = exp_dir / 'artifacts' / 'metrics_test.json'
    with open(test_metrics_path, 'w') as f:
        json.dump(test_metrics, f, indent=2)
    logger.info(f"Test metrics saved: {test_metrics_path}")
    
    # ========================================
    # 13. Export Final Model
    # ========================================
    logger.info("="*80)
    logger.info("FINAL MODEL EXPORT")
    logger.info("="*80)
    
    # Prepare run metadata
    run_meta = {
        'model_type': model_type.upper(),
        'experiment_name': exp_dir.name,
        'split_type': 'patient_stratified',  # This should come from config
        'seed': seed,
        'best_epoch': trainer.best_epoch if hasattr(trainer, 'best_epoch') else None,
        'training_time': None,  # TODO: track training time
        'config_hash': config_hash,
        'system_info': system_info
    }
    
    # Export final model
    exporter = FinalModelExporter(config, base_output_dir='../../outputs')
    
    latest_dir, archived_dir = exporter.export_final_model(
        model=model,
        scaler=data_module.scaler,
        calibrator=calibrator,
        feature_names=data_module.feature_names,
        metrics=test_metrics,
        threshold=threshold,
        threshold_method=threshold_method,
        best_epoch=trainer.best_epoch if hasattr(trainer, 'best_epoch') else 0,
        run_meta=run_meta
    )
    
    # ========================================
    # 14. Update Leaderboard
    # ========================================
    logger.info("="*80)
    logger.info("LEADERBOARD UPDATE")
    logger.info("="*80)
    
    leaderboard_path = Path('../../outputs/leaderboard.csv')
    
    # Get version
    version_file = archived_dir / 'VERSION'
    version = version_file.read_text().strip() if version_file.exists() else '1.0.0'
    
    update_leaderboard(
        leaderboard_path=leaderboard_path,
        model_name=model_type.upper(),
        experiment_name=exp_dir.name,
        split_type=run_meta['split_type'],
        seed=seed,
        metrics=test_metrics,
        version=version,
        archived_path=str(archived_dir),
        timestamp=datetime.now().isoformat()
    )
    
    logger.info(f"✓ Leaderboard updated: {leaderboard_path}")
    
    # ========================================
    # 15. Summary
    # ========================================
    logger.info("="*80)
    logger.info("TRAINING SUMMARY")
    logger.info("="*80)
    logger.info(f"Model: {model_type.upper()}")
    logger.info(f"Experiment: {exp_dir.name}")
    logger.info(f"Latest model: {latest_dir}")
    logger.info(f"Archived model: {archived_dir}")
    logger.info(f"Primary metric ({config['common']['promotion']['primary_metric'].upper()}): "
                f"{test_metrics.get(config['common']['promotion']['primary_metric'], 0.0):.4f}")
    logger.info("="*80)
    logger.info("✓ TRAINING COMPLETED SUCCESSFULLY")
    logger.info("="*80)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
