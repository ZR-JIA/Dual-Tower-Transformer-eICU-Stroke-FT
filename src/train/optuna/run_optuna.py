#!/usr/bin/env python3
"""
Optuna Hyperparameter Optimization Runner

Performs automated hyperparameter tuning using Optuna for stroke mortality prediction models.

Function:
    Optimizes model hyperparameters using Tree-structured Parzen Estimator (TPE) sampling
    with median pruning. Supports all model types (MLP, Transformer, XGBoost, Random Forest).

Usage:
    # Run MLP optimization with 50 trials
    python run_optuna.py --model mlp --n_trials 50
    
    # Run XGBoost with custom seed
    python run_optuna.py --model xgboost --n_trials 100 --seed 42
    
    # Continue existing study
    python run_optuna.py --model transformer --n_trials 50 --study_name my_study --storage sqlite:///optuna.db

Arguments:
    --model         Model type (mlp, transformer, xgboost, random_forest, nn)
    --n_trials      Number of optimization trials (default: 50)
    --seed          Random seed for reproducibility (default: 42)
    --study_name    Optuna study name (default: auto-generated)
    --storage       Database URL for persistent storage (default: local SQLite)

Output:
    - Best hyperparameters saved to best_params.yaml
    - Optimization history plots
    - Study database for resumption

Author: Stroke Prediction Research Team
Date: 2026-02-06
"""

import argparse
import sys
import yaml
import torch
import optuna
import logging
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import from new package structure
from utils.config_manager import ConfigManager
from utils.seed import set_all_seeds
from data_pipeline.loader import build_datamodule
from optuna_modules.objective import UnifiedObjective


def main():
    """Main execution function."""
    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Optuna Hyperparameter Optimization for Stroke Mortality Prediction'
    )
    parser.add_argument(
        '--model', 
        type=str, 
        required=True,
        choices=['mlp', 'xgboost', 'transformer', 'random_forest', 'nn', 'dualtower', 'dualtower_mlp', 'dualtower_ft', 'dualtower_ca', 'dualtower_as', 'dualtower_dcn'],
        help='Model type to optimize'
    )
    parser.add_argument(
        '--n_trials', 
        type=int, 
        default=50,
        help='Number of optimization trials (default: 50)'
    )
    parser.add_argument(
        '--seed', 
        type=int, 
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--study_name', 
        type=str, 
        default=None,
        help='Optuna study name (default: auto-generated)'
    )
    parser.add_argument(
        '--storage', 
        type=str, 
        default=None,
        help='Database URL for persistent storage (default: local SQLite)'
    )
    args = parser.parse_args()

    # ========================================================================
    # STEP 1: Setup Output Directory
    # ========================================================================
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(__file__).parents[2] / 'outputs' / f'optuna_{args.model}_{timestamp}'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ========================================================================
    # STEP 2: Configure Logging
    # ========================================================================
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(output_dir / 'optuna.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger(__name__)
    
    logger.info("="*80)
    logger.info("OPTUNA HYPERPARAMETER OPTIMIZATION")
    logger.info("="*80)
    logger.info(f"Model: {args.model.upper()}")
    logger.info(f"Trials: {args.n_trials}")
    logger.info(f"Seed: {args.seed}")
    logger.info(f"Output: {output_dir}")
    logger.info("="*80)

    # ========================================================================
    # STEP 3: Load Configuration
    # ========================================================================
    logger.info("Loading configuration...")
    try:
        config_manager = ConfigManager()
        config = config_manager.load_config(args.model)
        
        # Override seed if provided
        if 'common' in config and 'reproducibility' in config['common']:
            config['common']['reproducibility']['seed'] = args.seed
        
        logger.info(f"✅ Configuration loaded for {args.model}")
    except Exception as e:
        logger.error(f"❌ Failed to load configuration: {e}")
        sys.exit(1)
    
    # ========================================================================
    # STEP 4: Set Random Seeds
    # ========================================================================
    set_all_seeds(args.seed)
    logger.info(f"✅ Random seeds set to {args.seed}")
    
    # ========================================================================
    # STEP 5: Select Device
    # ========================================================================
    from utils.device import get_device
    device = get_device()
    logger.info(f"✅ Device: {device}")

    # ========================================================================
    # STEP 6: Load Data (once, shared across trials)
    # ========================================================================
    logger.info("Loading data...")
    try:
        data_module = build_datamodule(config)
        logger.info(f"✅ Data loaded: {len(data_module.X_train)} train samples")
    except Exception as e:
        logger.error(f"❌ Failed to load data: {e}")
        sys.exit(1)

    # ========================================================================
    # STEP 7: Create Optuna Study
    # ========================================================================
    study_name = args.study_name or f"{args.model}_{timestamp}"
    
    # Setup storage
    if args.storage:
        storage_url = args.storage
    else:
        # Default to local SQLite in output directory
        db_path = output_dir / "optuna.db"
        storage_url = f"sqlite:///{db_path}"
    
    logger.info(f"Creating study: {study_name}")
    logger.info(f"Storage: {storage_url}")
    
    try:
        study = optuna.create_study(
            study_name=study_name,
            storage=storage_url,
            direction='maximize',  # Maximize AUPRC
            sampler=optuna.samplers.TPESampler(seed=args.seed),
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=10
            ),
            load_if_exists=True
        )
        logger.info(f"✅ Study created")
    except Exception as e:
        logger.error(f"❌ Failed to create study: {e}")
        sys.exit(1)

    # ========================================================================
    # STEP 8: Run Optimization
    # ========================================================================
    logger.info("="*80)
    logger.info("STARTING OPTIMIZATION")
    logger.info("="*80)
    
    objective = UnifiedObjective(config, data_module, device, args.model)
    
    try:
        study.optimize(objective, n_trials=args.n_trials)
    except KeyboardInterrupt:
        logger.warning("⚠️  Optimization interrupted by user")
    except Exception as e:
        logger.error(f"❌ Optimization failed: {e}")
        import traceback
        logger.error(traceback.format_exc())

    # ========================================================================
    # STEP 9: Save Results
    # ========================================================================
    logger.info("="*80)
    logger.info("OPTIMIZATION COMPLETE")
    logger.info("="*80)
    
    if len(study.trials) > 0:
        logger.info(f"✅ Completed trials: {len(study.trials)}")
        logger.info(f"✅ Best AUPRC: {study.best_value:.4f}")
        logger.info("")
        logger.info("Best Hyperparameters:")
        logger.info("-" * 40)
        for k, v in study.best_params.items():
            logger.info(f"  {k}: {v}")
        
        # Save best parameters
        best_params_path = output_dir / 'best_params.yaml'
        with open(best_params_path, 'w') as f:
            yaml.dump(study.best_params, f)
        logger.info(f"\n✅ Best parameters saved to {best_params_path}")
        
        # Generate and save visualization plots
        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            import matplotlib.pyplot as plt
            
            # Optimization history
            fig = optuna.visualization.matplotlib.plot_optimization_history(study)
            fig.savefig(output_dir / 'optimization_history.png', dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            # Parameter importances
            if len(study.trials) >= 5:  # Need at least 5 trials for importance
                fig = optuna.visualization.matplotlib.plot_param_importances(study)
                fig.savefig(output_dir / 'param_importance.png', dpi=150, bbox_inches='tight')
                plt.close(fig)
            
            logger.info("✅ Visualization plots saved")
        except Exception as e:
            logger.warning(f"⚠️  Could not generate plots: {e}")
        
        # Save study statistics
        stats = {
            'model': args.model,
            'n_trials': len(study.trials),
            'best_value': float(study.best_value),
            'best_params': study.best_params,
            'study_name': study_name,
            'timestamp': timestamp,
        }
        
        stats_path = output_dir / 'study_stats.yaml'
        with open(stats_path, 'w') as f:
            yaml.dump(stats, f)
        logger.info(f"✅ Study statistics saved to {stats_path}")
        
    else:
        logger.warning("⚠️  No trials completed")
    
    logger.info("")
    logger.info("="*80)
    logger.info(f"All results saved to: {output_dir}")
    logger.info("="*80)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
