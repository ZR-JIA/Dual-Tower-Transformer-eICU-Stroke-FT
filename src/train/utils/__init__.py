"""
Utilities Package

Contains configuration management, logging, and helper utilities.
"""

# Core utilities
from .seed import set_all_seeds, seed_worker, get_generator
from .logger import (
    setup_logger, 
    get_logger, 
    log_system_info, 
    log_config_hash,
    save_environment,
    get_git_hash,
    get_git_branch,
    check_git_dirty,
    MetricsCSVLogger
)
from .config_manager import ConfigManager

# Data and model building (imported from data_pipeline)
try:
    from data_pipeline.loader import build_datamodule, build_model
except ImportError:
    # Graceful fallback if data_pipeline is not available
    build_datamodule = None
    build_model = None

# Validation and export
from .validator import DataValidator
from .exporter import FinalModelExporter, update_leaderboard
from .calibration import CalibratorWrapper

# Additional utilities
from .registry import ModelRegistry
from .inference import Predictor, batch_predict, select_threshold
from .visualization import (
    plot_learning_curves,
    plot_roc_curves_comparison,
    plot_pr_curves_comparison,
    plot_calibration_curves,
    plot_confusion_matrices,
    plot_model_comparison_bars,
    set_publication_style
)

__all__ = [
    # Seed management
    'set_all_seeds',
    'seed_worker',
    'get_generator',
    
    # Logging
    'setup_logger',
    'get_logger',
    'log_system_info',
    'log_config_hash',
    'save_environment',
    'get_git_hash',
    'get_git_branch',
    'check_git_dirty',
    'MetricsCSVLogger',
    
    # Configuration
    'ConfigManager',
    
    # Data and model building
    'build_datamodule',
    'build_model',
    
    # Validation and export
    'DataValidator',
    'FinalModelExporter',
    'update_leaderboard',
    'CalibratorWrapper',
    
    # Additional utilities
    'ModelRegistry',
    'Predictor',
    'batch_predict',
    'select_threshold',
    
    # Visualization
    'plot_learning_curves',
    'plot_roc_curves_comparison',
    'plot_pr_curves_comparison',
    'plot_calibration_curves',
    'plot_confusion_matrices',
    'plot_model_comparison_bars',
    'set_publication_style',
]
