"""
Data loading, model building, and trainer routing.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .preprocessor import TabularPreprocessor
from architectures.model_factory import ModelFactory, get_model_factory
from utils.seed import seed_worker, get_generator

logger = logging.getLogger(__name__)


class DataModule:
    """
    Data module: preprocessing, DataLoader creation, and numpy access.
    
    Attributes:
        preprocessor: TabularPreprocessor instance
        scaler: StandardScaler (from preprocessor)
        feature_names: List of feature names
        pos_weight: Positive class weight
        X_train, y_train, X_val, y_val, X_test, y_test: Processed data
    """
    
    def __init__(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        config: Dict[str, Any]
    ):
        """
        Initialize data module.
        
        Args:
            train_df: Training data
            val_df: Validation data
            test_df: Test data
            config: Configuration dictionary
        """
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.config = config
        
        # Initialize preprocessor
        self.preprocessor = TabularPreprocessor(config)
        
        # Will be set by preprocessing
        self.scaler = None
        self.feature_names = None
        self.pos_weight = None
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_test = None
        self.y_test = None
        
        # Run preprocessing
        self._preprocess()
    
    def _preprocess(self):
        """
        Preprocess data using TabularPreprocessor.
        
        Delegates to preprocessor.fit_transform() and preprocessor.transform().
        
        When combined eICU+MIMIC data is detected (source_mimic column present and
        contains 1s), imputer and scaler statistics are fitted on eICU-only rows to
        prevent MIMIC's shifted distributions from contaminating normalization.
        The fitted stats are then applied to all training rows (eICU + MIMIC).
        """
        is_combined = (
            'source_mimic' in self.train_df.columns
            and self.train_df['source_mimic'].any()
        )

        if is_combined:
            eicu_df = self.train_df[self.train_df['source_mimic'] == 0].copy()
            logger.info(
                f"Combined train detected: fitting preprocessor on eICU-only rows "
                f"({len(eicu_df):,} / {len(self.train_df):,}) to preserve eICU statistics"
            )
            _, _, self.feature_names, _ = self.preprocessor.fit_transform(eicu_df)
            self.X_train, self.y_train = \
                self.preprocessor.transform(self.train_df, split_name='Train')
            y_all = self.train_df[self.preprocessor.target_col].values
            n_pos = int(y_all.sum())
            n_neg = int((y_all == 0).sum())
            self.pos_weight = n_neg / n_pos
            logger.info(f"pos_weight (combined): {self.pos_weight:.4f} (neg={n_neg:,}, pos={n_pos:,})")
        else:
            self.X_train, self.y_train, self.feature_names, self.pos_weight = \
                self.preprocessor.fit_transform(self.train_df)

        # Transform validation data
        self.X_val, self.y_val = \
            self.preprocessor.transform(self.val_df, split_name='Val')
        
        # Transform test data
        self.X_test, self.y_test = \
            self.preprocessor.transform(self.test_df, split_name='Test')
        
        # Get scaler reference
        self.scaler = self.preprocessor.scaler
        
        # Log summary
        logger.info("="*80)
        logger.info("DATA MODULE SUMMARY")
        logger.info("="*80)
        logger.info(f"Features: {len(self.feature_names)}")
        logger.info(f"Train: {self.X_train.shape}, Val: {self.X_val.shape}, Test: {self.X_test.shape}")
        logger.info(f"Pos weight: {self.pos_weight:.4f}")
        logger.info("="*80)
    
    def get_dataloaders(
        self,
        batch_size: int = 512,
        num_workers: int = 4,
        pin_memory: bool = True
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Get PyTorch DataLoaders.
        
        Args:
            batch_size: Batch size
            num_workers: Number of workers
            pin_memory: Pin memory for GPU
        
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        seed = self.config.get('common', {}).get('reproducibility', {}).get('seed', 42)
        if seed is None:
            seed = self.config.get('reproducibility', {}).get('seed', 42)
        
        # Create datasets
        train_dataset = TensorDataset(
            torch.FloatTensor(self.X_train),
            torch.FloatTensor(self.y_train)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(self.X_val),
            torch.FloatTensor(self.y_val)
        )
        test_dataset = TensorDataset(
            torch.FloatTensor(self.X_test),
            torch.FloatTensor(self.y_test)
        )
        
        # Create loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            worker_init_fn=seed_worker,
            generator=get_generator(seed)
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        
        return train_loader, val_loader, test_loader
    
    def get_numpy_arrays(self) -> Tuple:
        """Get numpy arrays (for tree models)."""
        return (
            self.X_train, self.y_train,
            self.X_val, self.y_val,
            self.X_test, self.y_test
        )


def build_datamodule(config: Dict[str, Any]) -> DataModule:
    """
    Build data module from configuration.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        DataModule instance
    """
    logger.info("Building data module...")
    
    # Load data files
    data_root = Path(config['data']['train_ready_root'])
    if not data_root.is_absolute():
        script_dir = Path(__file__).parent.parent
        base_dir = (script_dir / data_root).resolve()
    else:
        base_dir = data_root
    
    train_path = base_dir / config['data']['splits']['train']
    val_path = base_dir / config['data']['splits']['val']
    test_path = base_dir / config['data']['splits']['test']
    
    logger.info(f"Loading train: {train_path}")
    train_df = pd.read_csv(train_path)
    logger.info(f"Train samples: {len(train_df)}")
    
    logger.info(f"Loading val: {val_path}")
    val_df = pd.read_csv(val_path)
    logger.info(f"Val samples: {len(val_df)}")
    
    logger.info(f"Loading test: {test_path}")
    test_df = pd.read_csv(test_path)
    logger.info(f"Test samples: {len(test_df)}")
    
    # Validate labels
    from utils.config_utils import validate_binary_labels
    
    target_col = config.get('common', {}).get('target_col', 'mortality')
    if target_col is None:
        target_col = config.get('target_col', 'mortality')
    
    logger.info("Validating labels...")
    validate_binary_labels(train_df[target_col], "train")
    validate_binary_labels(val_df[target_col], "val")
    validate_binary_labels(test_df[target_col], "test")
    
    # Create data module
    data_module = DataModule(train_df, val_df, test_df, config)
    
    logger.info("✅ Data module built successfully")
    return data_module


def build_model(config: Dict[str, Any], input_dim: int) -> Any:
    """
    Build model from configuration.
    
    Delegates to ModelFactory for actual instantiation.
    
    Args:
        config: Configuration dictionary
        input_dim: Input feature dimension
    
    Returns:
        Model instance
    """
    model_type = config['model_config']['model']
    logger.info(f"Building model: {model_type}")
    
    # Get factory and build
    factory = get_model_factory()
    model = factory.build_model(model_type, config, input_dim)
    
    # Log parameters if neural model
    if hasattr(model, 'parameters'):
        n_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Total parameters: {n_params:,}")
    
    return model


def build_trainer(model: Any, config: Dict[str, Any], device: torch.device) -> Any:
    """
    Build trainer based on model type.
    
    Routes to NeuralTrainer or TreeTrainer based on model.
    
    Args:
        model: Model instance
        config: Configuration dictionary
        device: PyTorch device
    
    Returns:
        Trainer instance
    """
    model_type = config['model_config']['model']
    factory = get_model_factory()
    
    logger.info(f"Building trainer for: {model_type}")
    
    # Route based on model type
    if factory.is_neural_model(model_type):
        from engine.trainers import NeuralTrainer
        logger.info(f"✅ Routing to NeuralTrainer")
        return NeuralTrainer(model, config, device)
    
    elif factory.is_tree_model(model_type):
        from engine.trainers import TreeTrainer
        logger.info(f"✅ Routing to TreeTrainer")
        return TreeTrainer(model, config)
    
    else:
        # Fallback: infer from model class
        logger.warning(f"Unknown model type '{model_type}', inferring trainer...")
        
        if hasattr(model, 'forward') or isinstance(model, nn.Module):
            from engine.trainers import NeuralTrainer
            logger.info(f"Inferred: NeuralTrainer")
            return NeuralTrainer(model, config, device)
        else:
            from engine.trainers import TreeTrainer
            logger.info(f"Inferred: TreeTrainer")
            return TreeTrainer(model, config)


def build_evaluator(
    model: Any,
    config: Dict[str, Any],
    device: Optional[torch.device] = None
) -> Any:
    """
    Build evaluator from configuration.
    
    Args:
        model: Model instance
        config: Configuration dictionary
        device: PyTorch device (optional)
    
    Returns:
        Evaluator instance
    """
    from .evaluator import UnifiedEvaluator
    
    logger.info("Building evaluator...")
    evaluator = UnifiedEvaluator(model, config, device)
    logger.info("✅ Evaluator built")
    
    return evaluator
