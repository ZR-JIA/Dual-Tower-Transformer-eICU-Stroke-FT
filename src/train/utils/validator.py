"""
Data validation before training: schema, label, and split checks.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)


class DataValidator:
    """
    Validates training data for quality and integrity.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize validator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.validation_rules = config['data'].get('validate', {})
    
    def validate_splits(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame
    ) -> Tuple[bool, list]:
        """
        Validate data splits.
        
        Args:
            train_df: Training data
            val_df: Validation data
            test_df: Test data
        
        Returns:
            tuple: (is_valid, list of issues)
        """
        issues = []
        
        logger.info("Validating data splits...")
        
        # Check non-empty
        if len(train_df) == 0:
            issues.append("Training set is empty")
        if len(val_df) == 0:
            issues.append("Validation set is empty")
        if len(test_df) == 0:
            issues.append("Test set is empty")
        
        # Check columns match
        train_cols = set(train_df.columns)
        val_cols = set(val_df.columns)
        test_cols = set(test_df.columns)
        
        if not (train_cols == val_cols == test_cols):
            issues.append("Column mismatch across splits")
            logger.warning(f"Train cols: {len(train_cols)}, Val cols: {len(val_cols)}, Test cols: {len(test_cols)}")
        
        # Check target column exists
        target_col = self.config['common']['target_col']
        if target_col not in train_df.columns:
            issues.append(f"Target column '{target_col}' not found in training data")
        
        # Check target distribution
        if self.validation_rules.get('target_balance', True):
            target_dist_train = train_df[target_col].mean()
            target_dist_val = val_df[target_col].mean()
            target_dist_test = test_df[target_col].mean()
            
            logger.info(f"Target distribution:")
            logger.info(f"  Train: {target_dist_train:.3f}")
            logger.info(f"  Val:   {target_dist_val:.3f}")
            logger.info(f"  Test:  {target_dist_test:.3f}")
            
            # Check minimum target proportion
            min_target = self.validation_rules.get('target_distribution_min', 0.01)
            if target_dist_train < min_target:
                issues.append(f"Training target proportion too low: {target_dist_train:.3f} < {min_target}")
        
        # Check for data overlap (if group column exists)
        if self.validation_rules.get('check_splits', {}).get('no_overlap', True):
            group_col = self.config['common']['cv'].get('group_col')
            if group_col and group_col in train_df.columns:
                train_groups = set(train_df[group_col].unique())
                val_groups = set(val_df[group_col].unique())
                test_groups = set(test_df[group_col].unique())
                
                train_val_overlap = train_groups & val_groups
                train_test_overlap = train_groups & test_groups
                val_test_overlap = val_groups & test_groups
                
                if train_val_overlap:
                    issues.append(f"Train-val overlap: {len(train_val_overlap)} groups")
                if train_test_overlap:
                    issues.append(f"Train-test overlap: {len(train_test_overlap)} groups")
                if val_test_overlap:
                    issues.append(f"Val-test overlap: {len(val_test_overlap)} groups")
        
        # Check missing values
        max_missing = self.validation_rules.get('missing_rate_max', 0.7)
        for col in train_df.columns:
            if col == target_col:
                continue
            missing_rate = train_df[col].isna().mean()
            if missing_rate > max_missing:
                issues.append(f"Column '{col}' has high missing rate: {missing_rate:.3f}")
        
        # Check feature variance
        min_variance = self.validation_rules.get('feature_variance_min', 1e-6)
        numeric_cols = train_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col == target_col:
                continue
            var = train_df[col].var()
            if var < min_variance:
                issues.append(f"Column '{col}' has very low variance: {var:.6f}")
        
        is_valid = len(issues) == 0
        
        if is_valid:
            logger.info("✓ Data validation passed")
        else:
            logger.warning(f"✗ Data validation found {len(issues)} issues:")
            for issue in issues:
                logger.warning(f"  - {issue}")
        
        return is_valid, issues
    
    def validate_config(self) -> Tuple[bool, list]:
        """
        Validate configuration.
        
        Returns:
            tuple: (is_valid, list of issues)
        """
        issues = []
        
        # Check required keys
        required_keys = ['common', 'data', 'train', 'eval', 'model_config']
        for key in required_keys:
            if key not in self.config:
                issues.append(f"Missing required config key: '{key}'")
        
        # Check data paths exist
        if 'data' in self.config:
            # Convert relative path to absolute path based on script location
            data_root = Path(self.config['data']['train_ready_root'])
            if not data_root.is_absolute():
                # Resolve relative to the train_modules directory
                script_dir = Path(__file__).parent.parent  # Go up to src/train/
                base_dir = (script_dir / data_root).resolve()
            else:
                base_dir = data_root
            
            if not base_dir.exists():
                issues.append(f"Data directory does not exist: {base_dir}")
            
            for split in ['train', 'val', 'test']:
                split_path = base_dir / self.config['data']['splits'][split]
                if not split_path.exists():
                    issues.append(f"{split.capitalize()} file does not exist: {split_path}")
        
        is_valid = len(issues) == 0
        return is_valid, issues


def validate_data_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: Dict[str, Any]
) -> Tuple[bool, list]:
    """
    Convenience function to validate data splits.
    """
    validator = DataValidator(config)
    return validator.validate_splits(train_df, val_df, test_df)

