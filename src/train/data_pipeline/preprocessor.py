"""
Tabular data preprocessor with fit/transform separation.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional, List
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class TabularPreprocessor:
    """
    Stateful preprocessor for tabular data.
    
    Implements the ENCODE-ALIGN-IMPUTE-VALIDATE-SCALE pipeline with
    proper fit/transform separation.
    
    Usage:
        preprocessor = TabularPreprocessor(config)
        X_train, y_train = preprocessor.fit_transform(train_df)
        X_val, y_val = preprocessor.transform(val_df)
        X_test, y_test = preprocessor.transform(test_df)
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize preprocessor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.target_col = config.get('common', {}).get('target_col', 'mortality')
        self.drop_cols = config.get('data', {}).get('features', {}).get('drop_cols', [])
        self.no_scale_cols = config.get('data', {}).get('features', {}).get('no_scale_cols', [])
        
        # State: fitted on train, applied to val/test
        self.scaler = None
        self.feature_names = None
        self.categorical_cols = None
        self.numeric_cols = None
        self.fill_values = None
        self.train_columns = None
        self.clip_bounds = None
        self._no_scale_indices = None  # column indices to skip during scaling
        
        # Computed statistics
        self.pos_weight = None
    
    def fit_transform(
        self,
        train_df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, List[str], float]:
        """
        Fit preprocessor on training data and transform.
        
        Args:
            train_df: Training DataFrame
        
        Returns:
            Tuple of (X_train, y_train, feature_names, pos_weight)
        """
        logger.info("="*80)
        logger.info("PREPROCESSING PIPELINE: FIT-TRANSFORM (TRAIN)")
        logger.info("="*80)
        
        # Separate features and target
        X_train, y_train = self._separate_features_target(train_df)
        
        # Step 1: Identify feature types
        X_train = self._identify_feature_types(X_train)
        
        # Step 2: Encode categoricals
        X_train = self._fit_encode_categoricals(X_train)
        
        # Step 3: Convert to numeric
        X_train = self._convert_to_numeric(X_train)
        
        # Step 4: Impute missing values
        X_train = self._fit_impute_missing(X_train)
        
        # Step 5: Validate categorical features (P0 fix)
        X_train = self._validate_categoricals(X_train, split_name='Train')
        
        # Step 6: Standardize
        X_train = self._fit_scale_features(X_train)
        
        # Step 7: Outlier clipping (optional)
        X_train = self._fit_clip_outliers(X_train)
        
        # Step 8: Compute class weights
        self.pos_weight = self._compute_pos_weight(y_train)
        
        # Final verification
        self._verify_no_nans(X_train, 'Train')
        
        logger.info("="*80)
        logger.info("✅ PREPROCESSING COMPLETED (TRAIN)")
        logger.info("="*80)
        
        return X_train, y_train, self.feature_names, self.pos_weight
    
    def transform(
        self,
        df: pd.DataFrame,
        split_name: str = 'Val/Test'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform validation/test data using fitted preprocessor.
        
        Args:
            df: DataFrame to transform
            split_name: Name for logging (e.g., 'Val', 'Test')
        
        Returns:
            Tuple of (X, y)
        """
        if self.feature_names is None:
            raise RuntimeError("Preprocessor not fitted. Call fit_transform first.")
        
        logger.info(f"Transforming {split_name} data...")
        
        # Separate features and target
        X, y = self._separate_features_target(df)
        
        # Encode categoricals (using fitted encoding)
        X = self._transform_encode_categoricals(X)
        
        # Convert to numeric
        X = self._convert_to_numeric(X)
        
        # Impute (using fitted imputation values)
        X = self._transform_impute_missing(X)
        
        # Validate categoricals
        X = self._validate_categoricals(X, split_name=split_name)
        
        # Scale (using fitted scaler)
        X = self._transform_scale_features(X)
        
        # Clip (using fitted bounds)
        X = self._transform_clip_outliers(X)
        
        # Verify
        self._verify_no_nans(X, split_name)
        
        logger.info(f"✅ {split_name} data transformed successfully")
        
        return X, y
    
    # ========================================================================
    # INTERNAL METHODS: STEP-BY-STEP PIPELINE
    # ========================================================================
    
    def _separate_features_target(
        self,
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """Separate features and target."""
        X = df.drop(columns=[self.target_col] + self.drop_cols, errors='ignore')
        y = df[self.target_col].values
        return X, y
    
    def _identify_feature_types(self, X: pd.DataFrame) -> pd.DataFrame:
        """Identify categorical vs numeric columns."""
        logger.info("-" * 80)
        logger.info("STEP 1: Identifying Feature Types")
        logger.info("-" * 80)
        
        self.categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        self.numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        
        logger.info(f"Categorical columns: {len(self.categorical_cols)}")
        if self.categorical_cols:
            logger.info(f"  Examples: {self.categorical_cols[:5]}")
        
        logger.info(f"Numeric columns: {len(self.numeric_cols)}")
        if self.numeric_cols:
            logger.info(f"  Examples: {self.numeric_cols[:5]}")
        
        return X
    
    def _fit_encode_categoricals(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit and encode categorical features."""
        logger.info("-" * 80)
        logger.info("STEP 2: Categorical Encoding (One-Hot)")
        logger.info("-" * 80)
        
        if self.categorical_cols:
            logger.info(f"Applying pd.get_dummies to {len(self.categorical_cols)} features...")
            X = pd.get_dummies(X, columns=self.categorical_cols, drop_first=True, dtype=float)
            logger.info(f"After encoding: {X.shape}")
            
            # Store train columns for alignment
            self.train_columns = X.columns.tolist()
            logger.info(f"Stored {len(self.train_columns)} column names for alignment")
        else:
            logger.info("No categorical columns detected")
            self.train_columns = X.columns.tolist()
        
        self.feature_names = self.train_columns
        return X
    
    def _transform_encode_categoricals(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform categorical features using fitted encoding."""
        if self.categorical_cols:
            X = pd.get_dummies(X, columns=self.categorical_cols, drop_first=True, dtype=float)
            
            # Align columns
            missing_cols = set(self.train_columns) - set(X.columns)
            extra_cols = set(X.columns) - set(self.train_columns)
            
            if missing_cols:
                logger.debug(f"Adding {len(missing_cols)} missing columns (filled with 0)")
            if extra_cols:
                logger.debug(f"Dropping {len(extra_cols)} extra columns")
            
            X = X.reindex(columns=self.train_columns, fill_value=0)
        
        return X
    
    def _convert_to_numeric(self, X: pd.DataFrame) -> pd.DataFrame:
        """Convert all columns to numeric."""
        logger.info("-" * 80)
        logger.info("STEP 3: Converting to Numeric")
        logger.info("-" * 80)
        
        X = X.apply(pd.to_numeric, errors='coerce')
        
        # Drop all-NaN columns
        all_nan_cols = X.columns[X.isna().all()].tolist()
        if all_nan_cols:
            logger.warning(f"Dropping {len(all_nan_cols)} all-NaN columns: {all_nan_cols}")
            X = X.drop(columns=all_nan_cols)
            self.feature_names = list(X.columns)
        
        logger.info(f"Feature count: {len(self.feature_names)}")
        return X
    
    def _fit_impute_missing(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit and impute missing values."""
        logger.info("-" * 80)
        logger.info("STEP 4: Numeric Imputation (FIT)")
        logger.info("-" * 80)
        
        nans_before = X.isna().sum().sum()
        logger.info(f"NaNs before imputation: {nans_before}")
        
        strategy = self.config.get('data', {}).get('features', {}).get('numeric_impute', 'median')
        logger.info(f"Strategy: {strategy}")
        
        if strategy == 'median':
            self.fill_values = X.median()
        elif strategy == 'mean':
            self.fill_values = X.mean()
        elif strategy == 'zero':
            self.fill_values = pd.Series(0, index=X.columns)
        else:
            logger.warning(f"Unknown strategy '{strategy}', using median")
            self.fill_values = X.median()
        
        if nans_before > 0:
            X = X.fillna(self.fill_values)
            logger.info(f"✅ Imputation applied ({nans_before} NaNs filled)")
        else:
            logger.info("✅ No NaNs in fit data — fill_values stored for transform step")
        
        return X
    
    def _transform_impute_missing(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform: impute missing values using fitted values."""
        if self.fill_values is not None:
            X = X.fillna(self.fill_values)
        return X
    
    def _validate_categoricals(
        self,
        X: pd.DataFrame,
        split_name: str = 'Train'
    ) -> pd.DataFrame:
        """Validate and clean categorical features (P0 fix)."""
        logger.info("-" * 80)
        logger.info(f"STEP 5: Categorical Feature Validation ({split_name})")
        logger.info("-" * 80)
        
        validation_rules = {
            'age': (0, 149, 'Age'),
            'gender': (0, 9, 'Gender'),
            'ethnicity': (0, 49, 'Ethnicity')
        }
        
        categorical_cols_found = {}
        for col_name in X.columns:
            col_lower = col_name.lower()
            for pattern, (min_val, max_val, desc) in validation_rules.items():
                if pattern in col_lower:
                    categorical_cols_found[col_name] = (min_val, max_val, desc)
                    break
        
        if categorical_cols_found:
            logger.info(f"Found {len(categorical_cols_found)} categorical columns")
            for col_name, (min_val, max_val, desc) in categorical_cols_found.items():
                violations = ((X[col_name] < min_val) | (X[col_name] > max_val)).sum()
                if violations > 0:
                    logger.warning(f"⚠️  {desc} ({split_name}): {violations} values out of range [{min_val}, {max_val}]")
                    X[col_name] = X[col_name].clip(min_val, max_val)
                    logger.info(f"✅ {desc}: Cleaned {violations} out-of-range values")
                else:
                    logger.debug(f"✅ {desc}: All values valid")
        else:
            logger.debug("No categorical columns to validate")
        
        return X
    
    def _fit_scale_features(self, X: pd.DataFrame) -> np.ndarray:
        """Fit and scale features, skipping no_scale_cols."""
        logger.info("-" * 80)
        logger.info("STEP 6: Standardization (FIT)")
        logger.info("-" * 80)
        
        if self.config.get('data', {}).get('features', {}).get('standardize', True):
            # Identify columns to exclude from scaling
            if self.no_scale_cols:
                self._no_scale_indices = [i for i, c in enumerate(X.columns)
                                         if c in self.no_scale_cols]
                scale_cols = [c for c in X.columns if c not in self.no_scale_cols]
                logger.info(f"Skipping scaling for {len(self._no_scale_indices)} cols: "
                            f"{[X.columns[i] for i in self._no_scale_indices]}")
            else:
                self._no_scale_indices = []
                scale_cols = list(X.columns)
            
            if scale_cols:
                logger.info(f"Fitting StandardScaler on {len(scale_cols)} cols...")
                self.scaler = StandardScaler()
                X_scaled = self.scaler.fit_transform(X[scale_cols])
            else:
                X_scaled = np.empty((len(X), 0))
            
            # Reassemble: put scaled + unscaled columns back in original order
            result = np.empty((len(X), len(X.columns)), dtype=np.float64)
            scale_j = 0
            for i, col in enumerate(X.columns):
                if i in self._no_scale_indices:
                    result[:, i] = X[col].values.astype(np.float64)
                else:
                    result[:, i] = X_scaled[:, scale_j]
                    scale_j += 1
            X = result
            logger.info("✅ StandardScaler fitted and applied")
        else:
            logger.info("Standardization disabled")
            X = X.values
        
        return X
    
    def _transform_scale_features(self, X: pd.DataFrame) -> np.ndarray:
        """Transform: scale features using fitted scaler, skipping no_scale_cols."""
        if self.scaler is not None:
            if self._no_scale_indices:
                # Scale only the columns that were fitted
                scale_cols = [c for c in X.columns if c not in self.no_scale_cols]
                if hasattr(self.scaler, 'feature_names_in_'):
                    expected = list(self.scaler.feature_names_in_)
                    scale_cols = expected
                X_scaled = self.scaler.transform(X[scale_cols])
                
                result = np.empty((len(X), len(X.columns)), dtype=np.float64)
                scale_j = 0
                for i, col in enumerate(X.columns):
                    if col in self.no_scale_cols:
                        result[:, i] = X[col].values.astype(np.float64)
                    else:
                        result[:, i] = X_scaled[:, scale_j]
                        scale_j += 1
                return result
            else:
                if hasattr(self.scaler, 'feature_names_in_'):
                    expected = list(self.scaler.feature_names_in_)
                    if list(X.columns) != expected:
                        X = X[expected]
                X = self.scaler.transform(X)
        else:
            X = X.values
        return X
    
    def _fit_clip_outliers(self, X: np.ndarray) -> np.ndarray:
        """Fit and clip outliers."""
        logger.info("-" * 80)
        logger.info("STEP 7: Outlier Clipping (FIT)")
        logger.info("-" * 80)
        
        clip_config = self.config.get('data', {}).get('features', {}).get('clip', {})
        if clip_config.get('enabled', False):
            p_low = clip_config.get('p_low', 0.001)
            p_high = clip_config.get('p_high', 0.999)
            
            logger.info(f"Computing clip bounds at {p_low:.3f} and {p_high:.3f} percentiles...")
            lower = np.percentile(X, p_low * 100, axis=0)
            upper = np.percentile(X, p_high * 100, axis=0)
            
            self.clip_bounds = (lower, upper)
            X = np.clip(X, lower, upper)
            logger.info("✅ Outliers clipped")
        else:
            logger.info("Outlier clipping disabled")
            self.clip_bounds = None
        
        return X
    
    def _transform_clip_outliers(self, X: np.ndarray) -> np.ndarray:
        """Transform: clip outliers using fitted bounds."""
        if self.clip_bounds is not None:
            lower, upper = self.clip_bounds
            X = np.clip(X, lower, upper)
        return X
    
    def _compute_pos_weight(self, y: np.ndarray) -> float:
        """Compute positive class weight for imbalanced data."""
        logger.info("-" * 80)
        logger.info("STEP 8: Class Imbalance Analysis")
        logger.info("-" * 80)
        
        n_pos = np.sum(y == 1)
        n_neg = np.sum(y == 0)
        pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
        
        logger.info(f"Train class distribution:")
        logger.info(f"  Positive (mortality=1): {n_pos} ({n_pos/len(y)*100:.1f}%)")
        logger.info(f"  Negative (mortality=0): {n_neg} ({n_neg/len(y)*100:.1f}%)")
        logger.info(f"  Computed pos_weight: {pos_weight:.4f}")
        
        return pos_weight
    
    def _verify_no_nans(self, X: np.ndarray, split_name: str):
        """Verify no NaNs remain."""
        if np.isnan(X).any():
            raise RuntimeError(f"FATAL: NaNs detected in {split_name} after preprocessing!")
        logger.debug(f"✅ {split_name}: Zero NaNs verified")
