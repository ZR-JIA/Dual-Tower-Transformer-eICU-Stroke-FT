"""
NeuralTrainer and TreeTrainer for all model types.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import logging
from tqdm import tqdm

from .losses import get_loss_fn
from .optimizers import get_optimizer
from .schedulers import get_scheduler
from .callbacks import EarlyStopping, ModelCheckpoint
from .metrics import compute_classification_metrics

logger = logging.getLogger(__name__)


class NeuralTrainer:
    """
    Trainer for PyTorch neural network models.
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        device: torch.device
    ):
        """
        Initialize trainer.
        
        Args:
            model: PyTorch model
            config: Configuration dictionary
            device: Device to train on
        """
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        self.stop_training = False
        self.current_epoch = 0
        self.best_metric = float('-inf')
        self.best_epoch = 0
        
        # Get configurations
        self.train_config = config['train']['training']
        self.model_config = config['model_config']
        
        # Build loss function (will be initialized with y_train in fit())
        self.criterion = None
        
        # Build optimizer
        self.optimizer = get_optimizer(
            self.model.parameters(),
            self.model_config
        )
        
        # Build scheduler
        self.scheduler = None
        if 'scheduler' in self.model_config and self.model_config['scheduler']['name'] != 'none':
            self.scheduler = get_scheduler(
                self.optimizer,
                self.config  # Pass full config instead of just model_config
            )
        
        # Build callbacks
        self.callbacks = self._build_callbacks()
        
        # Metrics history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': []
        }
    
    def _build_loss(self, y_train: Optional[torch.Tensor] = None) -> nn.Module:
        """
        Build loss function from config with optional y_train for auto pos_weight.
        
        Args:
            y_train: Training labels for auto pos_weight computation
        
        Returns:
            nn.Module: Loss function
        """
        # Pass full config with y_train to loss function builder
        return get_loss_fn(self.config, y_train=y_train, device=self.device)
    
    def _build_callbacks(self) -> list:
        """Build callbacks from config."""
        callbacks = []
        
        # Early stopping
        if self.config['train']['early_stopping']['enabled']:
            es_config = self.config['train']['early_stopping']
            early_stopping = EarlyStopping(
                monitor=es_config['metric'],
                mode=es_config['mode'],
                patience=es_config['patience'],
                min_delta=es_config['min_delta'],
                restore_best_weights=es_config['restore_best_weights']
            )
            callbacks.append(early_stopping)
        
        return callbacks
    
    def fit(
        self,
        train_loader,
        val_loader,
        epochs: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Train the model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs (override config)
        
        Returns:
            dict: Training history
        """
        if epochs is None:
            epochs = self.train_config.get('max_epochs', 100)
        
        logger.info(f"Starting training for {epochs} epochs")
        
        # Initialize loss function with y_train if not yet initialized
        if self.criterion is None:
            # Collect all training labels for auto pos_weight
            logger.info("Collecting training labels for loss initialization...")
            y_train_all = []
            for batch_idx, batch in enumerate(train_loader):
                if len(batch) == 2:
                    _, y_batch = batch
                    y_train_all.append(y_batch)
            if y_train_all:
                y_train_tensor = torch.cat(y_train_all, dim=0)
                self.criterion = self._build_loss(y_train=y_train_tensor)
                logger.info(f"Loss function initialized with {len(y_train_tensor)} training samples")
            else:
                self.criterion = self._build_loss()
                logger.info("Loss function initialized without auto pos_weight")
        
        # Trigger on_train_begin callbacks
        for callback in self.callbacks:
            callback.on_train_begin(self)
        
        for epoch in range(epochs):
            if self.stop_training:
                logger.info("Early stopping triggered")
                break
            
            self.current_epoch = epoch
            
            # Trigger on_epoch_begin callbacks
            for callback in self.callbacks:
                callback.on_epoch_begin(self, epoch)
            
            # Train one epoch
            train_loss, train_metrics = self._train_epoch(train_loader)
            
            # Validate
            val_loss, val_metrics = self._validate_epoch(val_loader)
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_metrics'].append(train_metrics)
            self.history['val_metrics'].append(val_metrics)
            
            # Log metrics
            logger.info(f"Epoch {epoch+1}/{epochs}")
            logger.info(f"  Train Loss: {train_loss:.4f}")
            logger.info(f"  Val Loss: {val_loss:.4f}")
            if 'auprc' in val_metrics:
                logger.info(f"  Val AUPRC: {val_metrics['auprc']:.4f}")
            if 'auroc' in val_metrics:
                logger.info(f"  Val AUROC: {val_metrics['auroc']:.4f}")
            
            # Prepare metrics for callbacks
            epoch_metrics = {
                'train/loss': train_loss,
                'val/loss': val_loss,
                **{f'train/{k}': v for k, v in train_metrics.items()},
                **{f'val/{k}': v for k, v in val_metrics.items()}
            }
            
            # Update best metric
            primary_metric = self.config['common']['promotion']['primary_metric']
            val_metric_key = f'val/{primary_metric}'
            if val_metric_key in epoch_metrics:
                if epoch_metrics[val_metric_key] > self.best_metric:
                    self.best_metric = epoch_metrics[val_metric_key]
                    self.best_epoch = epoch
            
            # Trigger on_epoch_end callbacks
            for callback in self.callbacks:
                callback.on_epoch_end(self, epoch, epoch_metrics)
            
            # Update scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
        
        # Trigger on_train_end callbacks
        for callback in self.callbacks:
            callback.on_train_end(self)
        
        logger.info(f"Training completed. Best epoch: {self.best_epoch+1}")
        
        return self.history
    
    def _train_epoch(self, train_loader) -> Tuple[float, Dict]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        all_preds = []
        all_targets = []
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            
            # Ensure targets match output shape for loss calculation
            targets = targets.view_as(outputs)
            
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.train_config.get('gradient_clip', {}).get('enabled', False):
                clip_norm = self.train_config['gradient_clip'].get('norm', 1.0)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_norm)
            
            self.optimizer.step()
            
            # Accumulate
            total_loss += loss.item() * len(targets)
            all_preds.append(torch.sigmoid(outputs).detach().cpu().numpy())
            all_targets.append(targets.detach().cpu().numpy())
        
        # Compute epoch loss
        epoch_loss = total_loss / len(train_loader.dataset)
        
        # Compute metrics
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        metrics = compute_classification_metrics(all_targets, all_preds, threshold=0.5)
        
        return epoch_loss, metrics
    
    def _validate_epoch(self, val_loader) -> Tuple[float, Dict]:
        """Validate for one epoch."""
        self.model.eval()
        
        total_loss = 0.0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                
                # Ensure targets match output shape for loss calculation
                targets = targets.view_as(outputs)
                
                loss = self.criterion(outputs, targets)
                
                # Accumulate
                total_loss += loss.item() * len(targets)
                all_preds.append(torch.sigmoid(outputs).cpu().numpy())
                all_targets.append(targets.cpu().numpy())
        
        # Compute epoch loss
        epoch_loss = total_loss / len(val_loader.dataset)
        
        # Compute metrics
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        metrics = compute_classification_metrics(all_targets, all_preds, threshold=0.5)
        
        return epoch_loss, metrics
    
    def reset_for_finetune(
        self,
        lr_factor: float = 0.1,
        weight_decay_factor: float = 1.0,
        max_epochs: int = 30,
        warmup_epochs: int = 3,
        patience: int = 10,
        freeze_patterns: Optional[list] = None
    ):
        """
        Reset trainer state for Stage 2 fine-tuning.

        Rebuilds optimizer (with lower LR), scheduler, and callbacks.
        Optionally freezes layers matching given name patterns.

        Args:
            lr_factor: Multiply original LR by this factor for Stage 2
            weight_decay_factor: Multiply original weight_decay by this factor
            max_epochs: Max epochs for Stage 2
            warmup_epochs: Warmup epochs for Stage 2 scheduler
            patience: Early stopping patience for Stage 2
            freeze_patterns: List of parameter name substrings to freeze
        """
        # --- 1. Optionally freeze layers ---
        n_frozen = 0
        if freeze_patterns:
            for name, param in self.model.named_parameters():
                if any(pat in name for pat in freeze_patterns):
                    param.requires_grad = False
                    n_frozen += 1
            logger.info(f"Stage 2 freeze: {n_frozen} parameters frozen "
                        f"(patterns={freeze_patterns})")

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        logger.info(f"Stage 2 trainable parameters: {sum(p.numel() for p in trainable_params):,}")

        # --- 2. Rebuild optimizer with lower LR ---
        original_lr = float(self.model_config['optimizer']['lr'])
        finetune_lr = original_lr * lr_factor

        original_wd = float(self.model_config.get('optimizer', {}).get('weight_decay', 0.0))
        finetune_wd = original_wd * weight_decay_factor

        # Patch model_config so get_optimizer picks up the new values
        ft_model_config = dict(self.model_config)
        ft_optimizer = dict(ft_model_config.get('optimizer', {}))
        ft_optimizer['lr'] = finetune_lr
        ft_optimizer['weight_decay'] = finetune_wd
        ft_model_config['optimizer'] = ft_optimizer

        self.optimizer = get_optimizer(trainable_params, ft_model_config)
        logger.info(f"Stage 2 optimizer: lr={finetune_lr:.2e}  "
                    f"(×{lr_factor} of {original_lr:.2e}), wd={finetune_wd:.2e}")

        # --- 3. Rebuild scheduler ---
        ft_config = dict(self.config)
        ft_config['model_config'] = ft_model_config
        # Override scheduler warmup and T_max for Stage 2
        ft_sched = dict(ft_model_config.get('scheduler', {}))
        ft_sched['warmup_epochs'] = warmup_epochs
        if 'cosine' in ft_sched:
            ft_cosine = dict(ft_sched['cosine'])
            ft_cosine['T_max'] = max_epochs
            ft_sched['cosine'] = ft_cosine
        ft_model_config['scheduler'] = ft_sched
        ft_config['model_config'] = ft_model_config

        self.scheduler = None
        sched_name = ft_sched.get('name', 'none')
        if sched_name and sched_name != 'none':
            self.scheduler = get_scheduler(self.optimizer, ft_config)

        # --- 4. Rebuild callbacks (fresh early stopping) ---
        es_config = self.config['train']['early_stopping']
        if es_config['enabled']:
            self.callbacks = [
                EarlyStopping(
                    monitor=es_config['metric'],
                    mode=es_config['mode'],
                    patience=patience,
                    min_delta=es_config['min_delta'],
                    restore_best_weights=es_config['restore_best_weights']
                )
            ]
        else:
            self.callbacks = []

        # --- 5. Reset training state ---
        self.stop_training = False
        self.current_epoch = 0
        self.best_metric = float('-inf')
        self.best_epoch = 0
        self.criterion = None  # Will be re-initialised with Stage 2 y_train
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': []
        }

        logger.info(f"Stage 2 ready: max_epochs={max_epochs}, patience={patience}, "
                    f"warmup={warmup_epochs}")

    def predict(self, data_loader) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            data_loader: Data loader
        
        Returns:
            np.ndarray: Predicted probabilities
        """
        self.model.eval()
        all_preds = []
        
        with torch.no_grad():
            for inputs, _ in data_loader:
                inputs = inputs.to(self.device)
                
                outputs = self.model(inputs)
                probs = torch.sigmoid(outputs).cpu().numpy()
                all_preds.append(probs)
        
        return np.concatenate(all_preds)


class TreeTrainer:
    """
    Trainer for tree-based models (XGBoost, Random Forest).
    """
    
    def __init__(
        self,
        model: Any,
        config: Dict[str, Any]
    ):
        """
        Initialize trainer.
        
        Args:
            model: Tree-based model (XGBoost or RandomForest)
            config: Configuration dictionary
        """
        self.model = model
        self.config = config
        self.model_type = config['model_config']['model']
        
        self.best_metric = float('-inf')
        self.best_epoch = 0
        self.best_iteration = None
        
        self.history = {}
    
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> Dict[str, Any]:
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
        
        Returns:
            dict: Training history
        """
        logger.info(f"Training {self.model_type} model")
        
        if self.model_type == 'xgboost':
            self._fit_xgboost(X_train, y_train, X_val, y_val)
        elif self.model_type == 'random_forest':
            self._fit_random_forest(X_train, y_train, X_val, y_val)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        return self.history
    
    def _fit_xgboost(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ):
        """Fit XGBoost model."""
        # Check if early stopping is enabled
        es_config = self.config['model_config'].get('early_stopping', {})
        early_stopping = es_config.get('enabled', True)
        
        if early_stopping:
            early_stopping_rounds = es_config.get('rounds', 50)
            
            # Detect XGBoost API version for compatibility
            import inspect
            fit_signature = inspect.signature(self.model.fit)
            supports_callbacks = 'callbacks' in fit_signature.parameters
            supports_early_stopping_rounds = 'early_stopping_rounds' in fit_signature.parameters
            
            logger.info(f"XGBoost fit API: supports_callbacks={supports_callbacks}, "
                       f"supports_early_stopping_rounds={supports_early_stopping_rounds}")
            
            if supports_callbacks:
                # XGBoost 2.x+ uses callbacks
                try:
                    from xgboost.callback import EarlyStopping
                    callbacks = [EarlyStopping(rounds=early_stopping_rounds, save_best=True)]
                    
                    self.model.fit(
                        X_train, y_train,
                        eval_set=[(X_val, y_val)],
                        callbacks=callbacks,
                        verbose=True
                    )
                    logger.info("Using XGBoost callbacks API for early stopping")
                except (ImportError, AttributeError) as e:
                    logger.warning(f"Failed to use callbacks API: {e}")
                    # Fallback
                    self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=True)
            
            elif supports_early_stopping_rounds:
                # XGBoost 1.x uses early_stopping_rounds parameter
                self.model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    early_stopping_rounds=early_stopping_rounds,
                    verbose=True
                )
                logger.info("Using XGBoost early_stopping_rounds parameter for early stopping")
            
            else:
                # Very old XGBoost version, no early stopping support
                logger.warning(
                    "XGBoost version too old, neither 'callbacks' nor 'early_stopping_rounds' "
                    "supported. Training without early stopping."
                )
                self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=True)
            
            # Get best iteration if available
            if hasattr(self.model, 'best_iteration'):
                self.best_iteration = self.model.best_iteration
                self.best_epoch = self.best_iteration
                logger.info(f"Best iteration: {self.best_iteration}")
            else:
                self.best_epoch = self.model.n_estimators
                logger.info(f"No best_iteration available, using n_estimators: {self.best_epoch}")
        else:
            self.model.fit(X_train, y_train)
            self.best_epoch = self.model.n_estimators
            logger.info("Training without early stopping")
        
        logger.info("XGBoost training completed")
    
    def _fit_random_forest(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ):
        """Fit Random Forest model."""
        # Random Forest doesn't have built-in early stopping
        # Train on combined train+val or just train
        
        logger.info("Training Random Forest...")
        self.model.fit(X_train, y_train)
        
        # Compute OOB score if available
        if self.model.oob_score:
            logger.info(f"OOB Score: {self.model.oob_score_:.4f}")
        
        logger.info("Random Forest training completed")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Features
        
        Returns:
            np.ndarray: Predicted probabilities
        """
        if hasattr(self.model, 'predict_proba'):
            probs = self.model.predict_proba(X)[:, 1]
        else:
            probs = self.model.predict(X)
        
        return probs

