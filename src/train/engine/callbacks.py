"""
Training callbacks: early stopping, checkpointing, LR monitoring.
"""

import os
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Literal
import logging

logger = logging.getLogger(__name__)


class Callback:
    """Base callback class."""
    
    def on_train_begin(self, trainer):
        """Called at the beginning of training."""
        pass
    
    def on_train_end(self, trainer):
        """Called at the end of training."""
        pass
    
    def on_epoch_begin(self, trainer, epoch):
        """Called at the beginning of each epoch."""
        pass
    
    def on_epoch_end(self, trainer, epoch, metrics):
        """Called at the end of each epoch."""
        pass
    
    def on_batch_begin(self, trainer, batch_idx):
        """Called at the beginning of each batch."""
        pass
    
    def on_batch_end(self, trainer, batch_idx, loss):
        """Called at the end of each batch."""
        pass


class EarlyStopping(Callback):
    """
    Early stopping callback.
    
    Stops training when a monitored metric stops improving.
    """
    
    def __init__(
        self,
        monitor: str = 'val/auprc',
        mode: Literal['min', 'max'] = 'max',
        patience: int = 10,
        min_delta: float = 0.0,
        restore_best_weights: bool = True
    ):
        """
        Initialize early stopping.
        
        Args:
            monitor: Metric to monitor
            mode: 'min' for loss, 'max' for metrics like AUROC
            patience: Number of epochs without improvement to wait
            min_delta: Minimum change to qualify as improvement
            restore_best_weights: Restore model weights to best epoch
        """
        super().__init__()
        self.monitor = monitor
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        
        self.best_value = None
        self.best_epoch = 0
        self.best_weights = None
        self.wait = 0
        self.stopped_epoch = 0
        
        if mode == 'min':
            self.monitor_op = np.less
            self.min_delta *= -1
        elif mode == 'max':
            self.monitor_op = np.greater
        else:
            raise ValueError(f"mode must be 'min' or 'max', got {mode}")
    
    def on_train_begin(self, trainer):
        """Initialize at training start."""
        self.best_value = np.inf if self.mode == 'min' else -np.inf
        self.best_epoch = 0
        self.wait = 0
        self.stopped_epoch = 0
        logger.info(f"Early stopping: monitoring {self.monitor} ({self.mode}), patience={self.patience}")
    
    def on_epoch_end(self, trainer, epoch, metrics):
        """Check early stopping condition."""
        current = metrics.get(self.monitor)
        
        if current is None:
            logger.warning(f"Early stopping metric '{self.monitor}' not found in metrics")
            return
        
        # Check if current value is better
        if self.monitor_op(current - self.min_delta, self.best_value):
            self.best_value = current
            self.best_epoch = epoch
            self.wait = 0
            
            # Save best weights
            if self.restore_best_weights:
                self.best_weights = {
                    k: v.cpu().clone()
                    for k, v in trainer.model.state_dict().items()
                }
            
            logger.info(f"Epoch {epoch}: {self.monitor} improved to {current:.4f}")
        else:
            self.wait += 1
            logger.info(f"Epoch {epoch}: {self.monitor}={current:.4f} (no improvement, {self.wait}/{self.patience})")
            
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                trainer.stop_training = True
                logger.info(f"Early stopping triggered at epoch {epoch}")
                logger.info(f"Best {self.monitor}: {self.best_value:.4f} at epoch {self.best_epoch}")
    
    def on_train_end(self, trainer):
        """Restore best weights if requested."""
        if self.stopped_epoch > 0:
            logger.info(f"Training stopped at epoch {self.stopped_epoch}")
        
        if self.restore_best_weights and self.best_weights is not None:
            logger.info(f"Restoring best weights from epoch {self.best_epoch}")
            trainer.model.load_state_dict(self.best_weights)


class ModelCheckpoint(Callback):
    """
    Model checkpointing callback.
    
    Saves model checkpoints based on monitored metric.
    """
    
    def __init__(
        self,
        filepath: str,
        monitor: str = 'val/auprc',
        mode: Literal['min', 'max'] = 'max',
        save_best_only: bool = True,
        save_last: bool = True,
        verbose: bool = True
    ):
        """
        Initialize model checkpoint.
        
        Args:
            filepath: Path to save checkpoints
            monitor: Metric to monitor
            mode: 'min' or 'max'
            save_best_only: Only save when metric improves
            save_last: Save last checkpoint regardless of metric
            verbose: Print save messages
        """
        super().__init__()
        self.filepath = Path(filepath)
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.save_last = save_last
        self.verbose = verbose
        
        self.best_value = None
        
        if mode == 'min':
            self.monitor_op = np.less
        elif mode == 'max':
            self.monitor_op = np.greater
        else:
            raise ValueError(f"mode must be 'min' or 'max', got {mode}")
        
        # Create directory
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
    
    def on_train_begin(self, trainer):
        """Initialize at training start."""
        self.best_value = np.inf if self.mode == 'min' else -np.inf
        if self.verbose:
            logger.info(f"Model checkpoint: monitoring {self.monitor} ({self.mode})")
    
    def on_epoch_end(self, trainer, epoch, metrics):
        """Save checkpoint if metric improved."""
        current = metrics.get(self.monitor)
        
        if current is None:
            logger.warning(f"Checkpoint metric '{self.monitor}' not found in metrics")
            return
        
        # Check if should save
        save = False
        if not self.save_best_only:
            save = True
        elif self.monitor_op(current, self.best_value):
            self.best_value = current
            save = True
        
        if save:
            # Save checkpoint
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': trainer.model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'metrics': metrics,
                self.monitor: current
            }
            
            if hasattr(trainer, 'scheduler') and trainer.scheduler is not None:
                checkpoint['scheduler_state_dict'] = trainer.scheduler.state_dict()
            
            torch.save(checkpoint, self.filepath)
            
            if self.verbose:
                logger.info(f"Saved checkpoint: {self.filepath} (epoch {epoch}, {self.monitor}={current:.4f})")
    
    def on_train_end(self, trainer):
        """Save last checkpoint if requested."""
        if self.save_last:
            last_path = self.filepath.parent / 'last_checkpoint.pth'
            checkpoint = {
                'epoch': trainer.current_epoch,
                'model_state_dict': trainer.model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
            }
            if hasattr(trainer, 'scheduler') and trainer.scheduler is not None:
                checkpoint['scheduler_state_dict'] = trainer.scheduler.state_dict()
            
            torch.save(checkpoint, last_path)
            if self.verbose:
                logger.info(f"Saved last checkpoint: {last_path}")


class LRMonitor(Callback):
    """Learning rate monitoring callback."""
    
    def __init__(self, log_every_n_epochs: int = 1):
        """
        Initialize LR monitor.
        
        Args:
            log_every_n_epochs: Log LR every N epochs
        """
        super().__init__()
        self.log_every_n_epochs = log_every_n_epochs
    
    def on_epoch_end(self, trainer, epoch, metrics):
        """Log current learning rate."""
        if epoch % self.log_every_n_epochs == 0:
            if hasattr(trainer, 'optimizer'):
                current_lr = trainer.optimizer.param_groups[0]['lr']
                logger.info(f"Epoch {epoch}: Learning rate = {current_lr:.6f}")


class MetricsLogger(Callback):
    """
    Metrics logging callback.
    
    Logs metrics to console and/or file.
    """
    
    def __init__(
        self,
        log_dir: Optional[Path] = None,
        log_every_n_epochs: int = 1
    ):
        """
        Initialize metrics logger.
        
        Args:
            log_dir: Directory to save metrics CSV
            log_every_n_epochs: Log every N epochs
        """
        super().__init__()
        self.log_dir = Path(log_dir) if log_dir else None
        self.log_every_n_epochs = log_every_n_epochs
        self.metrics_history = []
        
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.csv_path = self.log_dir / 'metrics.csv'
    
    def on_epoch_end(self, trainer, epoch, metrics):
        """Log metrics."""
        # Add epoch to metrics
        metrics_with_epoch = {'epoch': epoch, **metrics}
        self.metrics_history.append(metrics_with_epoch)
        
        # Log to console
        if epoch % self.log_every_n_epochs == 0:
            metrics_str = ', '.join([f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                                    for k, v in metrics.items()])
            logger.info(f"Epoch {epoch}: {metrics_str}")
        
        # Log to CSV
        if self.log_dir:
            self._save_csv()
    
    def _save_csv(self):
        """Save metrics history to CSV."""
        import pandas as pd
        df = pd.DataFrame(self.metrics_history)
        df.to_csv(self.csv_path, index=False)


class GradientClipping(Callback):
    """Gradient clipping callback."""
    
    def __init__(
        self,
        max_norm: float = 1.0,
        norm_type: float = 2.0
    ):
        """
        Initialize gradient clipping.
        
        Args:
            max_norm: Maximum gradient norm
            norm_type: Type of norm (2.0 for L2 norm)
        """
        super().__init__()
        self.max_norm = max_norm
        self.norm_type = norm_type
    
    def on_batch_end(self, trainer, batch_idx, loss):
        """Clip gradients after backward pass."""
        if hasattr(trainer, 'model'):
            torch.nn.utils.clip_grad_norm_(
                trainer.model.parameters(),
                max_norm=self.max_norm,
                norm_type=self.norm_type
            )


class CallbackList:
    """
    Container for managing multiple callbacks.
    """
    
    def __init__(self, callbacks: list):
        """
        Initialize callback list.
        
        Args:
            callbacks: List of callback instances
        """
        self.callbacks = callbacks if callbacks else []
    
    def on_train_begin(self, trainer):
        for callback in self.callbacks:
            callback.on_train_begin(trainer)
    
    def on_train_end(self, trainer):
        for callback in self.callbacks:
            callback.on_train_end(trainer)
    
    def on_epoch_begin(self, trainer, epoch):
        for callback in self.callbacks:
            callback.on_epoch_begin(trainer, epoch)
    
    def on_epoch_end(self, trainer, epoch, metrics):
        for callback in self.callbacks:
            callback.on_epoch_end(trainer, epoch, metrics)
    
    def on_batch_begin(self, trainer, batch_idx):
        for callback in self.callbacks:
            callback.on_batch_begin(trainer, batch_idx)
    
    def on_batch_end(self, trainer, batch_idx, loss):
        for callback in self.callbacks:
            callback.on_batch_end(trainer, batch_idx, loss)

