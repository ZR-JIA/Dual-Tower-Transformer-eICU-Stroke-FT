"""
LR scheduler builders from configuration.
"""

import torch
import torch.optim as optim
from typing import Dict, Any, Optional
import logging
import inspect
from .utils_cast import (
    safe_float, safe_int, safe_list_ints, validate_range
)

logger = logging.getLogger(__name__)


def _filter_scheduler_kwargs(scheduler_class, kwargs: dict) -> dict:
    """
    Filter kwargs to only include parameters supported by the scheduler.
    
    Args:
        scheduler_class: Scheduler class
        kwargs: Keyword arguments
    
    Returns:
        dict: Filtered kwargs
    """
    sig = inspect.signature(scheduler_class.__init__)
    valid_params = set(sig.parameters.keys()) - {'self'}
    
    filtered = {}
    ignored = []
    
    for key, value in kwargs.items():
        if key in valid_params:
            filtered[key] = value
        else:
            ignored.append(key)
    
    if ignored:
        logger.warning(f"Ignored unsupported scheduler parameters: {ignored}")
    
    return filtered


def get_scheduler(
    optimizer: optim.Optimizer,
    config: Dict[str, Any],
    steps_per_epoch: Optional[int] = None
) -> Optional[optim.lr_scheduler._LRScheduler]:
    """
    Get learning rate scheduler from configuration with safe type casting.
    
    Args:
        optimizer: Optimizer instance
        config: Full configuration dictionary (should contain 'model_config' and 'train' keys)
        steps_per_epoch: Number of steps per epoch (for step-based schedulers)
    
    Returns:
        torch.optim.lr_scheduler: Scheduler instance or None
    
    Raises:
        ValueError: If parameters are invalid or out of range
    """
    # Extract scheduler config from model_config if full config is passed
    if 'model_config' in config:
        sched_config = config['model_config'].get('scheduler', {})
    else:
        # Fallback to old behavior if only model_config is passed
        sched_config = config.get('scheduler', {})
    
    sched_name = sched_config.get('name', 'none').lower()
    
    if sched_name == 'none' or sched_name is None:
        logger.info("No learning rate scheduler")
        return None
    
    logger.info(f"Creating scheduler: {sched_name}")
    
    # Warmup epochs (if applicable)
    warmup_epochs = safe_int(sched_config.get('warmup_epochs', 0), 'scheduler.warmup_epochs')
    validate_range(warmup_epochs, 'scheduler.warmup_epochs', min_val=0, min_inclusive=True)
    
    if sched_name == 'cosine':
        cosine_config = sched_config.get('cosine', {})
        T_max = safe_int(cosine_config.get('T_max', config.get('train', {}).get('training', {}).get('max_epochs', 100)), 
                        'scheduler.cosine.T_max')
        eta_min = safe_float(cosine_config.get('eta_min', 0), 'scheduler.cosine.eta_min')
        
        validate_range(T_max, 'scheduler.cosine.T_max', min_val=1, min_inclusive=True)
        validate_range(eta_min, 'scheduler.cosine.eta_min', min_val=0, min_inclusive=True)
        
        if warmup_epochs > 0:
            # Create warmup + cosine annealing
            warmup_scheduler = optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=warmup_epochs
            )
            cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=T_max - warmup_epochs,
                eta_min=eta_min
            )
            scheduler = optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_epochs]
            )
            logger.info(f"  Warmup epochs: {warmup_epochs}")
            logger.info(f"  T_max: {T_max}, eta_min: {eta_min}")
        else:
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=T_max,
                eta_min=eta_min
            )
            logger.info(f"  T_max: {T_max}, eta_min: {eta_min}")
    
    elif sched_name == 'step':
        step_config = sched_config.get('step', {})
        step_size = safe_int(step_config.get('step_size', 30), 'scheduler.step.step_size')
        gamma = safe_float(step_config.get('gamma', 0.1), 'scheduler.step.gamma')
        last_epoch = safe_int(step_config.get('last_epoch', -1), 'scheduler.step.last_epoch') \
            if 'last_epoch' in step_config else -1
        
        validate_range(step_size, 'scheduler.step.step_size', min_val=1, min_inclusive=True)
        validate_range(gamma, 'scheduler.step.gamma', min_val=0, max_val=1, 
                      min_inclusive=False, max_inclusive=False)
        validate_range(last_epoch, 'scheduler.step.last_epoch', min_val=-1, min_inclusive=True)
        
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=step_size,
            gamma=gamma,
            last_epoch=last_epoch
        )
        logger.info(f"Scheduler=StepLR(step_size={step_size}, gamma={gamma})")
    
    elif sched_name == 'multistep':
        multistep_config = sched_config.get('multistep', {})
        milestones = safe_list_ints(multistep_config.get('milestones', [30, 60, 90]), 
                                    'scheduler.multistep.milestones', 
                                    strictly_increasing=True)
        gamma = safe_float(multistep_config.get('gamma', 0.1), 'scheduler.multistep.gamma')
        last_epoch = safe_int(multistep_config.get('last_epoch', -1), 'scheduler.multistep.last_epoch') \
            if 'last_epoch' in multistep_config else -1
        
        validate_range(gamma, 'scheduler.multistep.gamma', min_val=0, max_val=1, 
                      min_inclusive=False, max_inclusive=False)
        
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=milestones,
            gamma=gamma,
            last_epoch=last_epoch
        )
        logger.info(f"Scheduler=MultiStepLR(milestones={milestones}, gamma={gamma})")
    
    elif sched_name == 'exponential':
        exp_config = sched_config.get('exponential', {})
        gamma = safe_float(exp_config.get('gamma', 0.95), 'scheduler.exponential.gamma')
        last_epoch = safe_int(exp_config.get('last_epoch', -1), 'scheduler.exponential.last_epoch') \
            if 'last_epoch' in exp_config else -1
        
        validate_range(gamma, 'scheduler.exponential.gamma', min_val=0, max_val=1, 
                      min_inclusive=False, max_inclusive=False)
        
        scheduler = optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=gamma,
            last_epoch=last_epoch
        )
        logger.info(f"Scheduler=ExponentialLR(gamma={gamma})")
    
    elif sched_name == 'plateau' or sched_name == 'reduce_on_plateau':
        plateau_config = sched_config.get('plateau', {})
        mode = str(plateau_config.get('mode', 'max')).lower()
        factor = safe_float(plateau_config.get('factor', 0.5), 'scheduler.plateau.factor')
        patience = safe_int(plateau_config.get('patience', 5), 'scheduler.plateau.patience')
        threshold = safe_float(plateau_config.get('threshold', 1e-4), 'scheduler.plateau.threshold')
        threshold_mode = str(plateau_config.get('threshold_mode', 'rel')).lower() \
            if 'threshold_mode' in plateau_config else 'rel'
        cooldown = safe_int(plateau_config.get('cooldown', 0), 'scheduler.plateau.cooldown') \
            if 'cooldown' in plateau_config else 0
        min_lr = safe_float(plateau_config.get('min_lr', 1e-6), 'scheduler.plateau.min_lr')
        eps = safe_float(plateau_config.get('eps', 1e-8), 'scheduler.plateau.eps') \
            if 'eps' in plateau_config else 1e-8
        
        # Validate
        if mode not in ['min', 'max']:
            raise ValueError(f"scheduler.plateau.mode must be 'min' or 'max', got '{mode}'")
        if threshold_mode not in ['rel', 'abs']:
            raise ValueError(f"scheduler.plateau.threshold_mode must be 'rel' or 'abs', got '{threshold_mode}'")
        
        validate_range(factor, 'scheduler.plateau.factor', min_val=0, max_val=1, 
                      min_inclusive=False, max_inclusive=False)
        validate_range(patience, 'scheduler.plateau.patience', min_val=0, min_inclusive=True)
        validate_range(threshold, 'scheduler.plateau.threshold', min_val=0, min_inclusive=True)
        validate_range(cooldown, 'scheduler.plateau.cooldown', min_val=0, min_inclusive=True)
        validate_range(min_lr, 'scheduler.plateau.min_lr', min_val=0, min_inclusive=True)
        validate_range(eps, 'scheduler.plateau.eps', min_val=0, min_inclusive=False)
        
        # Filter parameters based on PyTorch version
        kwargs = {
            'mode': mode,
            'factor': factor,
            'patience': patience,
            'threshold': threshold,
            'threshold_mode': threshold_mode,
            'cooldown': cooldown,
            'min_lr': min_lr,
            'eps': eps,
            'verbose': True
        }
        kwargs = _filter_scheduler_kwargs(optim.lr_scheduler.ReduceLROnPlateau, kwargs)
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, **kwargs)
        logger.info(f"Scheduler=ReduceLROnPlateau(mode={mode}, factor={factor}, patience={patience}, "
                   f"threshold={threshold}, min_lr={min_lr})")
    
    elif sched_name == 'onecycle':
        onecycle_config = sched_config.get('onecycle', {})
        max_lr = config.get('model_config', {}).get('optimizer', {}).get('lr', 0.001)
        epochs = config.get('train', {}).get('training', {}).get('max_epochs', 100)
        
        if steps_per_epoch is None:
            raise ValueError("steps_per_epoch required for OneCycleLR")
        
        pct_start = onecycle_config.get('pct_start', 0.3)
        div_factor = onecycle_config.get('div_factor', 25.0)
        final_div_factor = onecycle_config.get('final_div_factor', 1e4)
        
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=pct_start,
            div_factor=div_factor,
            final_div_factor=final_div_factor
        )
        logger.info(f"  Max LR: {max_lr}, pct_start: {pct_start}")
        logger.info(f"  Div factor: {div_factor}, final div factor: {final_div_factor}")
    
    elif sched_name == 'cosine_warmup':
        # Custom cosine with warmup
        T_max = sched_config.get('T_max', config.get('train', {}).get('training', {}).get('max_epochs', 100))
        eta_min = sched_config.get('eta_min', 0)
        
        warmup_scheduler = optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_epochs
        )
        cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=T_max - warmup_epochs,
            eta_min=eta_min
        )
        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs]
        )
        logger.info(f"  Warmup: {warmup_epochs}, T_max: {T_max}, eta_min: {eta_min}")
    
    else:
        raise ValueError(f"Unknown scheduler: {sched_name}")
    
    return scheduler


class WarmupScheduler:
    """
    Simple warmup scheduler wrapper.
    
    Linearly increases learning rate from start_lr to base_lr
    over warmup_epochs.
    """
    
    def __init__(
        self,
        optimizer: optim.Optimizer,
        base_scheduler: Optional[optim.lr_scheduler._LRScheduler],
        warmup_epochs: int,
        start_factor: float = 0.1
    ):
        """
        Initialize warmup scheduler.
        
        Args:
            optimizer: Optimizer
            base_scheduler: Base scheduler to use after warmup
            warmup_epochs: Number of warmup epochs
            start_factor: Starting LR factor
        """
        self.optimizer = optimizer
        self.base_scheduler = base_scheduler
        self.warmup_epochs = warmup_epochs
        self.start_factor = start_factor
        self.current_epoch = 0
        
        # Store base learning rates
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
    
    def step(self, epoch: Optional[int] = None):
        """Step the scheduler."""
        if epoch is not None:
            self.current_epoch = epoch
        else:
            self.current_epoch += 1
        
        if self.current_epoch < self.warmup_epochs:
            # Warmup phase
            factor = self.start_factor + (1 - self.start_factor) * (self.current_epoch / self.warmup_epochs)
            for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                param_group['lr'] = base_lr * factor
        elif self.base_scheduler is not None:
            # Use base scheduler after warmup
            self.base_scheduler.step()
    
    def get_last_lr(self):
        """Get last learning rate."""
        return [group['lr'] for group in self.optimizer.param_groups]

