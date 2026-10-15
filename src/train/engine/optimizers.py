"""
Optimizer builders from configuration.
"""

import torch
import torch.optim as optim
from typing import Iterator, Dict, Any, List
import logging
from .utils_cast import (
    safe_float, safe_int, safe_bool, safe_tuple_floats, validate_range
)

logger = logging.getLogger(__name__)


def get_optimizer(
    parameters: Iterator[torch.nn.Parameter],
    config: Dict[str, Any]
) -> optim.Optimizer:
    """
    Get optimizer from configuration with safe type casting and validation.
    
    Args:
        parameters: Model parameters
        config: Configuration dictionary
    
    Returns:
        torch.optim.Optimizer: Optimizer instance
    
    Raises:
        ValueError: If parameters are invalid or out of range
    """
    opt_config = config.get('optimizer', {})
    
    opt_name = opt_config.get('name', 'adamw').lower()
    
    # Safe cast common parameters
    lr = safe_float(opt_config.get('lr', 1e-3), 'optimizer.lr')
    weight_decay = safe_float(opt_config.get('weight_decay', 0.0), 'optimizer.weight_decay')
    
    # Validate ranges
    validate_range(lr, 'optimizer.lr', min_val=0.0, min_inclusive=False)
    validate_range(weight_decay, 'optimizer.weight_decay', min_val=0.0, min_inclusive=True)
    
    logger.info(f"Creating optimizer: {opt_name}")
    
    if opt_name == 'adam':
        # Safe cast Adam-specific parameters
        betas = safe_tuple_floats(
            opt_config.get('betas', (0.9, 0.999)), 
            'optimizer.betas', 
            expected_len=2
        )
        eps = safe_float(opt_config.get('eps', 1e-8), 'optimizer.eps')
        amsgrad = safe_bool(opt_config.get('amsgrad', False), 'optimizer.amsgrad') \
            if 'amsgrad' in opt_config else False
        
        # Validate betas in (0, 1)
        for i, beta in enumerate(betas):
            validate_range(beta, f'optimizer.betas[{i}]', 
                         min_val=0.0, max_val=1.0, 
                         min_inclusive=False, max_inclusive=False)
        validate_range(eps, 'optimizer.eps', min_val=0.0, min_inclusive=False)
        
        optimizer = optim.Adam(
            parameters,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad
        )
        logger.info(f"Optimizer=Adam(lr={lr}, weight_decay={weight_decay}, "
                   f"betas={betas}, eps={eps}, amsgrad={amsgrad})")
    
    elif opt_name == 'adamw':
        betas = safe_tuple_floats(
            opt_config.get('betas', (0.9, 0.999)), 
            'optimizer.betas', 
            expected_len=2
        )
        eps = safe_float(opt_config.get('eps', 1e-8), 'optimizer.eps')
        amsgrad = safe_bool(opt_config.get('amsgrad', False), 'optimizer.amsgrad') \
            if 'amsgrad' in opt_config else False
        
        # Validate betas in (0, 1)
        for i, beta in enumerate(betas):
            validate_range(beta, f'optimizer.betas[{i}]', 
                         min_val=0.0, max_val=1.0, 
                         min_inclusive=False, max_inclusive=False)
        validate_range(eps, 'optimizer.eps', min_val=0.0, min_inclusive=False)
        
        optimizer = optim.AdamW(
            parameters,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad
        )
        logger.info(f"Optimizer=AdamW(lr={lr}, weight_decay={weight_decay}, "
                   f"betas={betas}, eps={eps}, amsgrad={amsgrad})")
    
    elif opt_name == 'sgd':
        momentum = safe_float(opt_config.get('momentum', 0.9), 'optimizer.momentum')
        nesterov = safe_bool(opt_config.get('nesterov', True), 'optimizer.nesterov') \
            if 'nesterov' in opt_config else True
        dampening = safe_float(opt_config.get('dampening', 0.0), 'optimizer.dampening') \
            if 'dampening' in opt_config else 0.0
        
        # Validate momentum in [0, 1)
        validate_range(momentum, 'optimizer.momentum', 
                      min_val=0.0, max_val=1.0, 
                      min_inclusive=True, max_inclusive=False)
        validate_range(dampening, 'optimizer.dampening', min_val=0.0, min_inclusive=True)
        
        optimizer = optim.SGD(
            parameters,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
            dampening=dampening
        )
        logger.info(f"Optimizer=SGD(lr={lr}, momentum={momentum}, "
                   f"weight_decay={weight_decay}, nesterov={nesterov})")
    
    elif opt_name == 'rmsprop':
        alpha = safe_float(opt_config.get('alpha', 0.99), 'optimizer.alpha')
        eps = safe_float(opt_config.get('eps', 1e-8), 'optimizer.eps')
        momentum = safe_float(opt_config.get('momentum', 0.0), 'optimizer.momentum')
        centered = safe_bool(opt_config.get('centered', False), 'optimizer.centered') \
            if 'centered' in opt_config else False
        
        # Validate alpha in (0, 1)
        validate_range(alpha, 'optimizer.alpha', 
                      min_val=0.0, max_val=1.0, 
                      min_inclusive=False, max_inclusive=False)
        validate_range(eps, 'optimizer.eps', min_val=0.0, min_inclusive=False)
        validate_range(momentum, 'optimizer.momentum', min_val=0.0, min_inclusive=True)
        
        optimizer = optim.RMSprop(
            parameters,
            lr=lr,
            alpha=alpha,
            eps=eps,
            weight_decay=weight_decay,
            momentum=momentum,
            centered=centered
        )
        logger.info(f"Optimizer=RMSprop(lr={lr}, alpha={alpha}, eps={eps}, "
                   f"momentum={momentum}, weight_decay={weight_decay})")
    
    elif opt_name == 'adagrad':
        eps = safe_float(opt_config.get('eps', 1e-10), 'optimizer.eps')
        lr_decay = safe_float(opt_config.get('lr_decay', 0.0), 'optimizer.lr_decay') \
            if 'lr_decay' in opt_config else 0.0
        
        validate_range(eps, 'optimizer.eps', min_val=0.0, min_inclusive=False)
        validate_range(lr_decay, 'optimizer.lr_decay', min_val=0.0, min_inclusive=True)
        
        optimizer = optim.Adagrad(
            parameters,
            lr=lr,
            weight_decay=weight_decay,
            eps=eps,
            lr_decay=lr_decay
        )
        logger.info(f"Optimizer=Adagrad(lr={lr}, weight_decay={weight_decay}, "
                   f"eps={eps}, lr_decay={lr_decay})")
    
    elif opt_name == 'adadelta':
        rho = safe_float(opt_config.get('rho', 0.9), 'optimizer.rho')
        eps = safe_float(opt_config.get('eps', 1e-6), 'optimizer.eps')
        
        # Validate rho in (0, 1)
        validate_range(rho, 'optimizer.rho', 
                      min_val=0.0, max_val=1.0, 
                      min_inclusive=False, max_inclusive=False)
        validate_range(eps, 'optimizer.eps', min_val=0.0, min_inclusive=False)
        
        optimizer = optim.Adadelta(
            parameters,
            lr=lr,
            rho=rho,
            eps=eps,
            weight_decay=weight_decay
        )
        logger.info(f"Optimizer=Adadelta(lr={lr}, rho={rho}, eps={eps}, "
                   f"weight_decay={weight_decay})")
    
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}. Supported: adam, adamw, sgd, rmsprop, adagrad, adadelta")
    
    return optimizer


def get_parameter_groups(
    model: torch.nn.Module,
    config: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Create parameter groups for differential learning rates.
    
    Args:
        model: Model instance
        config: Configuration dictionary
    
    Returns:
        list: List of parameter group dictionaries
    """
    param_groups_config = config.get('optimizer', {}).get('param_groups', {})
    
    if not param_groups_config.get('enabled', False):
        # Return all parameters in single group
        return [{'params': model.parameters()}]
    
    # Define parameter groups
    # Example: different LR for different layers
    groups = param_groups_config.get('groups', [])
    
    param_groups = []
    for group_config in groups:
        # Filter parameters by name pattern
        pattern = group_config.get('pattern', None)
        lr_multiplier = group_config.get('lr_multiplier', 1.0)
        weight_decay = group_config.get('weight_decay', None)
        
        params = []
        for name, param in model.named_parameters():
            if pattern is None or pattern in name:
                params.append(param)
        
        if params:
            group = {'params': params, 'lr_multiplier': lr_multiplier}
            if weight_decay is not None:
                group['weight_decay'] = weight_decay
            param_groups.append(group)
            logger.info(f"Parameter group: pattern={pattern}, lr_mult={lr_multiplier}, params={len(params)}")
    
    # Add remaining parameters
    assigned_params = set()
    for group in param_groups:
        assigned_params.update(id(p) for p in group['params'])
    
    remaining_params = [p for p in model.parameters() if id(p) not in assigned_params]
    if remaining_params:
        param_groups.append({'params': remaining_params})
        logger.info(f"Parameter group: remaining params={len(remaining_params)}")
    
    return param_groups


def adjust_learning_rate(
    optimizer: optim.Optimizer,
    epoch: int,
    config: Dict[str, Any]
) -> float:
    """
    Adjust learning rate based on epoch (manual scheduling).
    
    Args:
        optimizer: Optimizer instance
        epoch: Current epoch
        config: Configuration dictionary
    
    Returns:
        float: New learning rate
    """
    # This is a fallback for manual LR adjustment
    # Prefer using scheduler from schedulers.py
    
    base_lr = config.get('optimizer', {}).get('lr', 1e-3)
    
    # Example: decay by 0.5 every 30 epochs
    decay_factor = 0.5
    decay_epochs = 30
    
    lr = base_lr * (decay_factor ** (epoch // decay_epochs))
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    return lr


def get_current_lr(optimizer: optim.Optimizer) -> float:
    """
    Get current learning rate from optimizer.
    
    Args:
        optimizer: Optimizer instance
    
    Returns:
        float: Current learning rate
    """
    return optimizer.param_groups[0]['lr']

