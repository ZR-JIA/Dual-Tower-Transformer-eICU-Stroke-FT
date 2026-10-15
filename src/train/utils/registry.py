"""
Model registry for dynamic component registration and lookup.
"""

from typing import Dict, Callable, Any, Optional, List
import logging

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Registry for model builders and associated components."""
    
    def __init__(self):
        self._models: Dict[str, Callable] = {}
        self._loss_functions: Dict[str, Callable] = {}
        self._optimizers: Dict[str, Callable] = {}
        self._schedulers: Dict[str, Callable] = {}
        self._trainers: Dict[str, Callable] = {}
    
    def register_model(
        self,
        name: str,
        builder: Callable,
        force: bool = False
    ):
        """
        Register a model builder.
        
        Args:
            name: Model name/identifier
            builder: Function that builds the model
            force: Overwrite existing registration
        """
        if name in self._models and not force:
            raise ValueError(f"Model '{name}' already registered. Use force=True to overwrite.")
        self._models[name] = builder
        logger.debug(f"Registered model: {name}")
    
    def register_loss(self, name: str, loss_fn: Callable, force: bool = False):
        """Register a loss function."""
        if name in self._loss_functions and not force:
            raise ValueError(f"Loss '{name}' already registered.")
        self._loss_functions[name] = loss_fn
        logger.debug(f"Registered loss: {name}")
    
    def register_optimizer(self, name: str, opt_fn: Callable, force: bool = False):
        """Register an optimizer."""
        if name in self._optimizers and not force:
            raise ValueError(f"Optimizer '{name}' already registered.")
        self._optimizers[name] = opt_fn
        logger.debug(f"Registered optimizer: {name}")
    
    def register_scheduler(self, name: str, sched_fn: Callable, force: bool = False):
        """Register a learning rate scheduler."""
        if name in self._schedulers and not force:
            raise ValueError(f"Scheduler '{name}' already registered.")
        self._schedulers[name] = sched_fn
        logger.debug(f"Registered scheduler: {name}")
    
    def register_trainer(self, name: str, trainer_fn: Callable, force: bool = False):
        """Register a trainer class/builder."""
        if name in self._trainers and not force:
            raise ValueError(f"Trainer '{name}' already registered.")
        self._trainers[name] = trainer_fn
        logger.debug(f"Registered trainer: {name}")
    
    def get_model(self, name: str) -> Callable:
        """Get model builder by name."""
        if name not in self._models:
            raise ValueError(f"Model '{name}' not registered. Available: {self.list_models()}")
        return self._models[name]
    
    def get_loss(self, name: str) -> Callable:
        """Get loss function by name."""
        if name not in self._loss_functions:
            raise ValueError(f"Loss '{name}' not registered. Available: {list(self._loss_functions.keys())}")
        return self._loss_functions[name]
    
    def get_optimizer(self, name: str) -> Callable:
        """Get optimizer by name."""
        if name not in self._optimizers:
            raise ValueError(f"Optimizer '{name}' not registered. Available: {list(self._optimizers.keys())}")
        return self._optimizers[name]
    
    def get_scheduler(self, name: str) -> Callable:
        """Get scheduler by name."""
        if name not in self._schedulers:
            raise ValueError(f"Scheduler '{name}' not registered. Available: {list(self._schedulers.keys())}")
        return self._schedulers[name]
    
    def get_trainer(self, name: str) -> Callable:
        """Get trainer by name."""
        if name not in self._trainers:
            raise ValueError(f"Trainer '{name}' not registered. Available: {list(self._trainers.keys())}")
        return self._trainers[name]
    
    def list_models(self) -> List[str]:
        """List all registered models."""
        return list(self._models.keys())
    
    def list_losses(self) -> List[str]:
        """List all registered loss functions."""
        return list(self._loss_functions.keys())
    
    def list_optimizers(self) -> List[str]:
        """List all registered optimizers."""
        return list(self._optimizers.keys())
    
    def list_schedulers(self) -> List[str]:
        """List all registered schedulers."""
        return list(self._schedulers.keys())
    
    def list_trainers(self) -> List[str]:
        """List all registered trainers."""
        return list(self._trainers.keys())


# Global registry instance
_REGISTRY = ModelRegistry()


# Convenience functions
def register_model(name: str, builder: Callable, force: bool = False):
    """Register a model in the global registry."""
    _REGISTRY.register_model(name, builder, force)


def register_loss(name: str, loss_fn: Callable, force: bool = False):
    """Register a loss function in the global registry."""
    _REGISTRY.register_loss(name, loss_fn, force)


def register_optimizer(name: str, opt_fn: Callable, force: bool = False):
    """Register an optimizer in the global registry."""
    _REGISTRY.register_optimizer(name, opt_fn, force)


def register_scheduler(name: str, sched_fn: Callable, force: bool = False):
    """Register a scheduler in the global registry."""
    _REGISTRY.register_scheduler(name, sched_fn, force)


def register_trainer(name: str, trainer_fn: Callable, force: bool = False):
    """Register a trainer in the global registry."""
    _REGISTRY.register_trainer(name, trainer_fn, force)


def get_model(name: str) -> Callable:
    """Get model builder from the global registry."""
    return _REGISTRY.get_model(name)


def get_loss(name: str) -> Callable:
    """Get loss function from the global registry."""
    return _REGISTRY.get_loss(name)


def get_optimizer(name: str) -> Callable:
    """Get optimizer from the global registry."""
    return _REGISTRY.get_optimizer(name)


def get_scheduler(name: str) -> Callable:
    """Get scheduler from the global registry."""
    return _REGISTRY.get_scheduler(name)


def get_trainer(name: str) -> Callable:
    """Get trainer from the global registry."""
    return _REGISTRY.get_trainer(name)


def list_models() -> List[str]:
    """List all registered models."""
    return _REGISTRY.list_models()


def list_losses() -> List[str]:
    """List all registered losses."""
    return _REGISTRY.list_losses()


def list_optimizers() -> List[str]:
    """List all registered optimizers."""
    return _REGISTRY.list_optimizers()


def list_schedulers() -> List[str]:
    """List all registered schedulers."""
    return _REGISTRY.list_schedulers()


def list_trainers() -> List[str]:
    """List all registered trainers."""
    return _REGISTRY.list_trainers()


# Decorator for easy registration
def register(registry_type: str, name: str, force: bool = False):
    """
    Decorator for registering components.
    
    Usage:
        @register('model', 'mlp')
        def build_mlp(config):
            ...
    
    Args:
        registry_type: Type of component ('model', 'loss', 'optimizer', 'scheduler', 'trainer')
        name: Component name
        force: Overwrite existing registration
    """
    def decorator(func: Callable) -> Callable:
        if registry_type == 'model':
            register_model(name, func, force)
        elif registry_type == 'loss':
            register_loss(name, func, force)
        elif registry_type == 'optimizer':
            register_optimizer(name, func, force)
        elif registry_type == 'scheduler':
            register_scheduler(name, func, force)
        elif registry_type == 'trainer':
            register_trainer(name, func, force)
        else:
            raise ValueError(f"Unknown registry type: {registry_type}")
        return func
    return decorator

