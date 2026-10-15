"""
Tiered config system: _base/ defaults merged with per-model overrides,
with ${section.key} variable resolution.

    config/
    ├── _base/          # 00_global, 01_data, 02_train
    └── models/         # mlp.yaml, transformer.yaml, ...
"""

import os
import re
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from collections import OrderedDict

logger = logging.getLogger(__name__)


class ConfigManager:
    """
    Singleton Configuration Manager.
    
    Handles tiered configuration loading with Base + Model structure
    and robust variable resolution.
    """
    
    _instance = None
    _initialized = False
    
    def __new__(cls, config_dir: Optional[Path] = None):
        """Singleton pattern: only one instance exists."""
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, config_dir: Optional[Path] = None):
        """
        Initialize ConfigManager.
        
        Args:
            config_dir: Path to config directory (defaults to script_dir/config)
        """
        # Only initialize once
        if ConfigManager._initialized:
            return
        
        # Set config directory
        if config_dir is None:
            # Default: config/ relative to this file's parent directory
            config_dir = Path(__file__).parent.parent / 'config'
        
        self.config_dir = Path(config_dir)
        self.base_dir = self.config_dir / '_base'
        self.models_dir = self.config_dir / 'models'
        
        # Cache for loaded configs
        self._base_config_cache = None
        self._model_config_cache = {}
        
        # Validation
        self._validate_structure()
        
        ConfigManager._initialized = True
        logger.info(f"ConfigManager initialized with config_dir: {self.config_dir}")
    
    def _validate_structure(self):
        """Validate that config directory structure exists."""
        if not self.config_dir.exists():
            raise FileNotFoundError(f"Config directory not found: {self.config_dir}")
        
        if not self.base_dir.exists():
            logger.warning(f"Base config directory not found: {self.base_dir}")
            logger.warning("Expected structure: config/_base/ and config/models/")
        
        if not self.models_dir.exists():
            logger.warning(f"Models config directory not found: {self.models_dir}")
    
    def load_base_config(self) -> Dict[str, Any]:
        """
        Load all base configurations.
        
        Loads all YAML files in config/_base/ in alphabetical order
        (hence the 00_, 01_, 02_ prefixes ensure correct load order).
        
        Returns:
            dict: Merged base configuration
        """
        if self._base_config_cache is not None:
            logger.debug("Using cached base config")
            return self._base_config_cache
        
        logger.info("Loading base configuration...")
        base_config = {}
        
        # Load all base config files in order
        if self.base_dir.exists():
            base_files = sorted(self.base_dir.glob('*.yaml'))
            
            if not base_files:
                logger.warning(f"No base config files found in {self.base_dir}")
            
            for config_file in base_files:
                logger.info(f"  Loading: {config_file.name}")
                with open(config_file, 'r') as f:
                    partial_config = yaml.safe_load(f)
                    if partial_config:
                        base_config = self._deep_merge(base_config, partial_config)
        else:
            logger.warning("Base config directory does not exist, using empty base")
        
        # Cache the result
        self._base_config_cache = base_config
        logger.info(f"Base config loaded with {len(base_config)} top-level keys")
        
        return base_config
    
    def load_model_config(self, model_name: str) -> Dict[str, Any]:
        """
        Load model-specific configuration.
        
        Args:
            model_name: Name of model (e.g., 'mlp', 'transformer', 'xgboost')
        
        Returns:
            dict: Model-specific configuration
        """
        # Check cache
        if model_name in self._model_config_cache:
            logger.debug(f"Using cached config for model: {model_name}")
            return self._model_config_cache[model_name]
        
        # Find model config file
        model_file = self.models_dir / f"{model_name}.yaml"
        
        if not model_file.exists():
            raise FileNotFoundError(
                f"Model config not found: {model_file}\n"
                f"Available models: {self.list_available_models()}"
            )
        
        logger.info(f"Loading model config: {model_name}")
        with open(model_file, 'r') as f:
            model_config = yaml.safe_load(f)
        
        if not model_config:
            raise ValueError(f"Empty config file: {model_file}")
        
        # Cache the result
        self._model_config_cache[model_name] = model_config
        
        return model_config
    
    def load_config(
        self,
        model_name: str,
        resolve_variables: bool = True,
        max_resolve_iterations: int = 5
    ) -> Dict[str, Any]:
        """
        Load complete configuration for a model.
        
        This is the main entry point. It:
        1. Loads all base configs
        2. Loads model-specific config
        3. Merges model config on top of base
        4. Resolves ${...} variable references
        
        Args:
            model_name: Name of model
            resolve_variables: Whether to resolve ${...} references
            max_resolve_iterations: Maximum iterations for variable resolution
        
        Returns:
            dict: Complete merged and resolved configuration
        """
        logger.info("="*80)
        logger.info(f"Loading configuration for model: {model_name.upper()}")
        logger.info("="*80)
        
        # Step 1: Load base config
        config = self.load_base_config()
        
        # Step 2: Load model config
        model_config = self.load_model_config(model_name)
        
        # Step 3: Merge (model config takes precedence)
        config = self._deep_merge(config, model_config)
        
        # Step 4: Resolve variables
        if resolve_variables:
            logger.info("Resolving variable references...")
            config = self.resolve_variables(config, max_iterations=max_resolve_iterations)
            logger.info("✓ Variable resolution complete")
        
        logger.info("="*80)
        logger.info(f"Configuration loaded successfully for: {model_name.upper()}")
        logger.info("="*80)
        
        return config
    
    def resolve_variables(
        self,
        config: Dict[str, Any],
        max_iterations: int = 5
    ) -> Dict[str, Any]:
        """
        Resolve ${section.key} variable references in configuration.
        
        Supports:
        - ${section.key} - references to other config values
        - Nested references: ${paths.root}/data
        - Multiple levels of indirection
        
        Args:
            config: Configuration dictionary
            max_iterations: Maximum resolution passes (prevents infinite loops)
        
        Returns:
            dict: Configuration with all variables resolved
        """
        for iteration in range(max_iterations):
            config_before = str(config)  # Simple change detection
            config = self._resolve_pass(config, config)
            config_after = str(config)
            
            # Check if anything changed
            if config_before == config_after:
                logger.debug(f"Variable resolution converged after {iteration + 1} iteration(s)")
                return config
        
        # Check if there are still unresolved variables
        remaining = self._find_unresolved_variables(config)
        if remaining:
            logger.warning(
                f"Variable resolution did not fully converge after {max_iterations} iterations. "
                f"Unresolved: {remaining[:5]}"  # Show first 5
            )
        
        return config
    
    def _resolve_pass(self, value: Any, root_config: Dict) -> Any:
        """
        Single pass of variable resolution (recursive).
        
        Args:
            value: Current value to resolve
            root_config: Root configuration for lookups
        
        Returns:
            Resolved value
        """
        if isinstance(value, str):
            return self._resolve_string(value, root_config)
        elif isinstance(value, dict):
            return {k: self._resolve_pass(v, root_config) for k, v in value.items()}
        elif isinstance(value, list):
            return [self._resolve_pass(item, root_config) for item in value]
        else:
            return value
    
    def _resolve_string(self, value: str, root_config: Dict) -> Any:
        """
        Resolve variables in a string value.
        
        Examples:
            "${common.seed}" -> 42
            "${paths.root}/data" -> "/path/to/project/data"
        """
        # Pattern: ${section.key.subkey}
        pattern = r'\$\{([^}]+)\}'
        matches = re.findall(pattern, value)
        
        if not matches:
            return value
        
        # If the entire string is a single variable reference, return the actual value
        # (preserves type - e.g., int, float, dict)
        if len(matches) == 1 and value == f"${{{matches[0]}}}":
            resolved = self._get_nested_value(root_config, matches[0])
            return resolved if resolved is not None else value
        
        # Otherwise, do string substitution
        result = value
        for match in matches:
            resolved = self._get_nested_value(root_config, match)
            if resolved is not None:
                result = result.replace(f"${{{match}}}", str(resolved))
        
        return result
    
    def _get_nested_value(self, config: Dict, path: str) -> Any:
        """
        Get value from nested dict using dot notation.
        
        Args:
            config: Configuration dictionary
            path: Dot-separated path (e.g., "common.reproducibility.seed")
        
        Returns:
            Value at path, or None if not found
        """
        keys = path.split('.')
        value = config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                logger.debug(f"Could not resolve path: {path} (failed at {key})")
                return None
        
        return value
    
    def _find_unresolved_variables(self, config: Dict) -> List[str]:
        """Find any remaining ${...} patterns in config."""
        unresolved = []
        
        def search(value, path=""):
            if isinstance(value, str):
                matches = re.findall(r'\$\{([^}]+)\}', value)
                for match in matches:
                    unresolved.append(f"{path}: ${{{match}}}")
            elif isinstance(value, dict):
                for k, v in value.items():
                    search(v, f"{path}.{k}" if path else k)
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    search(item, f"{path}[{i}]")
        
        search(config)
        return unresolved
    
    def _deep_merge(self, base: Dict, override: Dict) -> Dict:
        """
        Deep merge two dictionaries.
        
        Override values take precedence over base values.
        Nested dictionaries are merged recursively.
        
        Args:
            base: Base configuration
            override: Override configuration
        
        Returns:
            dict: Merged configuration
        """
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                # Recursive merge for nested dicts
                result[key] = self._deep_merge(result[key], value)
            else:
                # Override takes precedence
                result[key] = value
        
        return result
    
    def list_available_models(self) -> List[str]:
        """List all available model configurations."""
        if not self.models_dir.exists():
            return []
        
        model_files = self.models_dir.glob('*.yaml')
        return sorted([f.stem for f in model_files])
    
    def get_config_summary(self, config: Dict) -> str:
        """
        Generate a human-readable summary of configuration.
        
        Args:
            config: Configuration dictionary
        
        Returns:
            str: Formatted summary
        """
        lines = []
        lines.append("Configuration Summary:")
        lines.append("-" * 60)
        
        # Key sections to summarize
        if 'common' in config:
            if 'reproducibility' in config['common']:
                seed = config['common']['reproducibility'].get('seed', 'N/A')
                lines.append(f"  Seed: {seed}")
        
        if 'model_config' in config:
            model_type = config['model_config'].get('model', 'N/A')
            lines.append(f"  Model: {model_type}")
            
            if 'architecture' in config['model_config']:
                arch = config['model_config']['architecture']
                lines.append(f"  Architecture: {list(arch.keys())}")
        
        if 'train' in config and 'training' in config['train']:
            epochs = config['train']['training'].get('max_epochs', 'N/A')
            lines.append(f"  Max Epochs: {epochs}")
        
        if 'data' in config:
            target = config['data'].get('target_col', 'N/A')
            lines.append(f"  Target: {target}")
        
        lines.append("-" * 60)
        
        return "\n".join(lines)
    
    def clear_cache(self):
        """Clear all cached configurations."""
        self._base_config_cache = None
        self._model_config_cache.clear()
        logger.info("Configuration cache cleared")
    
    @staticmethod
    def reset_singleton():
        """Reset singleton (useful for testing)."""
        ConfigManager._instance = None
        ConfigManager._initialized = False


# Convenience function for quick access
def get_config_manager(config_dir: Optional[Path] = None) -> ConfigManager:
    """
    Get ConfigManager singleton instance.
    
    Args:
        config_dir: Optional config directory path
    
    Returns:
        ConfigManager: Singleton instance
    """
    return ConfigManager(config_dir)


def load_model_config(model_name: str) -> Dict[str, Any]:
    """
    Convenience function to load a model config.
    
    Args:
        model_name: Name of model
    
    Returns:
        dict: Complete merged configuration
    """
    manager = get_config_manager()
    return manager.load_config(model_name)
