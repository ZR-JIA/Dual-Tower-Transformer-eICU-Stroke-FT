"""
Model Factory — registry-based model instantiation.
"""

import logging
from typing import Dict, Any, Callable
from sklearn.ensemble import RandomForestClassifier

try:
    import xgboost as xgb
except ImportError:
    xgb = None

from utils.config_utils import safe_cast

logger = logging.getLogger(__name__)


class ModelFactory:
    """
    Factory for creating models using registry pattern.
    
    Supports 5 model types:
    - Neural: nn, mlp, transformer
    - Tree: xgboost, random_forest
    
    Usage:
        factory = ModelFactory()
        model = factory.build_model('mlp', config, input_dim=94)
    """
    
    # Model type categories
    NEURAL_MODELS = {'mlp', 'nn', 'transformer', 'dualtower', 'dualtower_mlp', 'dualtower_ca', 'dualtower_ft', 'dualtower_ft_ma', 'dualtower_as', 'dualtower_dcn'}
    TREE_MODELS = {'xgboost', 'random_forest'}
    
    def __init__(self):
        """Initialize factory with empty registry."""
        self._registry: Dict[str, Callable] = {}
        self._register_default_models()
    
    def _register_default_models(self):
        """Register all default model builders."""
        # Neural models
        self.register('nn', self._build_nn)
        self.register('mlp', self._build_mlp)
        self.register('transformer', self._build_transformer)
        
        # Dual-Tower models
        self.register('dualtower', self._build_dualtower)
        self.register('dualtower_mlp', self._build_dualtower_mlp)
        self.register('dualtower_ca', self._build_dualtower_ca)
        self.register('dualtower_ft', self._build_dualtower_ft)
        self.register('dualtower_ft_ma', self._build_dualtower_ft)  # same arch, different config
        self.register('dualtower_as', self._build_dualtower_as)
        self.register('dualtower_dcn', self._build_dualtower_dcn)
        
        # Tree models
        self.register('xgboost', self._build_xgboost)
        self.register('random_forest', self._build_random_forest)
    
    def register(self, name: str, builder: Callable):
        """
        Register a model builder.
        
        Args:
            name: Model name (lowercase)
            builder: Function(config, input_dim) -> model
        """
        self._registry[name] = builder
        logger.debug(f"Registered model: {name}")
    
    def build_model(
        self,
        model_name: str,
        config: Dict[str, Any],
        input_dim: int
    ) -> Any:
        """
        Build a model by name.
        
        Args:
            model_name: Name of model (e.g., 'mlp', 'xgboost')
            config: Configuration dictionary
            input_dim: Input feature dimension
        
        Returns:
            Model instance
        
        Raises:
            ValueError: If model not registered
        """
        if model_name not in self._registry:
            available = sorted(self._registry.keys())
            raise ValueError(
                f"Model '{model_name}' not registered in ModelFactory.\n"
                f"Available models: {available}\n"
                f"Check your config or register the model using factory.register()"
            )
        
        logger.info(f"Building model: {model_name}")
        builder = self._registry[model_name]
        model = builder(config, input_dim)
        logger.info(f"✅ Model '{model_name}' built successfully")
        
        return model
    
    def list_models(self):
        """List all registered models."""
        return sorted(self._registry.keys())
    
    def is_neural_model(self, model_name: str) -> bool:
        """Check if model is a neural network."""
        return model_name in self.NEURAL_MODELS
    
    def is_tree_model(self, model_name: str) -> bool:
        """Check if model is tree-based."""
        return model_name in self.TREE_MODELS
    
    # ========================================================================
    # MODEL BUILDERS: NEURAL NETWORKS
    # ========================================================================
    
    def _build_nn(self, config: Dict[str, Any], input_dim: int):
        """Build Simple Neural Network."""
        from architectures.models.mlp import SimpleNN
        
        arch_config = config['model_config'].get('architecture', {})
        model = SimpleNN(
            input_dim=input_dim,
            hidden_dim=arch_config.get('hidden_dim', 128),
            dropout=arch_config.get('dropout', 0.3)
        )
        
        n_params = sum(p.numel() for p in model.parameters())
        logger.info(f"NN: {n_params:,} parameters")
        return model
    
    def _build_mlp(self, config: Dict[str, Any], input_dim: int):
        """Build Multi-Layer Perceptron."""
        from architectures.models.mlp import MLPModel
        
        arch_config = config['model_config'].get('architecture', {})
        model = MLPModel(
            input_dim=input_dim,
            hidden_dims=arch_config.get('hidden_dims', [256, 128, 64]),
            dropout=arch_config.get('dropout', 0.2),
            batch_norm=arch_config.get('batch_norm', True),
            activation=arch_config.get('activation', 'relu')
        )
        
        n_params = sum(p.numel() for p in model.parameters())
        logger.info(f"MLP: {n_params:,} parameters")
        return model
    
    def _build_transformer(self, config: Dict[str, Any], input_dim: int):
        """Build Transformer model."""
        from architectures.models.transformer import TransformerModel
        
        arch_config = config['model_config'].get('architecture', {})
        model = TransformerModel(
            input_dim=input_dim,
            d_model=arch_config.get('d_model', 128),
            nhead=arch_config.get('nhead', 4),
            num_layers=arch_config.get('num_layers', 2),
            dim_feedforward=arch_config.get('dim_feedforward', 256),
            dropout=arch_config.get('dropout', 0.1)
        )
        
        n_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Transformer: {n_params:,} parameters")
        return model
    
    # ========================================================================
    # MODEL BUILDERS: DUAL-TOWER
    # ========================================================================
    
    def _build_dualtower(self, config: Dict[str, Any], input_dim: int):
        """Build DualTower model with Transformer right tower."""
        from architectures.models.dualtower import DualTower
        
        model_config = config['model_config']
        model = DualTower(model_config, input_dim)
        
        n_params = sum(p.numel() for p in model.parameters())
        logger.info(f"DualTower (Transformer): {n_params:,} parameters")
        return model
    
    def _build_dualtower_mlp(self, config: Dict[str, Any], input_dim: int):
        """Build DualTower model with MLP right tower (for ablation study)."""
        from architectures.models.dualtower import DualTower
        
        model_config = config['model_config']
        model = DualTower(model_config, input_dim)
        
        n_params = sum(p.numel() for p in model.parameters())
        logger.info(f"DualTower (MLP): {n_params:,} parameters")
        return model
    
    def _build_dualtower_ca(self, config: Dict[str, Any], input_dim: int):
        """Build DualTower with Cross-Attention Fusion."""
        from architectures.models.dualtower_ca import DualTowerCA
        
        model_config = config['model_config']
        model = DualTowerCA(model_config, input_dim)
        
        n_params = sum(p.numel() for p in model.parameters())
        logger.info(f"DualTowerCA (Cross-Attention): {n_params:,} parameters")
        return model

    def _build_dualtower_ft(self, config: Dict[str, Any], input_dim: int):
        """Build DualTower with FT-Transformer Right Tower (CLS aggregation)."""
        from architectures.models.dualtower_ft import DualTowerFT
        
        model_config = config['model_config']
        model = DualTowerFT(model_config, input_dim)
        
        n_params = sum(p.numel() for p in model.parameters())
        logger.info(f"DualTowerFT (FT-Transformer): {n_params:,} parameters")
        return model

    def _build_dualtower_dcn(self, config: Dict[str, Any], input_dim: int):
        """Build DualTower with Demographic-Conditioned Numerical Attention (DT-DC)."""
        from architectures.models.dualtower_dcn import DualTowerDCN

        model_config = config['model_config']
        model = DualTowerDCN(model_config, input_dim)

        n_params = sum(p.numel() for p in model.parameters())
        logger.info(f"DualTowerDCN (DT-FT + Demographic Cross-Attention): {n_params:,} parameters")
        return model

    def _build_dualtower_as(self, config: Dict[str, Any], input_dim: int):
        """Build DualTower with Adaptive Safeguard Mechanism 3 (LearnableSoftNorm)."""
        from architectures.models.dualtower_as import DualTowerAS
        
        model_config = config['model_config']
        model = DualTowerAS(model_config, input_dim)
        
        n_params = sum(p.numel() for p in model.parameters())
        logger.info(f"DualTowerAS (DT + Adaptive Safeguard M3): {n_params:,} parameters")
        return model

    # ========================================================================
    # MODEL BUILDERS: TREE-BASED
    # ========================================================================
    
    def _build_xgboost(self, config: Dict[str, Any], input_dim: int):
        """Build XGBoost model."""
        if xgb is None:
            raise ImportError(
                "XGBoost not installed. Install with: pip install xgboost"
            )
        
        params = config['model_config']['params'].copy()
        
        # Type casting for XGBoost
        if 'random_state' in params:
            params['random_state'] = safe_cast(params['random_state'], int, 'xgboost.random_state')
        if 'n_estimators' in params:
            params['n_estimators'] = safe_cast(params['n_estimators'], int, 'xgboost.n_estimators')
        if 'max_depth' in params and params['max_depth'] is not None:
            params['max_depth'] = safe_cast(params['max_depth'], int, 'xgboost.max_depth')
        if 'learning_rate' in params:
            params['learning_rate'] = safe_cast(params['learning_rate'], float, 'xgboost.learning_rate')
        if 'scale_pos_weight' in params:
            params['scale_pos_weight'] = safe_cast(params['scale_pos_weight'], float, 'xgboost.scale_pos_weight')
        
        model = xgb.XGBClassifier(**params)
        logger.info(f"XGBoost: {params.get('n_estimators', 100)} estimators")
        return model
    
    def _build_random_forest(self, config: Dict[str, Any], input_dim: int):
        """Build Random Forest model."""
        params = config['model_config']['params'].copy()
        
        # Type casting for Random Forest
        if 'random_state' in params:
            params['random_state'] = safe_cast(params['random_state'], int, 'rf.random_state')
        if 'n_estimators' in params:
            params['n_estimators'] = safe_cast(params['n_estimators'], int, 'rf.n_estimators')
        if 'max_depth' in params and params['max_depth'] is not None:
            params['max_depth'] = safe_cast(params['max_depth'], int, 'rf.max_depth')
        if 'min_samples_split' in params:
            if isinstance(params['min_samples_split'], str):
                try:
                    params['min_samples_split'] = safe_cast(params['min_samples_split'], int, 'rf.min_samples_split')
                except:
                    params['min_samples_split'] = safe_cast(params['min_samples_split'], float, 'rf.min_samples_split')
        if 'min_samples_leaf' in params:
            if isinstance(params['min_samples_leaf'], str):
                try:
                    params['min_samples_leaf'] = safe_cast(params['min_samples_leaf'], int, 'rf.min_samples_leaf')
                except:
                    params['min_samples_leaf'] = safe_cast(params['min_samples_leaf'], float, 'rf.min_samples_leaf')
        
        model = RandomForestClassifier(**params)
        logger.info(f"RandomForest: {params.get('n_estimators', 100)} estimators")
        return model


# Global factory instance (singleton pattern)
_FACTORY = ModelFactory()


def get_model_factory() -> ModelFactory:
    """Get global ModelFactory instance."""
    return _FACTORY


def build_model(model_name: str, config: Dict[str, Any], input_dim: int):
    """
    Convenience function to build a model.
    
    Args:
        model_name: Model name
        config: Configuration
        input_dim: Input dimension
    
    Returns:
        Model instance
    """
    return _FACTORY.build_model(model_name, config, input_dim)
