"""
Search Spaces Definition
Defines hyperparameter search spaces for all models.
"""
import copy
import optuna
from typing import Dict, Any

def suggest_hyperparameters(model_name: str, config: Dict[str, Any], trial: optuna.Trial) -> Dict[str, Any]:
    """Factory function to get config with suggested parameters."""
    
    # Create deep copy to avoid modifying original config
    trial_config = copy.deepcopy(config)
    
    if model_name == 'mlp':
        # Architecture
        num_layers = trial.suggest_int('num_layers', 2, 4)
        hidden_dim = trial.suggest_categorical('hidden_dim', [128, 256, 512])
        dropout = trial.suggest_float('dropout', 0.1, 0.5)
        activation = trial.suggest_categorical('activation', ['relu', 'gelu', 'silu'])
        
        # Optimizer
        lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
        
        # Apply to config
        trial_config['model_config']['architecture']['hidden_dims'] = [hidden_dim] * num_layers
        trial_config['model_config']['architecture']['dropout'] = dropout
        trial_config['model_config']['architecture']['activation'] = activation
        trial_config['model_config']['optimizer']['lr'] = lr

    elif model_name == 'transformer':
        # Architecture
        d_model = trial.suggest_categorical('d_model', [64, 128, 256])
        nhead = trial.suggest_categorical('nhead', [2, 4, 8])
        num_layers = trial.suggest_int('num_layers', 1, 4)
        dim_feedforward = trial.suggest_categorical('dim_feedforward', [128, 256, 512])
        dropout = trial.suggest_float('dropout', 0.1, 0.5)
        
        # Optimizer
        lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
        weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)

        # Apply
        trial_config['model_config']['architecture']['d_model'] = d_model
        trial_config['model_config']['architecture']['nhead'] = nhead
        trial_config['model_config']['architecture']['num_layers'] = num_layers
        trial_config['model_config']['architecture']['dim_feedforward'] = dim_feedforward
        trial_config['model_config']['architecture']['dropout'] = dropout
        trial_config['model_config']['optimizer']['lr'] = lr
        trial_config['model_config']['optimizer']['weight_decay'] = weight_decay

    elif model_name == 'xgboost':
        # Tree params
        n_estimators = trial.suggest_int('n_estimators', 500, 2000, step=100)
        max_depth = trial.suggest_int('max_depth', 3, 8)
        lr = trial.suggest_float('learning_rate', 0.01, 0.1, log=True)
        subsample = trial.suggest_float('subsample', 0.6, 1.0)
        colsample_bytree = trial.suggest_float('colsample_bytree', 0.6, 1.0)
        reg_lambda = trial.suggest_float('reg_lambda', 0.0, 10.0)
        reg_alpha = trial.suggest_float('reg_alpha', 0.0, 1.0)
        
        # Apply
        params = trial_config['model_config']['params']
        params['n_estimators'] = n_estimators
        params['max_depth'] = max_depth
        params['learning_rate'] = lr
        params['subsample'] = subsample
        params['colsample_bytree'] = colsample_bytree
        params['reg_lambda'] = reg_lambda
        params['reg_alpha'] = reg_alpha

    elif model_name == 'random_forest':
        n_estimators = trial.suggest_int('n_estimators', 500, 2000, step=100)
        max_depth = trial.suggest_int('max_depth', 5, 30, step=5)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
        
        params = trial_config['model_config']['params']
        params['n_estimators'] = n_estimators
        params['max_depth'] = max_depth
        params['min_samples_split'] = min_samples_split
        
    elif model_name == 'nn':
        # --- Step A: Structure Adaptation ---
        # 1. If optimizer is at root level in YAML, move it under 'train'
        if 'optimizer' in trial_config and 'optimizer' not in trial_config.get('train', {}):
            if 'train' not in trial_config: trial_config['train'] = {}
            trial_config['train']['optimizer'] = trial_config['optimizer']
            
        # 2. Ensure train.optimizer exists; create a default if missing
        if 'train' not in trial_config: trial_config['train'] = {}
        if 'optimizer' not in trial_config['train']:
            trial_config['train']['optimizer'] = {'name': 'adam', 'lr': 1e-3}

        # 3. Map architecture to model_config (adapt for Builder)
        if 'model_config' not in trial_config: trial_config['model_config'] = {}
        # Force type to 'mlp' (the underlying code uses MLPModel for NN)
        trial_config['model_config']['type'] = 'mlp' 
        
        # If YAML has 'architecture', copy its params as defaults
        if 'architecture' in trial_config:
            trial_config['model_config'].update(trial_config['architecture'])


        # --- Step B: Hyperparameter Sampling ---
        # 1. Search learning rate (overrides YAML default of 1e-3)
        lr = trial.suggest_float('optimizer.lr', 1e-4, 1e-2, log=True)
        trial_config['train']['optimizer']['lr'] = lr
        
        # 2. Search hidden dim (overrides YAML default of 128)
        # NN is a baseline, so we search a smaller range (16-64) to differentiate from Deep MLP
        hidden_dim = trial.suggest_int('model.hidden_dim', 16, 64)
        trial_config['model_config']['hidden_dim'] = hidden_dim
        
        # 3. Search num_layers (1-2 layers to keep "Simple" positioning)
        num_layers = trial.suggest_int('model.num_layers', 1, 2)
        trial_config['model_config']['num_layers'] = num_layers
        
        # 4. Dropout (fine-tune around YAML default of 0.3)
        dropout = trial.suggest_float('model.dropout', 0.1, 0.5)
        trial_config['model_config']['dropout'] = dropout
        
        # 5. Ensure input_dim exists (prevent KeyError)
        trial_config['model_config']['input_dim'] = 94

    elif model_name == 'dualtower':
        # DualTower with Transformer right tower
        trans_d_model = trial.suggest_categorical('trans_d_model', [64, 128, 256])
        trans_nhead = trial.suggest_categorical('trans_nhead', [2, 4, 8])
        trans_num_layers = trial.suggest_int('trans_num_layers', 1, 4)
        dropout = trial.suggest_float('dropout', 0.1, 0.5)
        fusion_hidden_dim = trial.suggest_categorical('fusion_hidden_dim', [64, 128, 256])
        
        # Optimizer
        lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
        weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)
        
        # Apply
        arch = trial_config['model_config']['architecture']
        arch['trans_d_model'] = trans_d_model
        arch['trans_nhead'] = trans_nhead
        arch['trans_num_layers'] = trans_num_layers
        arch['dropout'] = dropout
        arch['fusion_hidden_dim'] = fusion_hidden_dim
        trial_config['model_config']['optimizer']['lr'] = lr
        trial_config['model_config']['optimizer']['weight_decay'] = weight_decay

    elif model_name == 'dualtower_mlp':
        # DualTower with MLP right tower
        right_tower_hidden_dim = trial.suggest_categorical('right_tower_hidden_dim', [128, 256, 512])
        dropout = trial.suggest_float('dropout', 0.1, 0.5)
        fusion_hidden_dim = trial.suggest_categorical('fusion_hidden_dim', [64, 128, 256])
        
        # Optimizer
        lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
        weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)
        
        # Apply
        arch = trial_config['model_config']['architecture']
        arch['right_tower_hidden_dim'] = right_tower_hidden_dim
        arch['dropout'] = dropout
        arch['fusion_hidden_dim'] = fusion_hidden_dim
        trial_config['model_config']['optimizer']['lr'] = lr
        trial_config['model_config']['optimizer']['weight_decay'] = weight_decay

    elif model_name == 'dualtower_ft':
        # FT-Transformer right tower: CLS token, FFN 4×
        # d_model search extended vs dualtower (FT-Transformer benefits from wider models)
        trans_d_model = trial.suggest_categorical('trans_d_model', [64, 128, 192, 256])
        trans_nhead = trial.suggest_categorical('trans_nhead', [4, 8])
        trans_num_layers = trial.suggest_int('trans_num_layers', 1, 4)
        dropout = trial.suggest_float('dropout', 0.05, 0.4)
        fusion_hidden_dim = trial.suggest_categorical('fusion_hidden_dim', [64, 128, 256])

        # Optimizer
        lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
        weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)

        # Apply
        arch = trial_config['model_config']['architecture']
        arch['trans_d_model'] = trans_d_model
        arch['trans_nhead'] = trans_nhead
        arch['trans_num_layers'] = trans_num_layers
        arch['dropout'] = dropout
        arch['fusion_hidden_dim'] = fusion_hidden_dim
        trial_config['model_config']['optimizer']['lr'] = lr
        trial_config['model_config']['optimizer']['weight_decay'] = weight_decay

    elif model_name == 'dualtower_dcn':
        trans_d_model = trial.suggest_categorical('trans_d_model', [64, 128, 192, 256])
        trans_nhead = trial.suggest_categorical('trans_nhead', [4, 8])
        trans_num_layers = trial.suggest_int('trans_num_layers', 1, 4)
        dropout = trial.suggest_float('dropout', 0.05, 0.4)
        fusion_hidden_dim = trial.suggest_categorical('fusion_hidden_dim', [64, 128, 256])

        lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
        weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)

        arch = trial_config['model_config']['architecture']
        arch['trans_d_model'] = trans_d_model
        arch['trans_nhead'] = trans_nhead
        arch['trans_num_layers'] = trans_num_layers
        arch['dropout'] = dropout
        arch['fusion_hidden_dim'] = fusion_hidden_dim
        trial_config['model_config']['optimizer']['lr'] = lr
        trial_config['model_config']['optimizer']['weight_decay'] = weight_decay

    elif model_name in ('dualtower_ca', 'dualtower_as'):
        # Same architecture parameters as dualtower — reuse that search space
        trans_d_model = trial.suggest_categorical('trans_d_model', [64, 128, 256])
        trans_nhead = trial.suggest_categorical('trans_nhead', [2, 4, 8])
        trans_num_layers = trial.suggest_int('trans_num_layers', 1, 4)
        dropout = trial.suggest_float('dropout', 0.1, 0.5)
        fusion_hidden_dim = trial.suggest_categorical('fusion_hidden_dim', [64, 128, 256])

        lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
        weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)

        arch = trial_config['model_config']['architecture']
        arch['trans_d_model'] = trans_d_model
        arch['trans_nhead'] = trans_nhead
        arch['trans_num_layers'] = trans_num_layers
        arch['dropout'] = dropout
        arch['fusion_hidden_dim'] = fusion_hidden_dim
        trial_config['model_config']['optimizer']['lr'] = lr
        trial_config['model_config']['optimizer']['weight_decay'] = weight_decay

    else:
        raise ValueError(f"No search space defined for model: {model_name}")
        
    return trial_config