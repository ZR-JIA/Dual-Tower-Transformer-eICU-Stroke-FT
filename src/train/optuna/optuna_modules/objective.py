"""
Unified Optuna Objective

Handles training loop execution for any model type during hyperparameter optimization.
"""
import optuna 
import torch
import logging
import gc
import sys
from pathlib import Path

# Add parent path to ensure we can import from new package structure
sys.path.insert(0, str(Path(__file__).parents[2]))

from data_pipeline.loader import build_model
from engine.trainers import NeuralTrainer, TreeTrainer
from engine.metrics import compute_classification_metrics
from .spaces import suggest_hyperparameters

logger = logging.getLogger(__name__)

class UnifiedObjective:
    def __init__(self, base_config, data_module, device, model_name):
        self.base_config = base_config
        self.data_module = data_module
        self.device = device
        self.model_name = model_name
        self.trial_count = 0

    def cleanup(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            torch.mps.empty_cache()

    def __call__(self, trial):
        self.trial_count += 1
        
        # 1. Get Config for this trial
        try:
            trial_config = suggest_hyperparameters(self.model_name, self.base_config, trial)
        except ValueError as e:
            logger.error(str(e))
            raise optuna.exceptions.TrialPruned()

        logger.info(f"--- Trial {self.trial_count} ({self.model_name.upper()}) ---")
        
        try:
            # 2. Build Model
            input_dim = len(self.data_module.feature_names)
            model = build_model(trial_config, input_dim)

            # 3. Select Trainer & Train
            if self.model_name in ['mlp', 'nn', 'transformer', 'dualtower', 'dualtower_mlp', 'dualtower_ft', 'dualtower_ca', 'dualtower_as', 'dualtower_dcn']:
                # Neural Training
                trainer = NeuralTrainer(model, trial_config, self.device)
                
                # Get Loaders
                batch_size = trial_config['model_config'].get('train_loop', {}).get('batch_size', 512)
                train_loader, val_loader, _ = self.data_module.get_dataloaders(
                    batch_size=batch_size,
                    num_workers=trial_config['common']['resources']['num_workers'],
                    pin_memory=trial_config['common']['resources']['pin_memory']
                )
                
                # Train (with smaller epochs for search speed)
                search_epochs = 50 # Speed up search
                trainer.fit(train_loader, val_loader, epochs=search_epochs)
                
                # Predict
                y_val_pred = trainer.predict(val_loader)
                
            else:
                # Tree Training
                trainer = TreeTrainer(model, trial_config)
                X_train = self.data_module.X_train
                y_train = self.data_module.y_train
                X_val = self.data_module.X_val
                y_val = self.data_module.y_val
                
                trainer.fit(X_train, y_train, X_val, y_val)
                y_val_pred = trainer.predict(X_val)

            # 4. Compute Metrics
            y_val = self.data_module.y_val
            metrics = compute_classification_metrics(y_val, y_val_pred, threshold=0.5)
            val_auprc = metrics.get('auprc', 0.0)
            
            logger.info(f"Trial {self.trial_count} Result: AUPRC = {val_auprc:.4f}")

            # 5. Cleanup
            del model, trainer
            self.cleanup()
            
            return val_auprc

        except Exception as e:
            logger.error(f"Trial {self.trial_count} failed: {e}")
            self.cleanup()
            return 0.0