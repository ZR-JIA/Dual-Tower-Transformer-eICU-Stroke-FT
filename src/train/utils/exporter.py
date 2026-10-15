"""
Exporter Module

Handles artifact export and final model management:
- Save model, scaler, calibrator, config
- Dual-track export (latest + archived)
- SHA256 verification
- VERSION management with semver
- Complete reproducibility package
"""

import json
import shutil
import pickle
import hashlib
import joblib
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


def compute_file_sha256(file_path: Path) -> str:
    """Compute SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def compute_dict_hash(data: Dict) -> str:
    """Compute SHA256 hash of a dictionary."""
    json_str = json.dumps(data, sort_keys=True)
    return hashlib.sha256(json_str.encode()).hexdigest()


def parse_version(version_str: str) -> Tuple[int, int, int]:
    """Parse semver version string to tuple."""
    try:
        parts = version_str.lstrip('v').split('.')
        return tuple(int(p) for p in parts[:3])
    except:
        return (1, 0, 0)


def increment_version(version_str: str, bump: str = 'patch') -> str:
    """Increment semantic version."""
    major, minor, patch = parse_version(version_str)
    
    if bump == 'major':
        major += 1
        minor = 0
        patch = 0
    elif bump == 'minor':
        minor += 1
        patch = 0
    else:  # patch
        patch += 1
    
    return f"{major}.{minor}.{patch}"


class FinalModelExporter:
    """
    Exports final models with dual-track system:
    1. outputs/final/<Model>/ - Latest version (can overwrite)
    2. outputs/final_model/<Model>_<Split>_<Seed>_<Metric><Val>/ - Archived (immutable)
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        base_output_dir: str = "outputs"
    ):
        """
        Initialize exporter.
        
        Args:
            config: Full training configuration
            base_output_dir: Base output directory
        """
        self.config = config
        self.base_output_dir = Path(base_output_dir)
        self.model_name = config['model_config']['model'].upper()
        
    def export_final_model(
        self,
        model: Any,
        scaler: Optional[Any],
        calibrator: Optional[Any],
        feature_names: List[str],
        metrics: Dict[str, float],
        threshold: float,
        threshold_method: str,
        best_epoch: int,
        run_meta: Dict[str, Any]
    ) -> Tuple[Path, Path]:
        """
        Export final model to both latest and archived locations.
        
        Args:
            model: Trained model
            scaler: Fitted scaler (StandardScaler, MinMaxScaler, etc.)
            calibrator: Fitted calibrator
            feature_names: List of feature names in correct order
            metrics: Evaluation metrics
            threshold: Selected classification threshold
            threshold_method: Method used to select threshold
            best_epoch: Best epoch number
            run_meta: Run metadata
        
        Returns:
            Tuple[Path, Path]: (latest_dir, archived_dir)
        """
        # Prepare metadata
        split_type = run_meta.get('split_type', 'unknown')
        seed = run_meta.get('seed', 42)
        primary_metric = self.config['common']['promotion']['primary_metric']
        metric_value = metrics.get(primary_metric, 0.0)
        
        # Create directory names
        latest_dir = self.base_output_dir / 'final' / self.model_name
        
        archived_name = self._generate_archived_name(
            split_type, seed, primary_metric, metric_value
        )
        archived_dir = self.base_output_dir / 'final_model' / archived_name
        
        # Export to both locations
        for export_dir in [latest_dir, archived_dir]:
            self._export_to_directory(
                export_dir=export_dir,
                model=model,
                scaler=scaler,
                calibrator=calibrator,
                feature_names=feature_names,
                metrics=metrics,
                threshold=threshold,
                threshold_method=threshold_method,
                best_epoch=best_epoch,
                run_meta=run_meta,
                is_archived=(export_dir == archived_dir)
            )
        
        logger.info(f"✓ Final model exported:")
        logger.info(f"  Latest: {latest_dir}")
        logger.info(f"  Archived: {archived_dir}")
        
        return latest_dir, archived_dir
    
    def _generate_archived_name(
        self,
        split_type: str,
        seed: int,
        metric_name: str,
        metric_value: float
    ) -> str:
        """
        Generate archived directory name.
        Format: <Model>_<Split>_seed<Seed>_<Metric><Val>
        Example: MLP_patient_stratified_seed42_aupr0.412
        """
        metric_str = f"{metric_name}{metric_value:.3f}".replace('.', '')
        archived_name = f"{self.model_name}_{split_type}_seed{seed}_{metric_str}"
        return archived_name
    
    def _export_to_directory(
        self,
        export_dir: Path,
        model: Any,
        scaler: Optional[Any],
        calibrator: Optional[Any],
        feature_names: List[str],
        metrics: Dict[str, float],
        threshold: float,
        threshold_method: str,
        best_epoch: int,
        run_meta: Dict[str, Any],
        is_archived: bool
    ):
        """Export complete model package to a directory."""
        # Create directory
        export_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Exporting to: {export_dir}")
        
        # 1. Save model weights
        model_path = self._save_model(export_dir, model)
        
        # 2. Save preprocessing
        preprocess_spec = self._save_preprocessing(export_dir, scaler)
        
        # 3. Save calibration
        calibration_spec = self._save_calibration(export_dir, calibrator)
        
        # 4. Save feature list
        feature_list_path = export_dir / 'feature_list.json'
        with open(feature_list_path, 'w') as f:
            json.dump({
                'features': feature_names,
                'n_features': len(feature_names)
            }, f, indent=2)
        
        # 5. Save model config
        model_config_path = export_dir / 'model_config.json'
        with open(model_config_path, 'w') as f:
            json.dump(self.config['model_config'], f, indent=2)
        
        # 6. Save metrics
        metrics_path = export_dir / 'metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump({
                **metrics,
                'threshold': threshold,
                'threshold_method': threshold_method,
                'best_epoch': best_epoch
            }, f, indent=2)
        
        # 7. Save run metadata
        run_meta_path = export_dir / 'run_meta.json'
        with open(run_meta_path, 'w') as f:
            json.dump({
                **run_meta,
                'export_timestamp': datetime.now().isoformat(),
                'model_type': self.model_name,
                'is_archived': is_archived
            }, f, indent=2)
        
        # 8. Save config snapshot
        config_snapshot_path = export_dir / 'config_snapshot.yaml'
        import yaml
        with open(config_snapshot_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)
        
        # 9. Generate VERSION
        version = self._manage_version(export_dir)
        version_path = export_dir / 'VERSION'
        version_path.write_text(version)
        
        # 10. Generate SHA256 checksums
        self._generate_checksums(export_dir)
        
        # 11. Generate README
        self._generate_readme(export_dir, metrics, threshold, threshold_method, version)
        
        logger.info(f"✓ Export complete: {export_dir}")
    
    def _save_model(self, export_dir: Path, model: Any) -> Path:
        """Save model weights based on model type."""
        if self.model_name in ['MLP', 'NN', 'TRANSFORMER']:
            # PyTorch model
            model_path = export_dir / 'model_weights.pt'
            if hasattr(model, 'state_dict'):
                torch.save(model.state_dict(), model_path)
            else:
                torch.save(model, model_path)
            logger.info(f"  Saved PyTorch model: {model_path}")
            
        elif self.model_name == 'XGBOOST':
            # XGBoost model
            model_path = export_dir / 'model_weights.json'
            model.save_model(model_path)
            logger.info(f"  Saved XGBoost model: {model_path}")
            
        elif self.model_name == 'RANDOM_FOREST':
            # Random Forest model
            model_path = export_dir / 'model_weights.joblib'
            joblib.dump(model, model_path)
            logger.info(f"  Saved Random Forest model: {model_path}")
            
        else:
            # Generic pickle
            model_path = export_dir / 'model_weights.pkl'
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            logger.info(f"  Saved model (pickle): {model_path}")
        
        return model_path
    
    def _save_preprocessing(self, export_dir: Path, scaler: Optional[Any]) -> Dict:
        """Save preprocessing specifications."""
        preprocess_spec = {
            'scaler_type': None,
            'scaler_params': None
        }
        
        if scaler is not None:
            # Save scaler object
            scaler_path = export_dir / 'scaler.pkl'
            with open(scaler_path, 'wb') as f:
                pickle.dump(scaler, f)
            
            # Extract scaler info
            scaler_type = type(scaler).__name__
            preprocess_spec['scaler_type'] = scaler_type
            
            if hasattr(scaler, 'mean_'):
                preprocess_spec['scaler_params'] = {
                    'mean': scaler.mean_.tolist() if hasattr(scaler.mean_, 'tolist') else None,
                    'scale': scaler.scale_.tolist() if hasattr(scaler, 'scale_') and hasattr(scaler.scale_, 'tolist') else None,
                    'var': scaler.var_.tolist() if hasattr(scaler, 'var_') and hasattr(scaler.var_, 'tolist') else None
                }
            elif hasattr(scaler, 'min_'):
                preprocess_spec['scaler_params'] = {
                    'min': scaler.min_.tolist(),
                    'scale': scaler.scale_.tolist()
                }
            
            logger.info(f"  Saved scaler: {scaler_path}")
        
        # Save preprocessing spec
        preprocess_path = export_dir / 'preprocess_spec.json'
        with open(preprocess_path, 'w') as f:
            json.dump(preprocess_spec, f, indent=2)
        
        return preprocess_spec
    
    def _save_calibration(self, export_dir: Path, calibrator: Optional[Any]) -> Dict:
        """Save calibration specifications."""
        calibration_spec = {
            'enabled': calibrator is not None,
            'method': None
        }
        
        if calibrator is not None:
            # Save calibrator object
            calibrator_path = export_dir / 'calibrator.pkl'
            calibrator.save(str(calibrator_path))
            
            calibration_spec['method'] = calibrator.method
            logger.info(f"  Saved calibrator: {calibrator_path}")
        
        # Save calibration spec
        calibration_path = export_dir / 'calibration.json'
        with open(calibration_path, 'w') as f:
            json.dump(calibration_spec, f, indent=2)
        
        return calibration_spec
    
    def _manage_version(self, export_dir: Path) -> str:
        """
        Manage version with semver auto-increment.
        
        Returns:
            str: Version string (e.g., "1.0.0")
        """
        version_file = export_dir / 'VERSION'
        
        if version_file.exists():
            current_version = version_file.read_text().strip()
            new_version = increment_version(current_version, bump='patch')
        else:
            new_version = "1.0.0"
        
        return new_version
    
    def _generate_checksums(self, export_dir: Path):
        """Generate SHA256 checksums for all important files."""
        checksums = {}
        
        # Files to checksum
        files_to_check = [
            'model_weights.*',
            'preprocess_spec.json',
            'feature_list.json',
            'calibration.json',
            'model_config.json'
        ]
        
        for pattern in files_to_check:
            for file_path in export_dir.glob(pattern):
                if file_path.is_file():
                    rel_path = file_path.relative_to(export_dir)
                    checksums[str(rel_path)] = compute_file_sha256(file_path)
        
        # Save checksums
        sha256_path = export_dir / 'SHA256.txt'
        with open(sha256_path, 'w') as f:
            f.write("# SHA256 Checksums\n")
            f.write(f"# Generated: {datetime.now().isoformat()}\n\n")
            for file_name, checksum in sorted(checksums.items()):
                f.write(f"{checksum}  {file_name}\n")
        
        logger.info(f"  Generated checksums: {sha256_path}")
    
    def _generate_readme(
        self,
        export_dir: Path,
        metrics: Dict[str, float],
        threshold: float,
        threshold_method: str,
        version: str
    ):
        """Generate README for the model package."""
        readme_path = export_dir / 'README.md'
        
        primary_metric = self.config['common']['promotion']['primary_metric']
        
        readme_content = f"""# {self.model_name} Model Package

**Version:** {version}  
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Primary Metric:** {primary_metric.upper()} = {metrics.get(primary_metric, 0.0):.4f}

## Model Performance

| Metric | Value |
|--------|-------|
| AUROC | {metrics.get('auroc', 0.0):.4f} |
| AUPRC | {metrics.get('auprc', 0.0):.4f} |
| Brier Score | {metrics.get('brier', 0.0):.4f} |
| ECE | {metrics.get('ece', 0.0):.4f} |
| F1 Score | {metrics.get('f1', 0.0):.4f} |
| Precision | {metrics.get('precision', 0.0):.4f} |
| Recall | {metrics.get('recall', 0.0):.4f} |
| Specificity | {metrics.get('specificity', 0.0):.4f} |

**Classification Threshold:** {threshold:.4f} (method: {threshold_method})

## Package Contents

- `model_weights.*` - Trained model weights
- `feature_list.json` - Feature names in correct order
- `preprocess_spec.json` - Preprocessing specifications
- `calibration.json` - Calibration settings
- `model_config.json` - Model configuration
- `metrics.json` - Complete evaluation metrics
- `run_meta.json` - Training run metadata
- `config_snapshot.yaml` - Full configuration snapshot
- `SHA256.txt` - File checksums for verification
- `VERSION` - Semantic version
- `README.md` - This file

## Usage

```python
import json
import pickle
import torch

# Load feature list
with open('feature_list.json') as f:
    feature_info = json.load(f)
    feature_names = feature_info['features']

# Load model
model = torch.load('model_weights.pt')  # Or appropriate loader for model type

# Load preprocessing
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Load calibrator (if exists)
with open('calibrator.pkl', 'rb') as f:
    calibrator = pickle.load(f)

# Make predictions
# X must have columns in the order specified in feature_list.json
X_scaled = scaler.transform(X)
predictions = model.predict(X_scaled)
calibrated_probs = calibrator.transform(predictions)
```

## Verification

Verify file integrity using SHA256 checksums:
```bash
sha256sum -c SHA256.txt
```

## Configuration

See `config_snapshot.yaml` for complete training configuration.

## Metadata

See `run_meta.json` for detailed training run metadata including:
- Seed used
- Split strategy
- Class imbalance handling
- Training duration
- Hardware used
"""
        
        readme_path.write_text(readme_content)
        logger.info(f"  Generated README: {readme_path}")


def export_artifacts(
    exp_dir: Path,
    config: Dict[str, Any],
    model: Any,
    scaler: Optional[Any],
    calibrator: Optional[Any],
    metrics: Dict[str, float],
    feature_names: List[str]
):
    """
    Legacy convenience function - exports to experiment directory only.
    """
    artifacts_dir = exp_dir / 'artifacts'
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Exporting experiment artifacts to: {artifacts_dir}")
    
    # Save all artifacts
    model_path = artifacts_dir / 'model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    if scaler is not None:
        scaler_path = artifacts_dir / 'scaler.pkl'
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
    
    if calibrator is not None:
        calibrator_path = artifacts_dir / 'calibrator.pkl'
        calibrator.save(str(calibrator_path))
    
    features_path = artifacts_dir / 'feature_names.json'
    with open(features_path, 'w') as f:
        json.dump({'features': feature_names}, f, indent=2)
    
    metrics_path = artifacts_dir / 'metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    config_path = artifacts_dir / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info("✓ Experiment artifacts exported")


def update_leaderboard(
    leaderboard_path: Path,
    model_name: str,
    experiment_name: str,
    split_type: str,
    seed: int,
    metrics: Dict[str, float],
    version: str,
    archived_path: str,
    timestamp: str
):
    """
    Update leaderboard CSV with new model results.
    
    Args:
        leaderboard_path: Path to leaderboard.csv
        model_name: Model name
        experiment_name: Experiment identifier
        split_type: Split strategy used
        seed: Random seed
        metrics: All metrics
        version: Model version
        archived_path: Path to archived model
        timestamp: Timestamp of training
    """
    import pandas as pd
    
    # Create new row
    new_row = {
        'timestamp': timestamp,
        'model': model_name,
        'experiment': experiment_name,
        'split': split_type,
        'seed': seed,
        'version': version,
        'archived_path': archived_path,
        **{k: v for k, v in metrics.items() if isinstance(v, (int, float))}
    }
    
    # Load existing leaderboard or create new
    if leaderboard_path.exists():
        df = pd.read_csv(leaderboard_path)
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    else:
        df = pd.DataFrame([new_row])
    
    # Sort by primary metric (AUPRC by default)
    primary_metric = 'auprc'
    if primary_metric in df.columns:
        df = df.sort_values(by=primary_metric, ascending=False)
    
    # Save
    df.to_csv(leaderboard_path, index=False)
    logger.info(f"✓ Leaderboard updated: {leaderboard_path}")


def update_final_model(*args, **kwargs):
    """
    Legacy compatibility function.
    Use FinalModelExporter.export_final_model() instead.
    """
    logger.warning("update_final_model() is deprecated. Use FinalModelExporter.export_final_model()")
    return False
