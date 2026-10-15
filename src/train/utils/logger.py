"""
Logging setup, git tracking, and system info utilities.
"""

import os
import sys
import json
import hashlib
import logging
import subprocess
import platform
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
import torch


class JSONFormatter(logging.Formatter):
    """Format log records as JSON."""
    
    def format(self, record):
        log_obj = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_obj['exception'] = self.formatException(record.exc_info)
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'created', 'filename', 'funcName',
                          'levelname', 'levelno', 'lineno', 'module', 'msecs',
                          'pathname', 'process', 'processName', 'relativeCreated',
                          'thread', 'threadName', 'exc_info', 'exc_text', 'stack_info']:
                log_obj[key] = value
        
        return json.dumps(log_obj)


def setup_logger(
    name: str,
    log_dir: Path,
    level: str = 'INFO',
    save_jsonl: bool = True,
    console: bool = True
) -> logging.Logger:
    """
    Setup logger with file and console handlers.
    
    Args:
        name: Logger name
        log_dir: Directory to save logs
        level: Logging level
        save_jsonl: Save structured JSONL logs
        console: Enable console output
    
    Returns:
        logging.Logger: Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    logger.handlers = []  # Clear existing handlers
    
    # Create log directory
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    
    # File handler (human-readable)
    log_file = log_dir / f'{name}_{timestamp}.log'
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # JSONL handler (structured)
    if save_jsonl:
        jsonl_file = log_dir / f'{name}_{timestamp}.jsonl'
        jsonl_handler = logging.FileHandler(jsonl_file)
        jsonl_handler.setLevel(logging.DEBUG)
        jsonl_handler.setFormatter(JSONFormatter())
        logger.addHandler(jsonl_handler)
    
    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level.upper()))
        console_formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    logger.info(f"Logger initialized: {name}")
    logger.info(f"Log file: {log_file}")
    if save_jsonl:
        logger.info(f"JSONL log: {jsonl_file}")
    
    return logger


def get_logger(name: str = __name__) -> logging.Logger:
    """Get logger by name."""
    return logging.getLogger(name)


def get_git_hash() -> Optional[str]:
    """
    Get current git commit hash.
    
    Returns:
        str or None: Git commit hash or None if not in git repo
    """
    try:
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True,
            text=True,
            check=True,
            timeout=5
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return None


def get_git_branch() -> Optional[str]:
    """Get current git branch name."""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            capture_output=True,
            text=True,
            check=True,
            timeout=5
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return None


def check_git_dirty() -> bool:
    """Check if git working directory is dirty (has uncommitted changes)."""
    try:
        result = subprocess.run(
            ['git', 'diff', '--quiet'],
            capture_output=True,
            timeout=5
        )
        return result.returncode != 0
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return False


def log_config_hash(config: Dict[str, Any]) -> str:
    """
    Compute and log hash of configuration.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        str: SHA256 hash of config
    """
    config_str = json.dumps(config, sort_keys=True)
    config_hash = hashlib.sha256(config_str.encode()).hexdigest()
    return config_hash


def log_system_info(logger: Optional[logging.Logger] = None) -> Dict[str, Any]:
    """
    Log system information.
    
    Args:
        logger: Logger instance (optional)
    
    Returns:
        dict: System information
    """
    info = {
        'platform': {
            'system': platform.system(),
            'release': platform.release(),
            'version': platform.version(),
            'machine': platform.machine(),
            'processor': platform.processor(),
            'python_version': platform.python_version(),
        },
        'torch': {
            'version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
            'cudnn_version': torch.backends.cudnn.version() if torch.cuda.is_available() else None,
            'num_gpus': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'mps_available': hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(),
        },
        'git': {
            'hash': get_git_hash(),
            'branch': get_git_branch(),
            'dirty': check_git_dirty(),
        }
    }
    
    # Add GPU details
    if torch.cuda.is_available():
        info['gpus'] = []
        for i in range(torch.cuda.device_count()):
            info['gpus'].append({
                'id': i,
                'name': torch.cuda.get_device_name(i),
                'memory_total': torch.cuda.get_device_properties(i).total_memory,
            })
    
    if logger:
        logger.info("="*60)
        logger.info("SYSTEM INFORMATION")
        logger.info("="*60)
        logger.info(f"OS: {info['platform']['system']} {info['platform']['release']}")
        logger.info(f"Python: {info['platform']['python_version']}")
        logger.info(f"PyTorch: {info['torch']['version']}")
        logger.info(f"CUDA Available: {info['torch']['cuda_available']}")
        if info['torch']['cuda_available']:
            logger.info(f"CUDA Version: {info['torch']['cuda_version']}")
            logger.info(f"GPUs: {info['torch']['num_gpus']}")
            for gpu in info['gpus']:
                logger.info(f"  - {gpu['name']} ({gpu['memory_total'] / 1e9:.1f} GB)")
        logger.info(f"MPS Available: {info['torch']['mps_available']}")
        logger.info(f"Git Hash: {info['git']['hash']}")
        logger.info(f"Git Branch: {info['git']['branch']}")
        if info['git']['dirty']:
            logger.warning("Git working directory is dirty (uncommitted changes)")
        logger.info("="*60)
    
    return info


def save_environment(output_path: Path):
    """
    Save pip freeze output to file.
    
    Args:
        output_path: Path to save requirements
    """
    try:
        result = subprocess.run(
            ['pip', 'freeze'],
            capture_output=True,
            text=True,
            check=True,
            timeout=30
        )
        output_path.write_text(result.stdout)
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as e:
        logger = get_logger(__name__)
        logger.warning(f"Failed to save environment: {e}")


class MetricsCSVLogger:
    """Logger for metrics in CSV format."""
    
    def __init__(self, log_path: Path):
        """
        Initialize CSV logger.
        
        Args:
            log_path: Path to CSV file
        """
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.header_written = False
    
    def log(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """
        Log metrics to CSV.
        
        Args:
            metrics: Dictionary of metrics
            step: Step number (optional)
        """
        # Add step if provided
        if step is not None:
            metrics = {'step': step, **metrics}
        
        # Write header if first time
        if not self.header_written:
            with open(self.log_path, 'w') as f:
                f.write(','.join(metrics.keys()) + '\n')
            self.header_written = True
        
        # Write metrics
        with open(self.log_path, 'a') as f:
            values = [str(v) for v in metrics.values()]
            f.write(','.join(values) + '\n')

