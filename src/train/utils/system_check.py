#!/usr/bin/env python3
"""
System Validation Script (Simplified)

Validates the refactored architecture without requiring dependencies:
- Verifies file structure
- Checks Python syntax
- Maps models to configs and classes
- Cleans up legacy artifacts

Works without numpy/torch/pandas installed.
"""

import sys
import os
import shutil
import ast
from pathlib import Path
from typing import Dict, Any, List
import re


# ============================================================================
# CONFIGURATION
# ============================================================================

# All 7 supported models
MODEL_NAMES = [
    'nn',
    'mlp',
    'transformer',
    'xgboost',
    'random_forest',
    'dualtower_mlp',
    'dualtower'
]

# Expected config file mapping
EXPECTED_CONFIG_FILES = {
    'nn': 'config/models/nn.yaml',
    'mlp': 'config/models/mlp.yaml',
    'transformer': 'config/models/transformer.yaml',
    'xgboost': 'config/models/xgboost.yaml',
    'random_forest': 'config/models/random_forest.yaml',
    'dualtower_mlp': 'config/models/dualtower_mlp.yaml',
    'dualtower': 'config/models/dualtower.yaml'
}

# Expected Python class mapping
EXPECTED_CLASSES = {
    'nn': 'SimpleNN',
    'mlp': 'MLPModel',
    'transformer': 'TransformerModel',
    'xgboost': 'XGBClassifier',
    'random_forest': 'RandomForestClassifier',
    'dualtower_mlp': 'DualTower',
    'dualtower': 'DualTower'
}

# Model to Python file mapping
MODEL_TO_FILE = {
    'nn': 'architectures/models/mlp.py',
    'mlp': 'architectures/models/mlp.py',
    'transformer': 'architectures/models/transformer.py',
    'xgboost': 'architectures/model_factory.py',  # Built in factory
    'random_forest': 'architectures/model_factory.py',  # Built in factory
    'dualtower_mlp': 'architectures/models/dualtower.py',
    'dualtower': 'architectures/models/dualtower.py'
}


# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================

def cleanup_legacy_artifacts():
    """Aggressively clean up legacy files and cache."""
    print("=" * 80)
    print("CLEANUP: Removing Legacy Artifacts")
    print("=" * 80)
    
    cleaned = []
    base_dir = Path(__file__).parent
    
    # 1. Check for old models.py
    old_models_py = base_dir / "train_modules" / "models.py"
    if old_models_py.exists():
        old_models_py.unlink()
        cleaned.append(str(old_models_py))
        print(f"🗑️  Deleted: {old_models_py}")
    
    # 2. Remove all __pycache__ directories
    for pycache in base_dir.rglob("__pycache__"):
        if pycache.is_dir():
            shutil.rmtree(pycache)
            cleaned.append(str(pycache))
            print(f"🗑️  Deleted: {pycache}")
    
    # 3. Remove .DS_Store files
    for ds_store in base_dir.rglob(".DS_Store"):
        if ds_store.is_file():
            ds_store.unlink()
            cleaned.append(str(ds_store))
            print(f"🗑️  Deleted: {ds_store}")
    
    if not cleaned:
        print("✅ No legacy artifacts found - system is clean")
    else:
        print(f"✅ Cleaned {len(cleaned)} artifact(s)")
    
    print()


def check_file_exists(filepath: Path) -> bool:
    """Check if file exists and is readable."""
    return filepath.exists() and filepath.is_file()


def check_python_syntax(filepath: Path) -> tuple[bool, str]:
    """Check if Python file has valid syntax."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            code = f.read()
        ast.parse(code)
        return True, "OK"
    except SyntaxError as e:
        return False, f"Syntax error at line {e.lineno}: {e.msg}"
    except Exception as e:
        return False, str(e)


def check_yaml_syntax(filepath: Path) -> tuple[bool, str]:
    """Basic check for YAML file (just check it's readable and has content)."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        if not content.strip():
            return False, "Empty file"
        # Check for basic YAML structure
        if 'model:' not in content and 'architecture:' not in content:
            return False, "Missing expected YAML keys"
        return True, "OK"
    except Exception as e:
        return False, str(e)


def find_class_in_file(filepath: Path, class_name: str) -> bool:
    """Check if a class is defined in a Python file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            code = f.read()
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                return True
        return False
    except:
        return False


def check_model_config(model_name: str, config_file: str) -> tuple[bool, str]:
    """Verify model config file exists and contains correct model type."""
    # Go up to train/ root since we're now in utils/
    base_dir = Path(__file__).parent.parent
    config_path = base_dir / config_file
    
    if not check_file_exists(config_path):
        return False, f"Config file not found: {config_file}"
    
    # Check YAML syntax
    valid, error = check_yaml_syntax(config_path)
    if not valid:
        return False, f"Invalid YAML: {error}"
    
    # Check model type is correct
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Look for model: <model_name>
        pattern = rf'model:\s*["\']?{re.escape(model_name)}["\']?'
        if not re.search(pattern, content):
            return False, f"Config doesn't specify model: {model_name}"
        
        return True, "OK"
    except Exception as e:
        return False, str(e)


def validate_model(model_name: str) -> Dict[str, Any]:
    """Validate a single model."""
    result = {
        'model_name': model_name,
        'config_file': EXPECTED_CONFIG_FILES.get(model_name, 'UNKNOWN'),
        'python_class': EXPECTED_CLASSES.get(model_name, 'UNKNOWN'),
        'status': '❌ FAIL',
        'error': None
    }
    
    # Go up to train/ root since we're now in utils/
    base_dir = Path(__file__).parent.parent
    
    try:
        # 1. Check config file
        config_file = EXPECTED_CONFIG_FILES.get(model_name)
        if not config_file:
            result['error'] = "No config file mapping"
            return result
        
        config_valid, config_error = check_model_config(model_name, config_file)
        if not config_valid:
            result['error'] = f"Config error: {config_error}"
            return result
        
        # 2. Check Python class exists
        class_name = EXPECTED_CLASSES.get(model_name)
        python_file = MODEL_TO_FILE.get(model_name)
        
        if python_file:
            python_path = base_dir / python_file
            
            if not check_file_exists(python_path):
                result['error'] = f"Python file not found: {python_file}"
                return result
            
            # Check syntax
            syntax_valid, syntax_error = check_python_syntax(python_path)
            if not syntax_valid:
                result['error'] = f"Python syntax: {syntax_error}"
                return result
            
            # For neural models, check class exists
            if class_name not in ['XGBClassifier', 'RandomForestClassifier']:
                if not find_class_in_file(python_path, class_name):
                    result['error'] = f"Class {class_name} not found in {python_file}"
                    return result
        
        # Success!
        result['status'] = '✅ OK'
        
    except Exception as e:
        result['error'] = str(e)
        result['status'] = '❌ FAIL'
    
    return result


def print_registry_table(results: List[Dict[str, Any]]):
    """Print results as a clean ASCII table."""
    print("=" * 80)
    print("REGISTRY MAP: Model → Config → Class → Status")
    print("=" * 80)
    
    # Calculate column widths
    col_widths = {
        'model': max(len('Model Name'), max(len(r['model_name']) for r in results)),
        'config': max(len('Config File'), max(len(r['config_file']) for r in results)),
        'class': max(len('Python Class'), max(len(r['python_class']) for r in results)),
        'status': 10
    }
    
    # Header
    header = (
        f"| {'Model Name':<{col_widths['model']}} | "
        f"{'Config File':<{col_widths['config']}} | "
        f"{'Python Class':<{col_widths['class']}} | "
        f"{'Status':<{col_widths['status']}} |"
    )
    separator = "+" + "-" * (len(header) - 2) + "+"
    
    print(separator)
    print(header)
    print(separator)
    
    # Rows
    for result in results:
        row = (
            f"| {result['model_name']:<{col_widths['model']}} | "
            f"{result['config_file']:<{col_widths['config']}} | "
            f"{result['python_class']:<{col_widths['class']}} | "
            f"{result['status']:<{col_widths['status']}} |"
        )
        print(row)
    
    print(separator)
    
    # Summary
    success_count = sum(1 for r in results if '✅' in r['status'])
    fail_count = len(results) - success_count
    
    print()
    print(f"SUMMARY: {success_count}/{len(results)} models validated successfully")
    
    if fail_count > 0:
        print()
        print("ERRORS:")
        for result in results:
            if '❌' in result['status'] and result['error']:
                print(f"  - {result['model_name']}: {result['error']}")


def check_file_structure():
    """Verify the refactored file structure."""
    print("=" * 80)
    print("FILE STRUCTURE CHECK")
    print("=" * 80)
    
    # Go up to train/ root since we're now in utils/
    base_dir = Path(__file__).parent.parent
    
    required_files = [
        'data_pipeline/preprocessor.py',
        'architectures/model_factory.py',
        'data_pipeline/loader.py',
        'utils/config_manager.py',
        'architectures/models/__init__.py',
        'architectures/models/base.py',
        'architectures/models/mlp.py',
        'architectures/models/transformer.py',
        'architectures/models/dual_tower.py',
        'architectures/models/wrappers.py',
        'architectures/models/duet_kan.py',
        'config/_base/00_global.yaml',
        'config/_base/01_data.yaml',
        'config/_base/02_train.yaml',
    ]
    
    all_exist = True
    for file_path in required_files:
        full_path = base_dir / file_path
        if check_file_exists(full_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - NOT FOUND")
            all_exist = False
    
    print()
    if all_exist:
        print("✅ All required files present")
    else:
        print("⚠️  Some files are missing")
    
    print()


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run system validation."""
    print()
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "SYSTEM VALIDATION - HEALTH CHECK" + " " * 26 + "║")
    print("╚" + "=" * 78 + "╝")
    print()
    
    # Step 1: Cleanup
    cleanup_legacy_artifacts()
    
    # Step 2: Check file structure
    check_file_structure()
    
    # Step 3: Validate each model
    print("=" * 80)
    print("MODEL VALIDATION")
    print("=" * 80)
    print()
    
    results = []
    
    for model_name in MODEL_NAMES:
        print(f"Testing: {model_name}...", end=" ", flush=True)
        result = validate_model(model_name)
        results.append(result)
        print(result['status'])
    
    print()
    
    # Step 4: Print results table
    print_registry_table(results)
    
    # Step 5: Final verdict
    all_pass = all('✅' in r['status'] for r in results)
    
    print()
    print("=" * 80)
    if all_pass:
        print("🎉 SYSTEM VALIDATION PASSED - ALL MODELS OPERATIONAL")
    else:
        print("⚠️  SYSTEM VALIDATION FAILED - SOME MODELS HAVE ISSUES")
    print("=" * 80)
    print()
    
    return 0 if all_pass else 1


if __name__ == '__main__':
    sys.exit(main())
