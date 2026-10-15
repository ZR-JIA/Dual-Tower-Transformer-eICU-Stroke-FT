#!/usr/bin/env python3
"""
CLI Smoke Test Suite

Verifies runtime integrity of entry point scripts after major refactoring.
Tests actual subprocess execution to catch path resolution errors, import issues,
and config loading problems.

Critical Tests:
1. Neural model training (MLP) - Quick 1 epoch
2. Tree model training (XGBoost) - Fast sklearn model
3. Optuna from subdirectory - Path resolution verification

Usage:
    python tests/smoke_test_cli.py
"""

import subprocess
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple


# Test suite configuration
TEST_TIMEOUT = 120  # seconds (generous for first-time imports)
TRAIN_ROOT = Path(__file__).parent.parent


class SmokeTest:
    """Individual smoke test case."""
    
    def __init__(
        self,
        name: str,
        command: List[str],
        cwd: Path = None,
        timeout: int = TEST_TIMEOUT,
        expect_failure: bool = False
    ):
        self.name = name
        self.command = command
        self.cwd = cwd or TRAIN_ROOT
        self.timeout = timeout
        self.expect_failure = expect_failure
        self.result = None
        self.stdout = None
        self.stderr = None
        self.duration = None
        self.exit_code = None
    
    def run(self) -> bool:
        """Execute the test and return success status."""
        print(f"\n{'='*80}")
        print(f"TEST: {self.name}")
        print(f"{'='*80}")
        print(f"Command: {' '.join(self.command)}")
        print(f"Working Dir: {self.cwd}")
        print(f"Timeout: {self.timeout}s")
        print(f"-" * 80)
        
        start_time = datetime.now()
        
        try:
            # Set up environment
            env = os.environ.copy()
            # Ensure Python can find our modules
            if 'PYTHONPATH' in env:
                env['PYTHONPATH'] = f"{self.cwd}:{env['PYTHONPATH']}"
            else:
                env['PYTHONPATH'] = str(self.cwd)
            
            # Run the command
            result = subprocess.run(
                self.command,
                cwd=self.cwd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                env=env
            )
            
            self.result = result
            self.stdout = result.stdout
            self.stderr = result.stderr
            self.exit_code = result.returncode
            self.duration = (datetime.now() - start_time).total_seconds()
            
            # Determine success
            if self.expect_failure:
                success = result.returncode != 0
            else:
                success = result.returncode == 0
            
            # Report result
            if success:
                print(f"✅ PASS ({self.duration:.1f}s)")
                if self.stdout:
                    print(f"\nLast 5 lines of output:")
                    print('\n'.join(self.stdout.strip().split('\n')[-5:]))
            else:
                print(f"❌ FAIL ({self.duration:.1f}s)")
                print(f"Exit code: {self.exit_code}")
                
                if self.stderr:
                    print(f"\nLast 15 lines of stderr:")
                    stderr_lines = self.stderr.strip().split('\n')
                    print('\n'.join(stderr_lines[-15:]))
                
                if self.stdout:
                    print(f"\nLast 10 lines of stdout:")
                    stdout_lines = self.stdout.strip().split('\n')
                    print('\n'.join(stdout_lines[-10:]))
            
            return success
            
        except subprocess.TimeoutExpired:
            self.duration = self.timeout
            print(f"❌ TIMEOUT after {self.timeout}s")
            return False
        
        except FileNotFoundError as e:
            print(f"❌ FILE NOT FOUND: {e}")
            return False
        
        except Exception as e:
            print(f"❌ ERROR: {e}")
            import traceback
            print(traceback.format_exc())
            return False


def create_test_suite() -> List[SmokeTest]:
    """Create the smoke test suite."""
    
    tests = []
    
    # ========================================================================
    # TEST 1: Neural Model Training (MLP - Quick)
    # ========================================================================
    tests.append(SmokeTest(
        name="Neural Model Training (MLP, 1 epoch)",
        command=[
            sys.executable,
            "run_unified_train.py",
            "--model", "mlp",
            # Note: We'll just let it fail gracefully if data is missing
            # The point is to test config loading and imports
        ],
        timeout=60
    ))
    
    # ========================================================================
    # TEST 2: Tree Model Training (XGBoost)
    # ========================================================================
    tests.append(SmokeTest(
        name="Tree Model Training (XGBoost)",
        command=[
            sys.executable,
            "run_unified_train.py",
            "--model", "xgboost",
        ],
        timeout=60
    ))
    
    # ========================================================================
    # TEST 3: Optuna from Subdirectory (CRITICAL)
    # ========================================================================
    # This is the most likely to fail due to path resolution issues
    tests.append(SmokeTest(
        name="Optuna from Subdirectory (Path Resolution Check)",
        command=[
            sys.executable,
            "run_optuna.py",
            "--model", "nn",
            "--n_trials", "1",
        ],
        cwd=TRAIN_ROOT / "optuna",
        timeout=60
    ))
    
    # ========================================================================
    # TEST 4: Config Loading Verification
    # ========================================================================
    tests.append(SmokeTest(
        name="Config Manager - Load All Models",
        command=[
            sys.executable,
            "-c",
            """
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
from utils.config_manager import ConfigManager

cm = ConfigManager()
models = ['nn', 'mlp', 'transformer', 'xgboost', 'random_forest', 'dualtower_mlp', 'dualtower']
for model in models:
    config = cm.load_config(model)
    print(f'✓ {model}: config loaded, model_type={config["model_config"]["model"]}')
print(f'✅ All {len(models)} configs loaded successfully')
"""
        ],
        timeout=10
    ))
    
    # ========================================================================
    # TEST 5: Import Verification
    # ========================================================================
    tests.append(SmokeTest(
        name="Import Verification - All Packages",
        command=[
            sys.executable,
            "-c",
            """
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

# Test all critical imports
from architectures.models import MLPModel, SimpleNN, TransformerModel, DualTower
from architectures.model_factory import ModelFactory, get_model_factory
from data_pipeline.preprocessor import TabularPreprocessor
from data_pipeline.loader import DataModule, build_datamodule, build_model
from engine.trainers import NeuralTrainer, TreeTrainer
from engine.evaluator import UnifiedEvaluator
from utils.config_manager import ConfigManager
from utils.seed import set_all_seeds

print('✅ All imports successful')
"""
        ],
        timeout=10
    ))
    
    # ========================================================================
    # TEST 6: System Health Check
    # ========================================================================
    tests.append(SmokeTest(
        name="System Health Check (utils/system_check.py)",
        command=[
            sys.executable,
            "utils/system_check.py"
        ],
        timeout=15
    ))
    
    return tests


def run_test_suite(tests: List[SmokeTest]) -> Dict[str, Any]:
    """Run all tests and collect results."""
    
    results = {
        'total': len(tests),
        'passed': 0,
        'failed': 0,
        'tests': []
    }
    
    for i, test in enumerate(tests, 1):
        print(f"\n{'#'*80}")
        print(f"# Test {i}/{len(tests)}")
        print(f"{'#'*80}")
        
        success = test.run()
        
        results['tests'].append({
            'name': test.name,
            'success': success,
            'duration': test.duration,
            'exit_code': test.exit_code
        })
        
        if success:
            results['passed'] += 1
        else:
            results['failed'] += 1
    
    return results


def print_summary(results: Dict[str, Any]):
    """Print test suite summary."""
    
    print("\n" + "="*80)
    print("SMOKE TEST SUMMARY")
    print("="*80)
    
    for i, test in enumerate(results['tests'], 1):
        status = "✅ PASS" if test['success'] else "❌ FAIL"
        duration = f"{test['duration']:.1f}s" if test['duration'] else "N/A"
        exit_code = f"(exit: {test['exit_code']})" if test['exit_code'] is not None else ""
        print(f"{i}. {status} - {test['name']} ({duration}) {exit_code}")
    
    print()
    print(f"Total:  {results['total']}")
    print(f"Passed: {results['passed']} ✅")
    print(f"Failed: {results['failed']} ❌")
    
    pass_rate = (results['passed'] / results['total']) * 100 if results['total'] > 0 else 0
    print(f"Pass Rate: {pass_rate:.1f}%")
    
    print("="*80)
    
    if results['failed'] == 0:
        print("\n🎉 ALL TESTS PASSED - SYSTEM OPERATIONAL")
    else:
        print(f"\n⚠️  {results['failed']} TEST(S) FAILED - REVIEW ERRORS ABOVE")
    
    print()


def main():
    """Main execution."""
    
    print()
    print("╔" + "="*78 + "╗")
    print("║" + " "*25 + "CLI SMOKE TEST SUITE" + " "*33 + "║")
    print("╚" + "="*78 + "╝")
    print()
    print("Purpose: Verify runtime integrity of entry points after refactoring")
    print("Focus:   Path resolution, imports, config loading, subprocess execution")
    print()
    print(f"Working Directory: {TRAIN_ROOT}")
    print(f"Python: {sys.executable}")
    print(f"Timeout: {TEST_TIMEOUT}s per test")
    print()
    
    # Check if we're in the right directory
    if not (TRAIN_ROOT / "run_unified_train.py").exists():
        print("❌ ERROR: run_unified_train.py not found!")
        print(f"   Expected location: {TRAIN_ROOT / 'run_unified_train.py'}")
        print("   Are you running from the correct directory?")
        return 1
    
    # Create test suite
    tests = create_test_suite()
    print(f"Created {len(tests)} smoke tests")
    
    # Run tests
    print("\nStarting test execution...\n")
    results = run_test_suite(tests)
    
    # Print summary
    print_summary(results)
    
    # Return exit code
    return 0 if results['failed'] == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
