#!/usr/bin/env python3
# ==============================================================================
# Copyright (c) 2026 Himanshu Kumar, IIT Bhubaneswar. All Rights Reserved.
# Email: hkphimanshukumar321@gmail.com
# ==============================================================================

"""
Smoke Tests
===========

Quick validation tests for system health:
- Module imports
- Configuration loading
- Directory structure
- Dependency availability
"""

import sys
import importlib
from pathlib import Path

TEMPLATE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(TEMPLATE_DIR))
sys.path.append(str(TEMPLATE_DIR.parent))  # Add template/ for common


class TestResult:
    """Test result container."""
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []
    
    def add_pass(self, name: str):
        self.passed += 1
        print(f"  [PASS] {name}")
    
    def add_fail(self, name: str, error: str):
        self.failed += 1
        self.errors.append((name, error))
        print(f"  [FAIL] {name}: {error}")


def test_python_imports(results: TestResult):
    """Test module imports."""
    print("\n[*] Testing Imports...")
    
    modules = [
        ('config', 'Configuration'),
        ('src.models', 'Models'),
        ('src.training', 'Training'),
        ('src.data_loader', 'Data Loader'),
        ('src.visualization', 'Visualization'),
        ('common.logger', 'Logger'),
    ]
    
    for module_name, description in modules:
        try:
            importlib.import_module(module_name)
            results.add_pass(f"{description} ({module_name})")
        except ImportError as e:
            results.add_fail(f"{description} ({module_name})", str(e))


def test_config_loading(results: TestResult):
    """Test configuration loading."""
    print("\n[*] Testing Configuration...")
    
    try:
        from config import ResearchConfig
        results.add_pass("ResearchConfig imports")
    except ImportError as e:
        results.add_fail("ResearchConfig imports", str(e))
        return
    
    try:
        config = ResearchConfig()
        results.add_pass("ResearchConfig instantiates")
    except Exception as e:
        results.add_fail("ResearchConfig instantiates", str(e))
        return
    
    checks = [
        ('config.data.img_size', lambda: config.data.img_size),
        ('config.model.growth_rate', lambda: config.model.growth_rate),
        ('config.training.epochs', lambda: config.training.epochs),
    ]
    
    for name, getter in checks:
        try:
            value = getter()
            results.add_pass(f"{name} = {value}")
        except Exception as e:
            results.add_fail(name, str(e))


def test_directory_structure(results: TestResult):
    """Test directory structure."""
    print("\n[*] Testing Directories...")
    
    required = [
        TEMPLATE_DIR / 'src',
        TEMPLATE_DIR / 'experiments',
        TEMPLATE_DIR / 'tests',
    ]
    
    optional = [
        TEMPLATE_DIR / 'results',
        TEMPLATE_DIR / 'data',
    ]
    
    for path in required:
        if path.exists():
            results.add_pass(f"Required: {path.name}")
        else:
            results.add_fail(f"Required: {path.name}", "Not found")
    
    for path in optional:
        if path.exists():
            results.add_pass(f"Optional: {path.name}")
        else:
            print(f"  [WARN] Optional: {path.name} (will be created)")


def test_experiment_files(results: TestResult):
    """Test experiment files."""
    print("\n[*] Testing Experiment Files...")
    
    files = [
        TEMPLATE_DIR / 'experiments' / 'run_ablation.py',
        TEMPLATE_DIR / 'experiments' / 'run_cross_validation.py',
        TEMPLATE_DIR / 'experiments' / 'run_snr_robustness.py',
        TEMPLATE_DIR / 'experiments' / 'run_baselines.py',
        TEMPLATE_DIR / 'run.py',
    ]
    
    for path in files:
        if not path.exists():
            results.add_fail(path.name, "Not found")
            continue
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                compile(f.read(), path, 'exec')
            results.add_pass(f"{path.name} (valid syntax)")
        except SyntaxError as e:
            results.add_fail(path.name, f"Syntax error: {e}")


def test_dependencies(results: TestResult):
    """Test dependencies."""
    print("\n[*] Testing Dependencies...")
    
    deps = [
        ('numpy', 'NumPy'),
        ('pandas', 'Pandas'),
        ('tensorflow', 'TensorFlow'),
        ('sklearn', 'Scikit-learn'),
        ('matplotlib', 'Matplotlib'),
        ('seaborn', 'Seaborn'),
        ('cv2', 'OpenCV'),
        ('tqdm', 'tqdm'),
    ]
    
    for module_name, display in deps:
        try:
            mod = importlib.import_module(module_name)
            version = getattr(mod, '__version__', 'unknown')
            results.add_pass(f"{display} v{version}")
        except ImportError:
            results.add_fail(display, "Not installed")


def run_smoke_tests() -> TestResult:
    """Run all smoke tests."""
    print("=" * 60)
    print("SMOKE TESTS")
    print("=" * 60)
    
    results = TestResult()
    
    test_python_imports(results)
    test_config_loading(results)
    test_directory_structure(results)
    test_experiment_files(results)
    test_dependencies(results)
    
    print("\n" + "=" * 60)
    print(f"RESULTS: {results.passed} passed, {results.failed} failed")
    print("=" * 60)
    
    if results.errors:
        print("\n[!] Errors:")
        for name, error in results.errors:
            print(f"  - {name}: {error}")
    
    return results


if __name__ == "__main__":
    results = run_smoke_tests()
    
    # Generate Audit Report
    try:
        from common.audit import generate_audit_report
        from common.hardware import get_system_info
        
        sys_info = get_system_info()
        audit_data = {
            'total': results.passed + results.failed,
            'passed': results.passed,
            'failed': results.failed,
            'details': [f"{name} | {err}" for name, err in results.errors],
            'system': sys_info.get('system', 'Unknown'),
            'processor': sys_info.get('processor', 'Unknown'),
            'duration': 0.0
        }
        
        # Save to template root
        root_dir = Path(__file__).parent.parent.parent
        generate_audit_report(audit_data, output_dir=root_dir)
        
    except Exception as e:
        print(f"\\n[!] Audit generation failed: {e}")
        import traceback
        traceback.print_exc()

    sys.exit(0 if results.failed == 0 else 1)