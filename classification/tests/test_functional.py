#!/usr/bin/env python3
"""
Functional Tests
================

Test individual components work correctly:
- Model creation and forward pass
- Data loading
- Training step
"""

import sys
import numpy as np
from pathlib import Path

TEMPLATE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(TEMPLATE_DIR))


class TestResult:
    """Test result container."""
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []
    
    def add_pass(self, name: str):
        self.passed += 1
        print(f"  âœ[*] {name}")
    
    def add_fail(self, name: str, error: str):
        self.failed += 1
        self.errors.append((name, error))
        print(f"  âŒ {name}: {error}")


def test_model_creation(results: TestResult):
    """Test model creation."""
    print("\nðŸ[*]ï¸ Testing Model Creation...")
    
    try:
        from src.models import create_model, get_model_metrics
        
        model = create_model((64, 64, 3), num_classes=10)
        results.add_pass("create_model() works")
        
        metrics = get_model_metrics(model)
        results.add_pass(f"Model has {metrics.total_params:,} parameters")
        
    except Exception as e:
        results.add_fail("Model creation", str(e))


def test_model_forward_pass(results: TestResult):
    """Test model forward pass."""
    print("\nðŸ[*][*] Testing Forward Pass...")
    
    try:
        from src.models import create_model
        
        model = create_model((64, 64, 3), num_classes=10)
        
        # Random input
        dummy_input = np.random.randn(2, 64, 64, 3).astype(np.float32)
        output = model.predict(dummy_input, verbose=0)
        
        assert output.shape == (2, 10), f"Expected (2, 10), got {output.shape}"
        assert np.allclose(output.sum(axis=1), 1.0, atol=1e-5), "Outputs should sum to 1"
        
        results.add_pass("Forward pass shape correct")
        results.add_pass("Softmax outputs valid")
        
    except Exception as e:
        results.add_fail("Forward pass", str(e))


def test_data_loader(results: TestResult):
    """Test data loader functions."""
    print("\nðŸ[*]¦ Testing Data Loader...")
    
    try:
        from src.data_loader import split_dataset, create_tf_dataset
        
        # Synthetic data
        X = np.random.randn(100, 64, 64, 3).astype(np.float32)
        Y = np.random.randint(0, 5, 100)
        
        splits = split_dataset(X, Y, test_size=0.2, val_size=0.2)
        
        assert 'train' in splits
        assert 'val' in splits
        assert 'test' in splits
        results.add_pass("split_dataset() works")
        
        ds = create_tf_dataset(splits['train'][0], splits['train'][1], batch_size=8)
        batch = next(iter(ds))
        assert batch[0].shape[0] == 8
        results.add_pass("create_tf_dataset() works")
        
    except Exception as e:
        results.add_fail("Data loader", str(e))


def test_training_utils(results: TestResult):
    """Test training utilities."""
    print("\nðŸŽ¯ Testing Training Utils...")
    
    try:
        from src.training import compile_model, get_device_info
        from src.models import create_model
        
        model = create_model((32, 32, 3), num_classes=5)
        model = compile_model(model, learning_rate=0.001)
        
        assert model.optimizer is not None
        results.add_pass("compile_model() works")
        
        info = get_device_info()
        assert 'hostname' in info
        assert 'tensorflow_version' in info
        results.add_pass("get_device_info() works")
        
    except Exception as e:
        results.add_fail("Training utils", str(e))


def test_visualization(results: TestResult):
    """Test visualization functions."""
    print("\nðŸ[*]Š Testing Visualization...")
    
    try:
        from src.visualization import set_publication_style, close_all_figures
        import matplotlib.pyplot as plt
        
        set_publication_style()
        results.add_pass("set_publication_style() works")
        
        # Create simple figure
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 9])
        
        close_all_figures()
        results.add_pass("close_all_figures() works")
        
    except Exception as e:
        results.add_fail("Visualization", str(e))


def run_functional_tests() -> TestResult:
    """Run all functional tests."""
    print("=" * 60)
    print("FUNCTIONAL TESTS")
    print("=" * 60)
    
    results = TestResult()
    
    test_model_creation(results)
    test_model_forward_pass(results)
    test_data_loader(results)
    test_training_utils(results)
    test_visualization(results)
    
    print("\n" + "=" * 60)
    print(f"RESULTS: {results.passed} passed, {results.failed} failed")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    results = run_functional_tests()
    sys.exit(0 if results.failed == 0 else 1)
