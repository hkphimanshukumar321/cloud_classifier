#!/usr/bin/env python3
# ==============================================================================
# Copyright (c) 2026 Himanshu Kumar, IIT Bhubaneswar. All Rights Reserved.
# Email: hkphimanshukumar321@gmail.com
# ==============================================================================

"""
Integration Tests
=================

End-to-end tests:
- Mini training run
- Result file generation
- Full pipeline execution
"""

import sys
import shutil
import tempfile
import numpy as np
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
        print(f"  ✅ {name}")
    
    def add_fail(self, name: str, error: str):
        self.failed += 1
        self.errors.append((name, error))
        print(f"  ❌ {name}: {error}")


def test_mini_training(results: TestResult):
    """Test a minimal training run."""
    print("\n🏋️ Testing Mini Training...")
    
    try:
        import tensorflow as tf
        from src.models import create_model
        from src.training import compile_model, train_model
        
        tf.keras.backend.clear_session()
        
        # Synthetic data
        X = np.random.randn(50, 32, 32, 3).astype(np.float32)
        Y = np.random.randint(0, 3, 50)
        
        X_train, X_val = X[:40], X[40:]
        y_train, y_val = Y[:40], Y[40:]
        
        model = create_model((32, 32, 3), num_classes=3, depth=(1, 1, 1))
        model = compile_model(model, learning_rate=0.01)
        
        # Train for 2 epochs
        with tempfile.TemporaryDirectory() as tmp_dir:
            history = train_model(
                model, X_train, y_train, X_val, y_val,
                run_dir=Path(tmp_dir),
                epochs=2,
                batch_size=10,
                verbose=0
            )
            
            assert 'loss' in history
            assert 'accuracy' in history
            assert len(history['loss']) == 2
            
            # Check files created
            assert (Path(tmp_dir) / 'training_log.csv').exists()
            assert (Path(tmp_dir) / 'history.json').exists()
        
        results.add_pass("Training completes successfully")
        results.add_pass("History contains expected keys")
        results.add_pass("Output files created")
        
    except Exception as e:
        results.add_fail("Mini training", str(e))


def test_logger_integration(results: TestResult):
    """Test experiment logger."""
    print("\n📝 Testing Logger Integration...")
    
    try:
        from common.logger import ExperimentLogger
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            logger = ExperimentLogger(tmp_dir)
            
            logger.log_machine_info()
            assert (Path(tmp_dir) / 'machine_info.json').exists()
            results.add_pass("Machine info logged")
            
            logger.log_experiment("exp_001", {"accuracy": 95.0, "loss": 0.15})
            logger.log_experiment("exp_002", {"accuracy": 96.5, "loss": 0.12})
            
            logger.save_all()
            
            assert (Path(tmp_dir) / 'experiments.csv').exists()
            assert (Path(tmp_dir) / 'experiments.json').exists()
            assert (Path(tmp_dir) / 'summary.json').exists()
            
            results.add_pass("Experiments saved to CSV")
            results.add_pass("Experiments saved to JSON")
            
    except Exception as e:
        results.add_fail("Logger integration", str(e))


def test_visualization_integration(results: TestResult):
    """Test visualization pipeline."""
    print("\n📊 Testing Visualization Integration...")
    
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        
        from src.visualization import (
            plot_training_history, plot_confusion_matrix,
            plot_model_comparison_bar, close_all_figures
        )
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            
            # Training history
            history = {
                'loss': [0.5, 0.3, 0.2],
                'val_loss': [0.6, 0.4, 0.3],
                'accuracy': [0.7, 0.8, 0.9],
                'val_accuracy': [0.65, 0.75, 0.85]
            }
            plot_training_history(history, save_path=tmp_path / 'history.png')
            assert (tmp_path / 'history.png').exists()
            results.add_pass("Training history plot saved")
            
            # Confusion matrix
            y_true = np.array([0, 0, 1, 1, 2, 2])
            y_pred = np.array([0, 0, 1, 2, 2, 2])
            plot_confusion_matrix(y_true, y_pred, save_path=tmp_path / 'cm.png')
            assert (tmp_path / 'cm.png').exists()
            results.add_pass("Confusion matrix plot saved")
            
            close_all_figures()
            
    except Exception as e:
        results.add_fail("Visualization integration", str(e))


def run_integration_tests() -> TestResult:
    """Run all integration tests."""
    print("=" * 60)
    print("INTEGRATION TESTS")
    print("=" * 60)
    
    results = TestResult()
    
    test_mini_training(results)
    test_logger_integration(results)
    test_visualization_integration(results)
    
    print("\n" + "=" * 60)
    print(f"RESULTS: {results.passed} passed, {results.failed} failed")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    results = run_integration_tests()
    sys.exit(0 if results.failed == 0 else 1)