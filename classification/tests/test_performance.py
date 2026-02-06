#!/usr/bin/env python3
# ==============================================================================
# Copyright (c) 2026 Himanshu Kumar, IIT Bhubaneswar. All Rights Reserved.
# Email: hkphimanshukumar321@gmail.com
# ==============================================================================

"""
Performance Tests
=================

GPU, multiprocessing, and system performance tests:
- GPU detection
- Memory usage
- Inference benchmarking
- Multiprocessing capability
"""

import sys
import time
import multiprocessing
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
        print(f"  ✅ {name}")
    
    def add_fail(self, name: str, error: str):
        self.failed += 1
        self.errors.append((name, error))
        print(f"  ❌ {name}: {error}")


def test_gpu_detection(results: TestResult):
    """Test GPU detection and configuration."""
    print("\n🖥️ Testing GPU Detection...")
    
    try:
        import tensorflow as tf
        
        gpus = tf.config.list_physical_devices('GPU')
        
        if len(gpus) > 0:
            results.add_pass(f"GPU detected: {len(gpus)} device(s)")
            for i, gpu in enumerate(gpus):
                print(f"     GPU {i}: {gpu.name}")
        else:
            results.add_pass("No GPU detected (CPU mode)")
        
        # Test device info
        from src.training import get_device_info
        info = get_device_info()
        results.add_pass(f"TensorFlow version: {info['tensorflow_version']}")
        
    except Exception as e:
        results.add_fail("GPU detection", str(e))


def test_memory_usage(results: TestResult):
    """Test memory usage estimation."""
    print("\n💾 Testing Memory Usage...")
    
    try:
        import psutil
        
        process = psutil.Process()
        initial_mem = process.memory_info().rss / (1024 * 1024)  # MB
        
        # Create model and check memory increase
        from src.models import create_model, get_model_metrics
        
        model = create_model((64, 64, 3), num_classes=10)
        metrics = get_model_metrics(model)
        
        after_mem = process.memory_info().rss / (1024 * 1024)
        mem_increase = after_mem - initial_mem
        
        results.add_pass(f"Model memory: {metrics.memory_mb:.1f} MB (estimated)")
        results.add_pass(f"Actual memory increase: {mem_increase:.1f} MB")
        
    except ImportError:
        print("  ⚠️ psutil not installed - skipping memory test")
    except Exception as e:
        results.add_fail("Memory usage", str(e))


def test_inference_benchmark(results: TestResult):
    """Test inference benchmarking."""
    print("\n⏱️ Testing Inference Benchmark...")
    
    try:
        from src.models import create_model
        from src.training import benchmark_inference
        
        model = create_model((64, 64, 3), num_classes=10, depth=(2, 2, 2))
        
        latency = benchmark_inference(
            model, (64, 64, 3),
            warmup_runs=3,
            benchmark_runs=10,
            batch_sizes=[1, 8]
        )
        
        results.add_pass(f"Batch 1 latency: {latency['batch_1']['mean_ms']:.2f}ms")
        results.add_pass(f"Batch 8 latency: {latency['batch_8']['mean_ms']:.2f}ms")
        results.add_pass(f"Throughput: {latency['batch_1']['throughput_fps']:.1f} FPS")
        
    except Exception as e:
        results.add_fail("Inference benchmark", str(e))


def _worker_task(x):
    """Simple worker task for multiprocessing test."""
    return x * x


def test_multiprocessing(results: TestResult):
    """Test multiprocessing capability."""
    print("\n🔀 Testing Multiprocessing...")
    
    try:
        n_cores = multiprocessing.cpu_count()
        results.add_pass(f"CPU cores available: {n_cores}")
        
        # Test pool
        with multiprocessing.Pool(processes=min(4, n_cores)) as pool:
            data = list(range(100))
            result = pool.map(_worker_task, data)
            
            assert len(result) == 100
            assert result[10] == 100  # 10^2
            
        results.add_pass("Multiprocessing pool works")
        
    except Exception as e:
        results.add_fail("Multiprocessing", str(e))


def test_data_loading_performance(results: TestResult):
    """Test data loading performance."""
    print("\n📦 Testing Data Loading Performance...")
    
    try:
        import tensorflow as tf
        from src.data_loader import create_tf_dataset
        
        # Synthetic data
        X = np.random.randn(1000, 64, 64, 3).astype(np.float32)
        Y = np.random.randint(0, 10, 1000)
        
        # Create dataset
        start = time.perf_counter()
        ds = create_tf_dataset(X, Y, batch_size=32, shuffle=True, prefetch=True)
        creation_time = time.perf_counter() - start
        
        # Iterate through dataset
        start = time.perf_counter()
        for batch in ds:
            pass
        iteration_time = time.perf_counter() - start
        
        results.add_pass(f"Dataset creation: {creation_time*1000:.1f}ms")
        results.add_pass(f"Full iteration (1000 samples): {iteration_time*1000:.1f}ms")
        
        throughput = 1000 / iteration_time
        results.add_pass(f"Data throughput: {throughput:.0f} samples/sec")
        
    except Exception as e:
        results.add_fail("Data loading performance", str(e))


def run_performance_tests() -> TestResult:
    """Run all performance tests."""
    print("=" * 60)
    print("PERFORMANCE TESTS")
    print("=" * 60)
    
    results = TestResult()
    
    test_gpu_detection(results)
    test_memory_usage(results)
    test_inference_benchmark(results)
    test_multiprocessing(results)
    test_data_loading_performance(results)
    
    print("\n" + "=" * 60)
    print(f"RESULTS: {results.passed} passed, {results.failed} failed")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    results = run_performance_tests()
    sys.exit(0 if results.failed == 0 else 1)