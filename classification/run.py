#!/usr/bin/env python3
# ==============================================================================
# Copyright (c) 2026 Himanshu Kumar, IIT Bhubaneswar. All Rights Reserved.
# Email: hkphimanshukumar321@gmail.com
# ==============================================================================

"""
Master Experiment Runner - Research Template
=============================================

Runs all research experiments for publication-ready results.

Experiments:
1. Full Factorial Ablation Study
2. Cross-Validation
3. SNR Robustness Testing
4. Baseline Model Comparisons

Usage:
    python run_all_experiments.py           # Full run
    python run_all_experiments.py --quick   # Quick test (2 epochs)
    python run_all_experiments.py --skip-tests  # Skip test suite
"""

import sys
import argparse
import time
from pathlib import Path
from datetime import datetime

# Add template root to path
# Add paths for imports
# 1. classification/ directory
CLASSIFICATION_DIR = Path(__file__).parent
sys.path.insert(0, str(CLASSIFICATION_DIR))
# 2. template/ directory (for common)
sys.path.append(str(CLASSIFICATION_DIR.parent))


def run_all(quick_test: bool = False, skip_tests: bool = False):
    """
    Run all experiments.
    
    Args:
        quick_test: Use 2 epochs per experiment
        skip_tests: Skip test suite
    """
    start_time = time.time()
    
    print("=" * 70)
    print("  RESEARCH TEMPLATE - MASTER EXPERIMENT RUNNER")
    print("=" * 70)
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Quick test: {'Yes' if quick_test else 'No'}")
    print("=" * 70)
    
    from config import ResearchConfig, print_experiment_summary
    config = ResearchConfig()
    
    print("\n[*] Configuration Summary:")
    print_experiment_summary(config)
    
    # Check data directory
    if not config.data.data_dir.exists():
        print("\n" + "[!] " * 20)
        print("DATA DIRECTORY NOT FOUND!")
        print(f"Configure 'data_dir' in config.py: {config.data.data_dir}")
        print("[!] " * 20)
        return False
    
    results = {}
    
    # 1. Run tests first
    if not skip_tests:
        print("\n\n" + "=" * 70)
        print("PHASE 1: RUNNING TEST SUITE")
        print("=" * 70)
        
        from tests.test_smoke import run_smoke_tests
        smoke_results = run_smoke_tests()
        
        if smoke_results.failed > 0:
            print("\n[!] Smoke tests failed. Fix issues before running experiments.")
            return False
        
        results['tests'] = 'passed'
    
    # 2. Ablation study
    print("\n\n" + "=" * 70)
    print("PHASE 2: ABLATION STUDY")
    print("=" * 70)
    
    try:
        from experiments.run_ablation import run_ablation
        ablation_df = run_ablation(quick_test=quick_test, single_seed=quick_test)
        
        if ablation_df is not None:
            results['ablation'] = {
                'experiments': len(ablation_df),
                'best_accuracy': ablation_df['test_accuracy'].max()
            }
    except Exception as e:
        print(f"[FAIL] Ablation failed: {e}")
        results['ablation'] = 'failed'
    
    # 3. Cross-validation
    if config.training.enable_cross_validation:
        print("\n\n" + "=" * 70)
        print("PHASE 3: CROSS-VALIDATION")
        print("=" * 70)
        
        try:
            from experiments.run_cross_validation import run_cross_validation
            cv_df = run_cross_validation(quick_test=quick_test)
            
            if cv_df is not None:
                results['cross_validation'] = {
                    'folds': len(cv_df),
                    'mean_accuracy': cv_df['val_accuracy'].mean()
                }
        except Exception as e:
            print(f"[FAIL] Cross-validation failed: {e}")
            results['cross_validation'] = 'failed'
    
    # 4. SNR robustness
    if config.training.enable_snr_testing:
        print("\n\n" + "=" * 70)
        print("PHASE 4: SNR ROBUSTNESS")
        print("=" * 70)
        
        try:
            from experiments.run_snr_robustness import run_snr_robustness
            snr_df = run_snr_robustness(quick_test=quick_test)
            
            if snr_df is not None:
                results['snr_robustness'] = {
                    'levels_tested': len(snr_df),
                    'min_snr_accuracy': snr_df['accuracy'].min()
                }
        except Exception as e:
            print(f"[FAIL] SNR testing failed: {e}")
            results['snr_robustness'] = 'failed'
    
    # 5. Baseline comparison (optional)
    print("\n\n" + "=" * 70)
    print("PHASE 5: BASELINE COMPARISON")
    print("=" * 70)
    
    try:
        from experiments.run_baselines import run_baselines
        baseline_df = run_baselines(
            models=['MobileNetV2', 'EfficientNetV2B0'],
            quick_test=quick_test
        )
        
        if baseline_df is not None:
            results['baselines'] = {
                'models_tested': len(baseline_df),
                'best_accuracy': baseline_df['accuracy'].max()
            }
    except Exception as e:
        print(f"[FAIL] Baseline comparison failed: {e}")
        results['baselines'] = 'failed'
    
    # Summary
    elapsed = time.time() - start_time
    elapsed_str = f"{elapsed/3600:.1f}h" if elapsed > 3600 else f"{elapsed/60:.1f}m"
    
    print("\n\n" + "=" * 70)
    print("ALL EXPERIMENTS COMPLETE!")
    print("=" * 70)
    print(f"\n  Total time: {elapsed_str}")
    print(f"  Results: {config.output.results_dir}")
    print("\n  Summary:")
    
    for phase, result in results.items():
        if isinstance(result, dict):
            print(f"    [PASS] {phase}: {result}")
        elif result == 'passed':
            print(f"    [PASS] {phase}")
        else:
            print(f"    [FAIL] {phase}: {result}")
    
    print("=" * 70)
    
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Master experiment runner')
    parser.add_argument('--quick', action='store_true', 
                        help='Quick test mode (2 epochs)')
    parser.add_argument('--skip-tests', action='store_true',
                        help='Skip test suite')
    args = parser.parse_args()
    
    success = run_all(quick_test=args.quick, skip_tests=args.skip_tests)
    sys.exit(0 if success else 1)