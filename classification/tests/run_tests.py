#!/usr/bin/env python3
# ==============================================================================
# Copyright (c) 2026 Himanshu Kumar, IIT Bhubaneswar. All Rights Reserved.
# Email: hkphimanshukumar321@gmail.com
# ==============================================================================

"""
Test Runner
===========

Master script to run all test categories.

Usage:
    python run_tests.py           # Run all tests
    python run_tests.py --smoke   # Smoke tests only
    python run_tests.py --quick   # Skip long tests
"""

import sys
import json
import argparse
from datetime import datetime
from pathlib import Path

# Add parent to path
TEMPLATE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(TEMPLATE_DIR))


def run_all_tests(smoke_only: bool = False, generate_report: bool = True):
    """Run all test categories."""
    from tests.test_smoke import run_smoke_tests
    
    print("\n" + "=" * 70)
    print("RESEARCH TEMPLATE - TEST SUITE")
    print("=" * 70)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    all_results = {}
    
    # Smoke tests
    print("\n\n" + "🔥 " * 10)
    print("SMOKE TESTS")
    print("🔥 " * 10)
    smoke_results = run_smoke_tests()
    all_results['smoke'] = {
        'passed': smoke_results.passed,
        'failed': smoke_results.failed,
        'errors': smoke_results.errors
    }
    
    if not smoke_only and smoke_results.failed == 0:
        # Functional tests
        print("\n\n" + "⚡ " * 10)
        print("FUNCTIONAL TESTS")
        print("⚡ " * 10)
        from tests.test_functional import run_functional_tests
        func_results = run_functional_tests()
        all_results['functional'] = {
            'passed': func_results.passed,
            'failed': func_results.failed,
            'errors': func_results.errors
        }
        
        # Integration tests
        print("\n\n" + "🔗 " * 10)
        print("INTEGRATION TESTS")
        print("🔗 " * 10)
        from tests.test_integration import run_integration_tests
        int_results = run_integration_tests()
        all_results['integration'] = {
            'passed': int_results.passed,
            'failed': int_results.failed,
            'errors': int_results.errors
        }
        
        # Performance tests
        print("\n\n" + "🚀 " * 10)
        print("PERFORMANCE TESTS")
        print("🚀 " * 10)
        from tests.test_performance import run_performance_tests
        perf_results = run_performance_tests()
        all_results['performance'] = {
            'passed': perf_results.passed,
            'failed': perf_results.failed,
            'errors': perf_results.errors
        }
    
    # Summary
    total_passed = sum(r['passed'] for r in all_results.values())
    total_failed = sum(r['failed'] for r in all_results.values())
    
    print("\n\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    
    for category, result in all_results.items():
        status = "✅" if result['failed'] == 0 else "❌"
        print(f"  {status} {category.upper()}: {result['passed']} passed, {result['failed']} failed")
    
    print("-" * 70)
    print(f"  TOTAL: {total_passed} passed, {total_failed} failed")
    print("=" * 70)
    
    # Generate report
    if generate_report:
        report_path = TEMPLATE_DIR / "results" / "test_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_passed': total_passed,
            'total_failed': total_failed,
            'all_pass': total_failed == 0,
            'categories': all_results
        }
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📝 Report: {report_path}")
    
    return total_failed == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run test suite')
    parser.add_argument('--smoke', action='store_true', help='Smoke tests only')
    parser.add_argument('--no-report', action='store_true', help='Skip report')
    args = parser.parse_args()
    
    success = run_all_tests(smoke_only=args.smoke, generate_report=not args.no_report)
    sys.exit(0 if success else 1)