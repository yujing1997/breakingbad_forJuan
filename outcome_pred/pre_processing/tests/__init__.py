#!/usr/bin/env python3
"""
Test Runner for HECKTOR Preprocessing Pipeline

This script runs all tests in the tests directory.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

def run_all_tests():
    """Run all available tests"""
    
    print("="*60)
    print("HECKTOR PREPROCESSING PIPELINE - TEST SUITE")
    print("="*60)
    
    tests_dir = Path(__file__).parent
    
    # Test 1: Unit tests
    print(f"\n🧪 Running Unit Tests...")
    try:
        from test_preprocessing import main as run_unit_tests
        run_unit_tests()
        print("✅ Unit tests passed")
    except Exception as e:
        print(f"❌ Unit tests failed: {e}")
    
    # Test 2: Visualization test (if user wants to run it)
    print(f"\n🎨 Visualization Test Available")
    print("To run visualization test:")
    print("cd tests && python visualize_preprocessing.py")
    
    print(f"\n✅ Test suite completed!")

if __name__ == "__main__":
    run_all_tests()
