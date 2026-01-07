#!/usr/bin/env python3
"""
Quick test script for online phase analysis setup verification
"""

import sys
import os

def check_dependencies():
    """Check if all required packages are installed"""
    print("Checking dependencies...")
    
    required = {
        'numpy': 'numpy',
        'pandas': 'pandas',
        'torch': 'torch',
        'psutil': 'psutil',
        'matplotlib': 'matplotlib',
        'sklearn': 'scikit-learn'
    }
    
    missing = []
    for module, package in required.items():
        try:
            __import__(module)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} (MISSING)")
            missing.append(package)
    
    if missing:
        print(f"\nMissing packages: {', '.join(missing)}")
        print("Install with: pip install " + " ".join(missing))
        return False
    
    print("All dependencies found!")
    return True


def check_data_structure(base_dir='./Mononito'):
    """Check if data directory structure exists"""
    print(f"\nChecking data structure in {base_dir}...")
    
    if not os.path.exists(base_dir):
        print(f"  ✗ Data directory not found: {base_dir}")
        return False
    
    print(f"  ✓ Data directory exists: {base_dir}")
    
    # Check for datasets
    datasets = ['skab', 'smd', 'anomaly_archive']
    found_datasets = []
    
    for dataset in datasets:
        dataset_path = os.path.join(base_dir, dataset)
        if os.path.exists(dataset_path):
            found_datasets.append(dataset)
            print(f"  ✓ Found dataset: {dataset}")
    
    if not found_datasets:
        print("  ⚠ No datasets found")
        return False
    
    return True


def check_trained_models(base_dir='./Mononito/trained_models'):
    """Check if trained models exist"""
    print(f"\nChecking trained models in {base_dir}...")
    
    if not os.path.exists(base_dir):
        print(f"  ✗ Trained models directory not found: {base_dir}")
        print("  → Run 'python app.py --dataset <dataset> --entity <entity>' to train models")
        return False
    
    print(f"  ✓ Trained models directory exists: {base_dir}")
    
    # Look for any .pth files
    model_count = 0
    for root, dirs, files in os.walk(base_dir):
        pth_files = [f for f in files if f.endswith('.pth')]
        if pth_files:
            rel_path = os.path.relpath(root, base_dir)
            print(f"  ✓ Found {len(pth_files)} models in {rel_path}")
            model_count += len(pth_files)
    
    if model_count == 0:
        print("  ⚠ No trained models found (.pth files)")
        print("  → Train models first using app.py")
        return False
    
    print(f"  Total: {model_count} trained models found")
    return True


def check_dataset_lists():
    """Check if dataset list files exist"""
    print("\nChecking dataset list files...")
    
    test_dir = 'testbed/file_list'
    if not os.path.exists(test_dir):
        print(f"  ✗ Dataset list directory not found: {test_dir}")
        return False
    
    print(f"  ✓ Dataset list directory exists: {test_dir}")
    
    lists = [
        'test_m_skab.csv',
        'test_m_smd.csv',
        'ucr_sample_10.csv',
        'test_single.csv'
    ]
    
    for list_file in lists:
        path = os.path.join(test_dir, list_file)
        if os.path.exists(path):
            print(f"  ✓ {list_file}")
        else:
            print(f"  ✗ {list_file} (missing)")
    
    return True


def main():
    print("="*60)
    print("RAMSeS Online Phase Analysis - Setup Verification")
    print("="*60)
    print()
    
    checks = [
        ("Dependencies", check_dependencies),
        ("Data Structure", lambda: check_data_structure()),
        ("Trained Models", lambda: check_trained_models()),
        ("Dataset Lists", check_dataset_lists)
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ Error checking {name}: {e}")
            results.append((name, False))
        print()
    
    print("="*60)
    print("Summary")
    print("="*60)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{name:20s} : {status}")
    
    all_passed = all(result for _, result in results)
    
    print()
    if all_passed:
        print("✓ All checks passed! Ready to run online phase analysis.")
        print()
        print("Next steps:")
        print("  1. Review configuration in ONLINE_PHASE_ANALYSIS_README.md")
        print("  2. Run: ./run_online_phase_analysis.sh testbed/file_list/test_single.csv")
        return 0
    else:
        print("✗ Some checks failed. Please resolve issues above.")
        print()
        print("Common solutions:")
        print("  - Install dependencies: pip install -r requirements.txt psutil")
        print("  - Download data: See README.md for dataset links")
        print("  - Train models: python app.py --dataset <dataset> --entity <entity>")
        return 1


if __name__ == '__main__':
    sys.exit(main())
