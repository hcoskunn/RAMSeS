#!/usr/bin/env python
"""
Configuration Validation Script

This script validates your RAMSeS configuration file and checks that all
required paths and parameters are properly set.

Usage:
    python validate_config.py
    python validate_config.py -c Configs/custom_config.yml
"""

import os
import sys
from pathlib import Path
from argparse import ArgumentParser
from Utils.config import Config


def validate_config(config_path):
    """Validate the configuration file and check all paths."""
    
    print("=" * 80)
    print("RAMSeS Configuration Validator")
    print("=" * 80)
    print(f"\nValidating config file: {config_path}")
    
    errors = []
    warnings = []
    
    # Check if config file exists
    if not os.path.exists(config_path):
        print(f"\n❌ ERROR: Config file not found: {config_path}")
        return False
    
    print("✓ Config file exists")
    
    # Load config
    try:
        config = Config(config_file_path=config_path).parse()
        print("✓ Config file is valid YAML")
    except Exception as e:
        print(f"\n❌ ERROR: Failed to parse config file: {e}")
        return False
    
    # Required parameters
    required_params = [
        'dataset',
        'entity',
        'trained_model_path',
        'dataset_path',
        'downsampling',
        'min_length',
    ]
    
    print("\nChecking required parameters:")
    for param in required_params:
        if param in config:
            print(f"  ✓ {param}: {config[param]}")
        else:
            errors.append(f"Missing required parameter: {param}")
            print(f"  ❌ {param}: MISSING")
    
    # Validate paths
    print("\nValidating paths:")
    
    path_params = ['trained_model_path', 'dataset_path', 'results_path']
    for param in path_params:
        if param in config:
            path = config[param]
            # Skip validation for Docker paths
            if path.startswith('/trained_models') or path.startswith('/datasets') or path.startswith('/results'):
                print(f"  ⚠ {param}: {path} (Docker path - skipping validation)")
                continue
            
            if os.path.exists(path):
                print(f"  ✓ {param}: {path}")
            else:
                warnings.append(f"Path does not exist: {param} = {path}")
                print(f"  ⚠ {param}: {path} (does not exist - will be created if needed)")
    
    # Validate dataset and entity
    print("\nValidating dataset configuration:")
    
    dataset = config.get('dataset')
    entity = config.get('entity')
    dataset_path = config.get('dataset_path')
    
    if dataset and entity and dataset_path:
        # Check if dataset is single or list
        datasets = [dataset] if isinstance(dataset, str) else dataset
        entities = [entity] if isinstance(entity, (str, int)) else entity
        
        print(f"  Dataset(s): {datasets}")
        print(f"  Entity/Entities: {entities}")
        
        # Check if dataset files exist
        if not dataset_path.startswith('/datasets'):  # Skip for Docker paths
            for ds in datasets:
                for ent in entities:
                    dataset_file = os.path.join(dataset_path, ds, f"{ent}.csv")
                    if os.path.exists(dataset_file):
                        print(f"    ✓ Found: {dataset_file}")
                    else:
                        warnings.append(f"Dataset file not found: {dataset_file}")
                        print(f"    ⚠ Not found: {dataset_file}")
    
    # Validate trained models
    print("\nValidating trained models:")
    
    trained_model_path = config.get('trained_model_path')
    if dataset and entity and trained_model_path:
        if not trained_model_path.startswith('/trained_models'):  # Skip for Docker paths
            datasets = [dataset] if isinstance(dataset, str) else dataset
            entities = [entity] if isinstance(entity, (str, int)) else entity
            
            for ds in datasets:
                for ent in entities:
                    model_dir = os.path.join(trained_model_path, ds, str(ent))
                    if os.path.exists(model_dir):
                        model_files = list(Path(model_dir).glob('*.pth'))
                        if model_files:
                            print(f"    ✓ Found {len(model_files)} model(s) in: {model_dir}")
                        else:
                            warnings.append(f"No .pth files in: {model_dir}")
                            print(f"    ⚠ No .pth files in: {model_dir}")
                    else:
                        warnings.append(f"Model directory not found: {model_dir}")
                        print(f"    ⚠ Not found: {model_dir}")
    
    # Summary
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    
    if errors:
        print(f"\n❌ ERRORS ({len(errors)}):")
        for error in errors:
            print(f"  - {error}")
    
    if warnings:
        print(f"\n⚠ WARNINGS ({len(warnings)}):")
        for warning in warnings:
            print(f"  - {warning}")
    
    if not errors and not warnings:
        print("\n✓ All checks passed! Configuration is valid.")
        return True
    elif not errors:
        print("\n⚠ Configuration is valid but has warnings.")
        print("  These may be okay if you're setting up for the first time.")
        return True
    else:
        print("\n❌ Configuration has errors that must be fixed.")
        return False


def main():
    parser = ArgumentParser(description='Validate RAMSeS configuration file')
    parser.add_argument(
        '--config_file_path', '-c',
        type=str,
        default='Configs/config.yml',
        help='Path to config file'
    )
    
    args = parser.parse_args()
    
    success = validate_config(args.config_file_path)
    
    print("\n" + "=" * 80)
    if success:
        print("Configuration validation completed successfully!")
        sys.exit(0)
    else:
        print("Configuration validation failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
