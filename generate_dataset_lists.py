#!/usr/bin/env python3
"""
Generate dataset list files for RAMSeS testbed

This script scans dataset directories and creates CSV files listing
all available datasets organized by domain.
"""

import os
import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_dataset_list(base_dir: str, output_file: str, pattern: str = "*.csv"):
    """
    Generate dataset list from a directory
    
    Parameters
    ----------
    base_dir : str
        Base directory to scan
    output_file : str
        Output CSV file path
    pattern : str
        File pattern to match
    """
    datasets = []
    
    base_path = Path(base_dir)
    
    if not base_path.exists():
        logger.error(f"Directory not found: {base_dir}")
        return
    
    # Scan for CSV files
    for csv_file in base_path.rglob(pattern):
        # Get relative path
        rel_path = csv_file.relative_to(base_path)
        
        # Determine domain from path structure
        parts = rel_path.parts
        if len(parts) > 1:
            domain = parts[0]
        else:
            domain = base_path.name
        
        datasets.append({
            'file_name': csv_file.name,
            'domain_name': domain,
            'full_path': str(csv_file)
        })
    
    # Create DataFrame
    df = pd.DataFrame(datasets)
    
    # Sort by domain and filename
    df = df.sort_values(['domain_name', 'file_name'])
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    logger.info(f"Generated dataset list with {len(df)} datasets")
    logger.info(f"Saved to: {output_file}")
    
    # Print summary
    logger.info("\nDatasets per domain:")
    for domain, count in df['domain_name'].value_counts().items():
        logger.info(f"  {domain}: {count} datasets")


def generate_skab_list():
    """Generate SKAB dataset list"""
    logger.info("Generating SKAB dataset list...")
    
    datasets = []
    
    # SKAB main datasets
    for i in range(16):  # SKAB_0 to SKAB_15
        datasets.append({
            'file_name': f'SKAB_{i}.csv',
            'domain_name': 'SKAB'
        })
    
    # SKAB valve1 datasets
    for i in range(16):
        datasets.append({
            'file_name': f'SKAB_valve1_{i}.csv',
            'domain_name': 'SKAB_valve1'
        })
    
    # SKAB valve2 datasets
    for i in range(4):
        datasets.append({
            'file_name': f'SKAB_valve2_{i}.csv',
            'domain_name': 'SKAB_valve2'
        })
    
    df = pd.DataFrame(datasets)
    output_file = 'testbed/file_list/test_skab_all.csv'
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)
    logger.info(f"Generated SKAB list: {output_file} ({len(df)} datasets)")


def generate_smd_list():
    """Generate SMD dataset list"""
    logger.info("Generating SMD dataset list...")
    
    datasets = []
    
    # SMD has machine-1-1 through machine-3-11 format
    machines = [
        ('machine-1', 8),
        ('machine-2', 9),
        ('machine-3', 11)
    ]
    
    for machine_prefix, count in machines:
        for i in range(1, count + 1):
            datasets.append({
                'file_name': f'{machine_prefix}-{i}.csv',
                'domain_name': 'SMD'
            })
    
    df = pd.DataFrame(datasets)
    output_file = 'testbed/file_list/test_smd_all.csv'
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)
    logger.info(f"Generated SMD list: {output_file} ({len(df)} datasets)")


def generate_combined_list():
    """Generate combined list of all datasets"""
    logger.info("Generating combined dataset list...")
    
    all_datasets = []
    
    # Read all individual lists
    list_files = [
        'testbed/file_list/test_skab_all.csv',
        'testbed/file_list/test_smd_all.csv'
    ]
    
    for list_file in list_files:
        if os.path.exists(list_file):
            df = pd.read_csv(list_file)
            all_datasets.extend(df.to_dict('records'))
    
    # Create combined DataFrame
    df = pd.DataFrame(all_datasets)
    output_file = 'testbed/file_list/test_all_datasets.csv'
    df.to_csv(output_file, index=False)
    logger.info(f"Generated combined list: {output_file} ({len(df)} datasets)")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate dataset lists for testbed')
    parser.add_argument(
        '--type',
        type=str,
        choices=['skab', 'smd', 'all', 'custom'],
        default='all',
        help='Type of dataset list to generate'
    )
    parser.add_argument(
        '--input-dir',
        type=str,
        help='Input directory for custom dataset scan'
    )
    parser.add_argument(
        '--output-file',
        type=str,
        help='Output file for custom dataset list'
    )
    
    args = parser.parse_args()
    
    if args.type == 'skab':
        generate_skab_list()
    elif args.type == 'smd':
        generate_smd_list()
    elif args.type == 'all':
        generate_skab_list()
        generate_smd_list()
        generate_combined_list()
    elif args.type == 'custom':
        if not args.input_dir or not args.output_file:
            parser.error("--input-dir and --output-file required for custom type")
        generate_dataset_list(args.input_dir, args.output_file)


if __name__ == '__main__':
    main()
