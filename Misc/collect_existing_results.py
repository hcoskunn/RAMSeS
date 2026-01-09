#!/usr/bin/env python3
"""
Collect existing RAMSeS comprehensive results into testbed format

Since app.py currently has hardcoded dataset/entity values, this script
collects already-generated comprehensive results and organizes them
into the testbed format with aggregated statistics.

This is useful when you've already run app.py multiple times and want
to analyze the results collectively.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import json
import re
from collections import defaultdict
import logging

# Add the testbed runner to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_testbed_comprehensive import TestbedRunner

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def collect_existing_results(base_results_dir: str = "myresults/comprehensive", 
                            output_dir: str = "testbed_results_collected"):
    """
    Collect existing comprehensive results and organize them
    
    Parameters
    ----------
    base_results_dir : str
        Base directory where comprehensive results are stored
    output_dir : str
        Output directory for testbed results
    """
    logger.info("=" * 80)
    logger.info("Collecting Existing RAMSeS Results")
    logger.info("=" * 80)
    
    if not os.path.exists(base_results_dir):
        logger.error(f"Results directory not found: {base_results_dir}")
        return
    
    # Create a dummy runner just for parsing
    class ResultsCollector:
        def __init__(self):
            self.runner = TestbedRunner.__new__(TestbedRunner)
            self.results_by_domain = defaultdict(list)
        
        def scan_results(self, base_dir):
            """Scan for comprehensive results"""
            results_pattern = re.compile(r'comprehensive_results_([^_]+)_(\d+)_iter0\.txt')
            
            for root, dirs, files in os.walk(base_dir):
                for file in files:
                    if file.startswith('comprehensive_results_') and file.endswith('.txt'):
                        match = results_pattern.match(file)
                        if match:
                            domain = match.group(1)
                            entity = match.group(2)
                            full_path = os.path.join(root, file)
                            
                            logger.info(f"Found: {domain}/{entity}")
                            
                            # Parse results
                            metrics = self.runner.parse_comprehensive_results(full_path)
                            
                            result = {
                                'dataset_file': f"{domain}_{entity}.csv",
                                'domain': domain,
                                'entity': entity,
                                'total_runtime': metrics.get('timing', {}).get('end_to_end', 0),
                                'avg_memory_mb': 0,  # Not available from file
                                'peak_memory_mb': 0,  # Not available from file
                                'metrics': metrics
                            }
                            
                            self.results_by_domain[domain].append(result)
            
            return dict(self.results_by_domain)
    
    collector = ResultsCollector()
    all_results = collector.scan_results(base_results_dir)
    
    if not all_results:
        logger.warning("No comprehensive results found!")
        logger.warning(f"Looked in: {base_results_dir}")
        logger.warning("Run app.py to generate results first")
        return
    
    logger.info(f"\nFound results for {len(all_results)} domains")
    for domain, results in all_results.items():
        logger.info(f"  {domain}: {len(results)} datasets")
    
    # Generate reports
    os.makedirs(output_dir, exist_ok=True)
    
    # Set output_base_dir for the runner
    collector.runner.output_base_dir = output_dir
    
    for domain, results in all_results.items():
        logger.info(f"\nProcessing domain: {domain}")
        
        # Save intermediate results
        save_results(output_dir, domain, results)
        
        # Compute statistics
        stats = collector.runner.compute_domain_statistics(results)
        
        # Generate report
        collector.runner.generate_domain_report(domain, results, stats)
    
    # Generate overall summary
    collector.runner.output_base_dir = output_dir
    collector.runner.generate_overall_summary(all_results)
    
    logger.info("\n" + "=" * 80)
    logger.info("Collection Complete!")
    logger.info(f"Results saved in: {output_dir}/")
    logger.info("=" * 80)
    
    # Print summary
    logger.info("\nNext steps:")
    logger.info(f"  1. View summary: cat {output_dir}/overall_summary.txt")
    logger.info(f"  2. Generate plots: python visualize_testbed_comprehensive.py --results-dir {output_dir}")


def save_results(output_dir, domain, results):
    """Helper to save intermediate results"""
    output_file = f"{output_dir}/{domain}/intermediate_results.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Collect existing RAMSeS comprehensive results'
    )
    parser.add_argument(
        '--results-dir',
        type=str,
        default='myresults/comprehensive',
        help='Directory containing comprehensive results'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='testbed_results_collected',
        help='Output directory for testbed format'
    )
    
    args = parser.parse_args()
    
    collect_existing_results(args.results_dir, args.output_dir)


if __name__ == '__main__':
    main()
