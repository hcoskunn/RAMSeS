#!/usr/bin/env python3
"""
Quick test of the testbed runner with a small subset of datasets.
"""

import sys
sys.path.insert(0, '.')

from run_full_testbed import TestbedRunner

# Test with just 2 datasets
runner = TestbedRunner('Configs/custom_config.yml', 'myresults/testbed_test')
runner.datasets = [
    ("SKAB", "2"),
    ("SKAB", "3"),
]

print("Testing with 2 datasets: SKAB/2 and SKAB/3")
runner.run_all_experiments()
runner.generate_aggregated_report()
print("\n✅ Test completed! Check myresults/testbed_test/ for outputs")
