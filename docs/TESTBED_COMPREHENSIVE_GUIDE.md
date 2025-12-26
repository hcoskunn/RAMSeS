# RAMSeS Comprehensive Testbed System

A comprehensive testing framework for running RAMSeS across multiple datasets and collecting detailed performance metrics.

## Features

✅ **Multi-Dataset Testing**: Run RAMSeS on entire dataset collections  
✅ **Domain Organization**: Results organized by dataset domain (SKAB, SMD, etc.)  
✅ **Comprehensive Metrics**: Tracks computational overhead, F1 scores, and memory usage  
✅ **Automated Reports**: Generates detailed reports and summaries  
✅ **Visualization**: Creates publication-ready plots and dashboards  

## Collected Metrics

### Computational Overhead
- Overall computational time (total and per-dataset)
- Average runtime per dataset
- Per-module timing breakdown
- End-to-end execution time

### Performance Metrics
- F1 scores for each module:
  - Genetic Algorithm (Ensemble)
  - Thompson Sampling
  - GAN Robustness
  - Borderline Sensitivity
  - Monte Carlo Simulation
  - Final Selected Model/Ensemble
- PR-AUC scores for all modules

### Resource Usage
- Average memory footprint
- Peak memory usage
- Memory usage distribution

## Quick Start

### 1. Prepare Dataset Lists

Generate dataset lists for your testbed:

```bash
# Generate all standard dataset lists (SKAB, SMD)
python generate_dataset_lists.py --type all

# Generate only SKAB datasets
python generate_dataset_lists.py --type skab

# Generate only SMD datasets
python generate_dataset_lists.py --type smd

# Generate custom dataset list from a directory
python generate_dataset_lists.py --type custom \
    --input-dir /path/to/datasets \
    --output-file testbed/file_list/my_datasets.csv
```

This creates CSV files in `testbed/file_list/` with format:
```csv
file_name,domain_name
SKAB_0.csv,SKAB
SKAB_1.csv,SKAB
...
```

### 2. Run the Testbed

Run the comprehensive testbed on all datasets:

```bash
# Run all domains in the dataset list
python run_testbed_comprehensive.py \
    --dataset-list testbed/file_list/test_skab_all.csv \
    --output-dir testbed_results

# Run only a specific domain
python run_testbed_comprehensive.py \
    --dataset-list testbed/file_list/test_all_datasets.csv \
    --output-dir testbed_results \
    --domain SKAB
```

### 3. Generate Visualizations

Create plots and dashboards from results:

```bash
# Generate all plots
python visualize_testbed_comprehensive.py \
    --results-dir testbed_results \
    --plot all

# Generate specific plot
python visualize_testbed_comprehensive.py \
    --results-dir testbed_results \
    --plot f1  # Options: overhead, modules, f1, memory, summary
```

## Output Structure

The testbed creates the following output structure:

```
testbed_results/
├── overall_summary.txt           # Text summary across all domains
├── overall_summary.json          # JSON summary with all statistics
├── SKAB/                         # Domain-specific folder
│   ├── domain_report.txt         # Detailed report for SKAB domain
│   ├── intermediate_results.json # Raw results for all SKAB datasets
│   └── ...
├── SMD/                          # Another domain
│   ├── domain_report.txt
│   ├── intermediate_results.json
│   └── ...
└── plots/                        # Visualization outputs
    ├── computational_overhead.png
    ├── module_timing_breakdown.png
    ├── f1_scores_comparison.png
    ├── memory_usage.png
    └── overall_summary_dashboard.png
```

## Output Reports

### Domain Report (per domain)

Each domain gets a detailed report (`domain_report.txt`) containing:

```
================================================================================
RAMSeS Testbed Report - Domain: SKAB
================================================================================

OVERALL STATISTICS
--------------------------------------------------
Total Datasets Processed: 36
Total Computational Time: 3845.23s (64.09 min)
Average Runtime per Dataset: 106.81s
Average Memory Usage: 2456.78 MB
Average Peak Memory: 3012.45 MB
Maximum Peak Memory: 3567.89 MB

AVERAGE MODULE COMPUTATIONAL OVERHEAD
--------------------------------------------------
  1_Genetic_Algorithm            :    45.3421s
  2_Thompson_Sampling            :    14.8765s
  3_GAN_Robustness               :    15.2341s
  4_Borderline_Sensitivity       :     6.0123s
  5_Monte_Carlo                  :    22.4567s
  6_Rank_Aggregation             :     0.0087s
  End-to-End (Average)           :   106.8123s

AVERAGE F1 SCORES BY MODULE
--------------------------------------------------
  ga                   : 0.985432 ± 0.012345
  thompson             : 0.876543 ± 0.045678
  gan                  : 0.987654 ± 0.009876
  borderline           : 0.987123 ± 0.010234
  monte_carlo          : 0.345678 ± 0.123456
  final_selected       : 0.989012 ± 0.008765

PER-DATASET RESULTS
--------------------------------------------------
Dataset: SKAB_0.csv
  Entity: 0
  Runtime: 105.32s
  Avg Memory: 2401.23 MB
  Peak Memory: 2987.45 MB
  F1 Scores:
    GA Ensemble: 0.995123
    Final (ensemble): 0.995123
...
```

### Overall Summary

The overall summary (`overall_summary.txt`) provides:

```
================================================================================
RAMSeS Testbed - Overall Summary
================================================================================

Total Domains: 3
Total Datasets: 72
Total Computational Time: 7890.45s (2.19 hours)

PER-DOMAIN SUMMARY
--------------------------------------------------
Domain: SKAB
  Datasets: 36
  Total Time: 3845.23s
  Avg Time: 106.81s
  Avg Memory: 2456.78 MB
  Avg Peak Memory: 3012.45 MB
  Avg F1 (Final): 0.989012

Domain: SMD
  Datasets: 28
  Total Time: 3234.56s
  Avg Time: 115.52s
  Avg Memory: 2678.90 MB
  Avg Peak Memory: 3234.56 MB
  Avg F1 (Final): 0.976543
...
```

## Visualization Examples

The testbed generates comprehensive visualizations:

### 1. Computational Overhead
- Bar charts showing average and total runtime per domain
- Module timing breakdown across domains

### 2. F1 Score Comparison
- Grouped bar charts comparing F1 scores across modules and domains
- Shows performance consistency

### 3. Memory Usage
- Average and peak memory usage per domain
- Helps identify resource requirements

### 4. Summary Dashboard
- Comprehensive single-view dashboard
- All key metrics in one figure
- Publication-ready format

## Advanced Usage

### Custom Configuration

Modify the `create_config` method in `run_testbed_comprehensive.py` to customize:

```python
def create_config(self, dataset: str, entity: str) -> Dict:
    return {
        'dataset': dataset,
        'entity': entity,
        'iterations': 1,
        'anomaly_list': ['point', 'contextual'],
        'population_size': 20,      # Adjust GA parameters
        'generations': 20,
        'meta_model_type': 'rf',    # Options: 'rf', 'lr', 'gbm', 'svm'
        'mutation_rate': 0.1
    }
```

### Parallel Execution

To speed up testbed execution, you can modify the runner to process multiple datasets in parallel:

```python
from concurrent.futures import ProcessPoolExecutor

def run_domain_parallel(self, domain: str, max_workers: int = 4):
    domain_datasets = self.datasets_df[
        self.datasets_df['domain_name'] == domain
    ]['file_name'].tolist()
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(self.run_single_dataset, ds, domain)
            for ds in domain_datasets
        ]
        results = [f.result() for f in futures if f.result()]
    
    return results
```

### Custom Metrics

Add custom metric extraction in `parse_comprehensive_results`:

```python
# Extract custom metrics
custom_match = re.search(r'Custom Metric\s*:\s*([\d.]+)', content)
if custom_match:
    metrics['custom']['my_metric'] = float(custom_match.group(1))
```

## Tips for Large-Scale Testing

1. **Incremental Testing**: Test one domain first to validate setup
2. **Monitor Disk Space**: Results can accumulate, ensure sufficient storage
3. **Background Execution**: Use `nohup` or `screen` for long-running tests
4. **Resource Monitoring**: Use `htop` or similar to monitor system resources
5. **Checkpointing**: The testbed saves intermediate results after each domain

## Example Workflow

```bash
# 1. Setup
cd /home/maxoud/local-storage/projects/RAMSeS

# 2. Generate dataset lists
python generate_dataset_lists.py --type all

# 3. Run testbed (start with one domain)
python run_testbed_comprehensive.py \
    --dataset-list testbed/file_list/test_skab_all.csv \
    --output-dir testbed_results_skab \
    --domain SKAB

# 4. Check results
cat testbed_results_skab/SKAB/domain_report.txt

# 5. Generate visualizations
python visualize_testbed_comprehensive.py \
    --results-dir testbed_results_skab \
    --plot all

# 6. View plots
ls testbed_results_skab/plots/

# 7. Run full testbed on all domains
python run_testbed_comprehensive.py \
    --dataset-list testbed/file_list/test_all_datasets.csv \
    --output-dir testbed_results_full

# 8. Generate comprehensive visualizations
python visualize_testbed_comprehensive.py \
    --results-dir testbed_results_full \
    --plot all
```

## Troubleshooting

### Memory Issues
If you encounter memory errors:
- Reduce dataset batch size
- Monitor with `free -h` during execution
- Increase swap space if needed

### Long Runtime
For faster execution:
- Use smaller `generations` and `population_size` in config
- Process domains separately
- Consider parallel execution

### Missing Results
If comprehensive results are not found:
- Check that `app.py` completed successfully
- Verify dataset paths in the CSV file
- Check logs for errors

## Requirements

- Python 3.7+
- pandas
- numpy
- matplotlib
- seaborn
- psutil
- pyyaml

Install with:
```bash
pip install pandas numpy matplotlib seaborn psutil pyyaml
```

## Contributing

To add new metrics or visualizations:
1. Modify `parse_comprehensive_results()` to extract new metrics
2. Update `compute_domain_statistics()` to aggregate new metrics
3. Add new plotting functions to `TestbedVisualizer`
4. Update report generation templates

## Citation

If you use this testbed system in your research, please cite:

```
@misc{ramses_testbed,
  title={RAMSeS Comprehensive Testbed System},
  author={Your Name},
  year={2025}
}
```
