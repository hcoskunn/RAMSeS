# RAMSeS Full Testbed Runner - Documentation

## Overview
This tool runs the RAMSeS framework across all 36 datasets and generates comprehensive aggregated reports.

## Datasets Covered
- **SKAB**: 16 entities (0-15)
- **SKAB_valve1**: 16 entities (0-15)
- **SKAB_valve2**: 4 entities (0-3)

**Total: 36 datasets**

## Usage

### Quick Start (Recommended)
```bash
./run_testbed.sh
```

### Advanced Usage
```bash
# Run with default config
python run_full_testbed.py

# Run with custom config
python run_full_testbed.py -c path/to/config.yml

# Specify output directory
python run_full_testbed.py -o myresults/custom_output

# Full options
python run_full_testbed.py -c Configs/custom_config.yml -o myresults/testbed_aggregated
```

## Output Files

The script generates 5 types of output files in `myresults/testbed_aggregated/`:

### 1. **Detailed Results CSV** (`detailed_results_YYYYMMDD_HHMMSS.csv`)
Complete dataset with all experiments and metrics:
- Dataset name, entity ID
- Success/failure status
- F1 scores for all 5 methods (GA, Thompson, GAN, Borderline, Monte Carlo)
- PR-AUC scores for all 5 methods
- Runtime for each module
- Total runtime per experiment

### 2. **Summary Statistics** (`summary_statistics_YYYYMMDD_HHMMSS.txt`)
Human-readable text report containing:
- **Overall Statistics**: Success/failure counts
- **Performance Metrics**: Mean ± Std for all F1 and PR-AUC scores
- **Computational Overhead**: Mean, std, and total time for each module
- **Per-Dataset Breakdown**: Statistics grouped by dataset (SKAB, SKAB_valve1, SKAB_valve2)

Example:
```
OVERALL PERFORMANCE METRICS (Mean ± Std)
Genetic Algorithm F1        : 0.765432 ± 0.045678
Thompson Sampling F1        : 0.887654 ± 0.023456
...

COMPUTATIONAL OVERHEAD (seconds)
Genetic Algorithm           :    30.45s ±     5.23s  (Total:   1096.20s)
Thompson Sampling           :    14.32s ±     2.11s  (Total:    515.52s)
...
```

### 3. **JSON Aggregate** (`aggregated_results_YYYYMMDD_HHMMSS.json`)
Machine-readable structured data:
```json
{
  "metadata": {
    "total_experiments": 36,
    "successful_experiments": 36,
    "total_runtime_hours": 24.5
  },
  "overall_metrics": {
    "ga_f1": {"mean": 0.77, "std": 0.05, "min": 0.65, "max": 0.85},
    ...
  },
  "timing_metrics": {
    "ga_duration": {"mean": 30.5, "std": 5.2, "total": 1098.0},
    ...
  },
  "per_dataset_metrics": {
    "SKAB": {"count": 16, "ga_f1_mean": 0.78, ...},
    ...
  },
  "detailed_results": [...]
}
```

### 4. **Method Comparison CSV** (`method_comparison_YYYYMMDD_HHMMSS.csv`)
Side-by-side comparison of all 5 methods:
```
Method              F1_Mean   F1_Std   PR_AUC_Mean  PR_AUC_Std  Time_Mean_s  Time_Std_s  Time_Total_s
GA                  0.7654    0.0456   0.0234       0.0123      30.45        5.23        1096.20
THOMPSON            0.8876    0.0234   0.4567       0.0234      14.32        2.11        515.52
GAN                 0.8123    0.0345   0.3456       0.0234      15.67        2.34        564.12
BORDERLINE          0.8234    0.0456   0.4234       0.0345      6.78         1.23        244.08
MONTECARLO          0.8345    0.0456   0.4456       0.0456      15.12        2.45        544.32
```

### 5. **Intermediate Results** (`intermediate_results_YYYYMMDD_HHMMSS.json`)
Auto-saved after each experiment completes (for crash recovery).

## Features

### ✅ Robust Error Handling
- Catches and logs all errors
- Continues processing remaining datasets even if one fails
- Timeout protection (1 hour per experiment)
- Saves intermediate results after each experiment

### ✅ Progress Tracking
- Real-time progress display (X/36 completed)
- Status indicators (✅ success, ❌ failed, ⏱️ timeout)
- Runtime estimation

### ✅ Comprehensive Metrics
For each method (GA, Thompson, GAN, Borderline, Monte Carlo):
- F1 Score (mean, std, min, max)
- PR-AUC (mean, std, min, max)
- Runtime (mean, std, total)

### ✅ Statistical Analysis
- Overall aggregates across all datasets
- Per-dataset breakdowns
- Method-to-method comparisons

## Estimations

### Time Requirements
- **Per experiment**: ~40-50 minutes (varies by dataset size)
- **Total for 36 datasets**: 24-30 hours
- **Can be parallelized**: Run multiple instances on different datasets

### Resource Requirements
- **CPU**: Moderate to high (TensorFlow + PyTorch)
- **Memory**: 8-16 GB recommended
- **Disk**: ~2 GB for all results
- **GPU**: Optional (will speed up deep learning models)

## Tips

### 1. Run in Background
```bash
# Using nohup
nohup ./run_testbed.sh > testbed_run.log 2>&1 &

# Using screen
screen -S ramses_testbed
./run_testbed.sh
# Ctrl+A then D to detach
# screen -r ramses_testbed to reattach
```

### 2. Monitor Progress
```bash
# Watch the log
tail -f testbed_run.log

# Check intermediate results
ls -lh myresults/testbed_aggregated/intermediate_*
```

### 3. Parallel Execution (Advanced)
You can run multiple instances in parallel by splitting datasets:
```bash
# Terminal 1: SKAB datasets
python run_full_testbed.py --datasets SKAB

# Terminal 2: SKAB_valve1 datasets
python run_full_testbed.py --datasets SKAB_valve1

# Terminal 3: SKAB_valve2 datasets
python run_full_testbed.py --datasets SKAB_valve2
```

## Troubleshooting

### Issue: Experiment times out
- Increase timeout in `run_full_testbed.py` (line with `timeout=3600`)
- Check if dataset is too large or model selection is stuck

### Issue: Script crashes midway
- Intermediate results are saved automatically
- Check `intermediate_results_*.json` for completed experiments
- Manually resume by modifying dataset list in script

### Issue: Memory errors
- Reduce batch sizes in config file
- Run fewer experiments in parallel
- Close other applications

## Example Output

```
================================================================================
Starting Full Testbed Run
Total datasets: 36
Start time: 2025-12-24 10:30:00
================================================================================

Progress: 1/36 (2.8%)
================================================================================
Running: SKAB/0
================================================================================
✅ SUCCESS: SKAB/0 (Runtime: 1847.32s)

Progress: 2/36 (5.6%)
================================================================================
Running: SKAB/1
================================================================================
✅ SUCCESS: SKAB/1 (Runtime: 1923.45s)

...

================================================================================
Testbed Run Completed
Total time: 24.67 hours
================================================================================

Generating aggregated report...
Successful experiments: 36/36
✅ Detailed CSV saved: myresults/testbed_aggregated/detailed_results_20251224_103000.csv
✅ Summary statistics saved: myresults/testbed_aggregated/summary_statistics_20251224_103000.txt
✅ JSON aggregate saved: myresults/testbed_aggregated/aggregated_results_20251224_103000.json
✅ Method comparison saved: myresults/testbed_aggregated/method_comparison_20251224_103000.csv
```

## Integration with Analysis Tools

The generated CSV and JSON files can be easily imported into:
- **Python/Pandas**: `df = pd.read_csv('detailed_results_*.csv')`
- **R**: `data <- read.csv('detailed_results_*.csv')`
- **Excel**: Open CSV files directly
- **Jupyter Notebooks**: Load JSON for interactive analysis
- **Plotting libraries**: matplotlib, seaborn, plotly

## Additional Notes

- All individual experiment results are preserved in `myresults/comprehensive/`
- The testbed runner only aggregates and summarizes existing results
- Safe to re-run: will generate new aggregated reports without re-running experiments
- Compatible with existing RAMSeS output format
