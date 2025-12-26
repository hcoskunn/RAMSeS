# RAMSeS Comprehensive Testbed System - Summary

## 🎯 What You Got

I've created a complete testbed system for running RAMSeS across multiple datasets with comprehensive metrics collection and visualization. This system is ready to use!

## 📦 Files Created

### Core Scripts

1. **`run_testbed_comprehensive.py`** (Main testbed runner)
   - Runs RAMSeS on multiple datasets
   - Organizes results by domain (SKAB, SMD, etc.)
   - Collects comprehensive metrics
   - Monitors memory usage in real-time
   - Generates detailed reports

2. **`generate_dataset_lists.py`** (Dataset list generator)
   - Creates CSV files listing datasets
   - Supports SKAB, SMD, and custom datasets
   - Organizes by domain

3. **`visualize_testbed_comprehensive.py`** (Visualization generator)
   - Creates publication-ready plots
   - Comprehensive dashboard
   - Multiple chart types

4. **`run_testbed_quickstart.sh`** (Interactive quick start)
   - User-friendly interface
   - Step-by-step guidance
   - Multiple run modes

### Documentation

5. **`TESTBED_COMPREHENSIVE_GUIDE.md`** (Complete guide)
   - Full documentation
   - Usage examples
   - Troubleshooting tips

## 📊 Metrics Collected

### ✅ Computational Overhead
- **Overall**: Total time across all datasets
- **Average**: Mean time per dataset
- **Per-Module**: Timing for each RAMSeS module:
  - Genetic Algorithm
  - Thompson Sampling
  - GAN Robustness
  - Borderline Sensitivity
  - Monte Carlo
  - Rank Aggregation
- **End-to-End**: Complete pipeline timing

### ✅ F1 Scores
For each module:
- Genetic Algorithm (Ensemble)
- Thompson Sampling
- GAN Robustness
- Borderline Sensitivity
- Monte Carlo
- **Final Selected** (Ensemble or Single Model)

Plus standard deviation across datasets!

### ✅ Memory Footprint
- **Average Memory**: Mean memory usage during execution
- **Peak Memory**: Maximum memory reached
- **Per-Dataset**: Individual memory profiles

## 🚀 Quick Start

### Option 1: Use Existing Dataset List

```bash
./run_testbed_quickstart.sh
# Select option 1
```

### Option 2: Command Line

```bash
# Run on specific dataset list
python run_testbed_comprehensive.py \
    --dataset-list /path/to/test_m_skab.csv \
    --output-dir testbed_results

# Generate visualizations
python visualize_testbed_comprehensive.py \
    --results-dir testbed_results \
    --plot all
```

### Option 3: Full Automated Pipeline

```bash
./run_testbed_quickstart.sh
# Select option 5 for full pipeline
```

## 📁 Output Structure

```
testbed_results/
├── overall_summary.txt          # ⭐ Human-readable summary
├── overall_summary.json         # 📊 Machine-readable data
├── SKAB/                        # Per-domain results
│   ├── domain_report.txt        # Detailed domain report
│   └── intermediate_results.json
├── SMD/
│   ├── domain_report.txt
│   └── intermediate_results.json
└── plots/                       # 📈 Visualizations
    ├── computational_overhead.png
    ├── module_timing_breakdown.png
    ├── f1_scores_comparison.png
    ├── memory_usage.png
    └── overall_summary_dashboard.png
```

## 📈 Visualizations Generated

1. **Computational Overhead**
   - Average runtime per dataset (by domain)
   - Total runtime per domain

2. **Module Timing Breakdown**
   - Grouped bar chart showing time spent in each module
   - Compare across domains

3. **F1 Score Comparison**
   - F1 scores for all modules
   - Compare across domains
   - Shows final selected performance

4. **Memory Usage**
   - Average memory usage
   - Peak memory usage

5. **Summary Dashboard**
   - All metrics in one comprehensive view
   - Publication-ready format

## 🎯 Example Use Cases

### Use Case 1: Benchmark on SKAB Dataset

```bash
# Using the existing test_m_skab.csv
python run_testbed_comprehensive.py \
    --dataset-list /home/maxoud/local-storage/projects/TSB-AutoAD/testbed/file_list/test_m_skab.csv \
    --output-dir testbed_skab

# View results
cat testbed_skab/overall_summary.txt
```

### Use Case 2: Compare Multiple Domains

```bash
# Generate lists for all domains
python generate_dataset_lists.py --type all

# Run testbed on all
python run_testbed_comprehensive.py \
    --dataset-list testbed/file_list/test_all_datasets.csv \
    --output-dir testbed_comparison

# Generate comparison plots
python visualize_testbed_comprehensive.py \
    --results-dir testbed_comparison \
    --plot all
```

### Use Case 3: Single Domain Deep Dive

```bash
# Run only SKAB domain
python run_testbed_comprehensive.py \
    --dataset-list testbed/file_list/test_all_datasets.csv \
    --output-dir testbed_skab_only \
    --domain SKAB

# View detailed report
cat testbed_skab_only/SKAB/domain_report.txt
```

## 📊 Sample Output

### Overall Summary
```
================================================================================
RAMSeS Testbed - Overall Summary
================================================================================

Total Domains: 3
Total Datasets: 36
Total Computational Time: 3845.67s (1.07 hours)

PER-DOMAIN SUMMARY
--------------------------------------------------
Domain: SKAB
  Datasets: 16
  Total Time: 1709.23s
  Avg Time: 106.83s
  Avg Memory: 2456.78 MB
  Avg Peak Memory: 3012.45 MB
  Avg F1 (Final): 0.989012

Domain: SKAB_valve1
  Datasets: 16
  Total Time: 1850.34s
  Avg Time: 115.65s
  Avg Memory: 2501.23 MB
  Avg Peak Memory: 3145.67 MB
  Avg F1 (Final): 0.983456

Domain: SKAB_valve2
  Datasets: 4
  Total Time: 286.10s
  Avg Time: 71.53s
  Avg Memory: 2234.56 MB
  Avg Peak Memory: 2789.01 MB
  Avg F1 (Final): 0.991234
```

### Domain Report (Detailed)
```
================================================================================
AVERAGE MODULE COMPUTATIONAL OVERHEAD
================================================================================

  1_Genetic_Algorithm            :    45.3421s
  2_Thompson_Sampling            :    14.8765s
  3_GAN_Robustness               :    15.2341s
  4_Borderline_Sensitivity       :     6.0123s
  5_Monte_Carlo                  :    22.4567s
  6_Rank_Aggregation             :     0.0087s
  End-to-End (Average)           :   106.8123s

AVERAGE F1 SCORES BY MODULE
================================================================================

  ga                   : 0.985432 ± 0.012345
  thompson             : 0.876543 ± 0.045678
  gan                  : 0.987654 ± 0.009876
  borderline           : 0.987123 ± 0.010234
  monte_carlo          : 0.345678 ± 0.123456
  final_selected       : 0.989012 ± 0.008765
```

## 🔧 Customization

### Modify RAMSeS Configuration

Edit `create_config()` in `run_testbed_comprehensive.py`:

```python
def create_config(self, dataset: str, entity: str) -> Dict:
    return {
        'dataset': dataset,
        'entity': entity,
        'iterations': 1,
        'anomaly_list': ['point', 'contextual'],
        'population_size': 20,       # 🔧 Adjust GA params
        'generations': 20,            # 🔧 Adjust GA params
        'meta_model_type': 'rf',      # 🔧 Change: 'rf', 'lr', 'gbm', 'svm'
        'mutation_rate': 0.1
    }
```

### Add Custom Metrics

Add to `parse_comprehensive_results()`:

```python
# Extract your custom metric
custom_match = re.search(r'My Metric\s*:\s*([\d.]+)', content)
if custom_match:
    metrics['custom']['my_metric'] = float(custom_match.group(1))
```

## ⚡ Performance Tips

1. **Test First**: Run on 1 domain before running all
2. **Monitor Resources**: Use `htop` to watch memory/CPU
3. **Background Execution**: Use `nohup` or `screen` for long runs
4. **Disk Space**: Ensure 10GB+ free for large testbeds
5. **Parallel Domains**: Run different domains in parallel terminals

## 🐛 Troubleshooting

### "Results file not found"
- Check that `app.py` completed successfully
- Verify the comprehensive results file was created
- Look for error messages in terminal output

### Memory errors
- Reduce `population_size` and `generations` in config
- Process fewer datasets at once
- Monitor with `free -h`

### Long runtime
- Normal! Large testbeds can take hours
- Use `--domain` flag to test one domain first
- Consider reducing GA parameters

## 📚 Next Steps

1. **Test the System**:
   ```bash
   ./run_testbed_quickstart.sh
   ```

2. **Run Small Test**:
   ```bash
   # Test with just SKAB domain
   python run_testbed_comprehensive.py \
       --dataset-list /path/to/test_m_skab.csv \
       --output-dir test_run \
       --domain SKAB
   ```

3. **Analyze Results**:
   ```bash
   cat test_run/overall_summary.txt
   ls test_run/plots/
   ```

4. **Full Production Run**:
   ```bash
   ./run_testbed_quickstart.sh
   # Select option 5
   ```

## ✅ System Ready!

Everything is set up and ready to use:
- ✅ Core scripts created and executable
- ✅ Directory structure prepared
- ✅ Documentation complete
- ✅ Quick start script ready
- ✅ Visualization tools configured

Just run `./run_testbed_quickstart.sh` to get started!

## 📞 Support

For issues or questions:
1. Check `TESTBED_COMPREHENSIVE_GUIDE.md` for detailed docs
2. Review error messages in terminal output
3. Verify dataset paths and configurations
4. Check available disk space and memory

Enjoy your comprehensive RAMSeS testbed system! 🚀
