# RAMSeS Full Testbed Automation - Quick Reference

## 🎯 What You Asked For

You want to run RAMSeS across all your time-series datasets and get:
1. **F1 scores** for all datasets (individual and average)
2. **Computational overhead** for all datasets (individual and average)
3. **Aggregated output file** with all results

## ✅ What I Created

### 1. Main Script: `run_full_testbed.py`
**Purpose**: Automated batch processing of all 36 datasets

**Features**:
- ✅ Runs app.py on all datasets automatically
- ✅ Collects F1 scores from all 5 methods
- ✅ Tracks computational overhead per module
- ✅ Generates 5 types of output reports
- ✅ Saves progress after each experiment
- ✅ Error handling & timeout protection

### 2. Quick Launcher: `run_testbed.sh`
**Purpose**: One-command execution

```bash
./run_testbed.sh
```

### 3. Test Script: `test_testbed_runner.py`
**Purpose**: Test with 2 datasets before running all 36

```bash
python test_testbed_runner.py
```

## 📊 Output Files You'll Get

### File 1: Detailed Results CSV
**Filename**: `detailed_results_YYYYMMDD_HHMMSS.csv`

| dataset | entity | status | ga_f1 | thompson_f1 | gan_f1 | borderline_f1 | montecarlo_f1 | ga_duration | thompson_duration | ... |
|---------|--------|--------|-------|-------------|--------|---------------|---------------|-------------|-------------------|-----|
| SKAB | 0 | success | 0.7654 | 0.8876 | 0.8123 | 0.8234 | 0.8345 | 30.45 | 14.32 | ... |
| SKAB | 1 | success | 0.7712 | 0.8923 | 0.8201 | 0.8312 | 0.8423 | 31.23 | 14.67 | ... |
| ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |

### File 2: Summary Statistics (Text)
**Filename**: `summary_statistics_YYYYMMDD_HHMMSS.txt`

```
OVERALL PERFORMANCE METRICS (Mean ± Std)
Genetic Algorithm F1        : 0.771037 ± 0.045678
Thompson Sampling F1        : 0.887654 ± 0.023456
GAN Robustness F1          : 0.812345 ± 0.034567
Borderline Sensitivity F1  : 0.823456 ± 0.045678
Monte Carlo F1             : 0.834567 ± 0.045678

COMPUTATIONAL OVERHEAD (seconds)
Genetic Algorithm           :    30.45s ±     5.23s  (Total:   1096.20s)
Thompson Sampling           :    14.32s ±     2.11s  (Total:    515.52s)
GAN Robustness             :    15.67s ±     2.34s  (Total:    564.12s)
Borderline Sensitivity     :     6.78s ±     1.23s  (Total:    244.08s)
Monte Carlo                :    15.12s ±     2.45s  (Total:    544.32s)
Total Pipeline             :    82.34s ±    10.56s  (Total:   2964.24s)
```

### File 3: JSON Aggregate
**Filename**: `aggregated_results_YYYYMMDD_HHMMSS.json`

```json
{
  "metadata": {
    "total_experiments": 36,
    "successful_experiments": 36
  },
  "overall_metrics": {
    "ga_f1": {
      "mean": 0.771037,
      "std": 0.045678,
      "min": 0.665432,
      "max": 0.854321
    },
    ...
  },
  "timing_metrics": {
    "ga_duration": {
      "mean": 30.45,
      "std": 5.23,
      "total": 1096.20
    },
    ...
  }
}
```

### File 4: Method Comparison CSV
**Filename**: `method_comparison_YYYYMMDD_HHMMSS.csv`

| Method | F1_Mean | F1_Std | PR_AUC_Mean | PR_AUC_Std | Time_Mean_s | Time_Std_s | Time_Total_s |
|--------|---------|--------|-------------|------------|-------------|------------|--------------|
| GA | 0.7710 | 0.0457 | 0.0234 | 0.0123 | 30.45 | 5.23 | 1096.20 |
| THOMPSON | 0.8877 | 0.0234 | 0.4567 | 0.0234 | 14.32 | 2.11 | 515.52 |
| GAN | 0.8123 | 0.0346 | 0.3456 | 0.0234 | 15.67 | 2.34 | 564.12 |
| BORDERLINE | 0.8235 | 0.0457 | 0.4234 | 0.0345 | 6.78 | 1.23 | 244.08 |
| MONTECARLO | 0.8346 | 0.0457 | 0.4456 | 0.0456 | 15.12 | 2.45 | 544.32 |

### File 5: Intermediate Results
**Filename**: `intermediate_results_YYYYMMDD_HHMMSS.json`
- Auto-saved after each experiment
- For crash recovery

## 🚀 How to Use

### Option 1: Run All 36 Datasets (Recommended)
```bash
cd /home/maxoud/local-storage/projects/RAMSeS
./run_testbed.sh
```

**Time**: ~24-30 hours  
**Output**: `myresults/testbed_aggregated/`

### Option 2: Test First (Smart Choice)
```bash
# Test with just 2 datasets
python test_testbed_runner.py

# If successful, run the full testbed
./run_testbed.sh
```

### Option 3: Custom Execution
```bash
# With custom config
python run_full_testbed.py -c path/to/config.yml

# With custom output directory
python run_full_testbed.py -o myresults/my_custom_output
```

### Option 4: Run in Background (Best for Long Runs)
```bash
# Using nohup
nohup ./run_testbed.sh > testbed.log 2>&1 &

# Monitor progress
tail -f testbed.log

# Or using screen
screen -S ramses
./run_testbed.sh
# Press Ctrl+A then D to detach
# screen -r ramses to reattach
```

## 📋 Dataset List (36 total)

### SKAB (16 entities)
```
SKAB/0, SKAB/1, SKAB/2, SKAB/3, SKAB/4, SKAB/5, SKAB/6, SKAB/7,
SKAB/8, SKAB/9, SKAB/10, SKAB/11, SKAB/12, SKAB/13, SKAB/14, SKAB/15
```

### SKAB_valve1 (16 entities)
```
SKAB_valve1/0, SKAB_valve1/1, SKAB_valve1/2, SKAB_valve1/3,
SKAB_valve1/4, SKAB_valve1/5, SKAB_valve1/6, SKAB_valve1/7,
SKAB_valve1/8, SKAB_valve1/9, SKAB_valve1/10, SKAB_valve1/11,
SKAB_valve1/12, SKAB_valve1/13, SKAB_valve1/14, SKAB_valve1/15
```

### SKAB_valve2 (4 entities)
```
SKAB_valve2/0, SKAB_valve2/1, SKAB_valve2/2, SKAB_valve2/3
```

## 💡 What the Script Does

1. **Loop through all 36 datasets**
2. **For each dataset**:
   - Run `python app.py --dataset X --entity Y`
   - Wait for completion (max 1 hour timeout)
   - Extract results from JSON output
   - Save intermediate progress
3. **After all complete**:
   - Calculate mean/std for all F1 scores
   - Calculate mean/std/total for all timings
   - Generate 5 comprehensive reports
   - Show summary in console

## 📈 Expected Results

You'll get averages across all datasets for:

**F1 Scores** (5 methods):
- Genetic Algorithm
- Thompson Sampling
- GAN Robustness
- Borderline Sensitivity
- Monte Carlo

**Computational Overhead** (6 metrics):
- GA duration
- Thompson duration
- GAN duration
- Borderline duration
- Monte Carlo duration
- Total pipeline duration

**All with**: Mean, Standard Deviation, Min, Max, Total

## 🔍 Progress Tracking

The script shows real-time progress:
```
Progress: 5/36 (13.9%)
================================================================================
Running: SKAB/4
================================================================================
✅ SUCCESS: SKAB/4 (Runtime: 1847.32s)
```

Status indicators:
- ✅ SUCCESS: Experiment completed successfully
- ❌ FAILED: Experiment failed (error logged)
- ⏱️ TIMEOUT: Experiment exceeded 1 hour
- ⚠️ WARNING: Results file not found

## 📁 File Locations

```
RAMSeS/
├── run_full_testbed.py          # Main script
├── run_testbed.sh               # Quick launcher
├── test_testbed_runner.py       # Test script
├── TESTBED_RUNNER_GUIDE.md      # Detailed documentation
├── TESTBED_QUICK_REFERENCE.md   # This file
└── myresults/
    └── testbed_aggregated/       # Output directory
        ├── detailed_results_*.csv
        ├── summary_statistics_*.txt
        ├── aggregated_results_*.json
        ├── method_comparison_*.csv
        └── intermediate_results_*.json
```

## ⚡ Quick Commands

```bash
# Test with 2 datasets (~1.5 hours)
python test_testbed_runner.py

# Run full testbed (~24-30 hours)
./run_testbed.sh

# Run in background
nohup ./run_testbed.sh > testbed.log 2>&1 &

# Monitor progress
tail -f testbed.log

# Check intermediate results
cat myresults/testbed_aggregated/intermediate_results_*.json | python -m json.tool
```

## 🎯 Your Requirements Met

✅ **F1 scores for all datasets**: Detailed CSV with all F1 scores  
✅ **Average F1 scores**: Summary statistics with mean ± std  
✅ **Computational overhead for all**: CSV with all module timings  
✅ **Average computational overhead**: Summary with mean ± std  
✅ **Single aggregated file**: JSON with all metrics + summaries  

**You get 5 comprehensive output files with everything you need!**
