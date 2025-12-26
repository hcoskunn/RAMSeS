# Testbed Output Locations

## Overview
When you run the testbed, outputs are saved in **3 main locations**:

---

## 1. 📋 Execution Log (Real-time Progress)

### Location:
```
/home/maxoud/local-storage/projects/RAMSeS/testbed_run_YYYYMMDD_HHMMSS.log
```

### Example:
```
/home/maxoud/local-storage/projects/RAMSeS/testbed_run_20251225_141315.log
```

### Contains:
- ✅ All terminal output (everything you see on screen)
- ✅ Detailed debugging information
- ✅ Progress updates every 30 seconds
- ✅ Model loading messages
- ✅ Generation progress
- ✅ Error messages and warnings
- ✅ Timing information

### Created by:
Line 20 in `run_testbed.sh`:
```bash
2>&1 | tee testbed_run_$(date +%Y%m%d_%H%M%S).log
```

### View in real-time:
```bash
tail -f testbed_run_*.log
```

---

## 2. 📊 Individual Dataset Results (Detailed Metrics)

### Location Pattern:
```
myresults/comprehensive/{domain}/{entity}/comprehensive_results_{domain}_{entity}_iter0.txt
```

### Examples:
```bash
myresults/comprehensive/servermachinedataset/machine-1-1/comprehensive_results_servermachinedataset_machine-1-1_iter0.txt
myresults/comprehensive/servermachinedataset/machine-1-2/comprehensive_results_servermachinedataset_machine-1-2_iter0.txt
myresults/comprehensive/skab/3/comprehensive_results_skab_3_iter0.txt
```

### Directory Structure:
```
myresults/comprehensive/
├── servermachinedataset/
│   ├── machine-1-1/
│   │   └── comprehensive_results_servermachinedataset_machine-1-1_iter0.txt
│   ├── machine-1-2/
│   │   └── comprehensive_results_servermachinedataset_machine-1-2_iter0.txt
│   └── ...
├── skab/
│   ├── 1/
│   │   └── comprehensive_results_skab_1_iter0.txt
│   ├── 2/
│   │   └── comprehensive_results_skab_2_iter0.txt
│   └── ...
```

### Contains:
- ✅ Individual model F1 scores and PR-AUC
- ✅ Genetic Algorithm best ensemble
- ✅ Thompson Sampling results
- ✅ GAN robustness scores
- ✅ Off-by-threshold analysis
- ✅ Monte Carlo simulation results
- ✅ Final decision (single vs ensemble)
- ✅ Detailed timing for each module
- ✅ Model rankings

### Created by:
`app.py` during execution (see `comprehensive_results_writer.py`)

### Check if results exist:
```bash
ls -lh myresults/comprehensive/servermachinedataset/machine-1-1/
```

---

## 3. 📈 Testbed Aggregated Results (Summary Statistics)

### Location:
```
testbed_results/
```

### Full path:
```
/home/maxoud/local-storage/projects/RAMSeS/testbed_results/
```

### Directory Structure:
```
testbed_results/
├── servermachinedataset/
│   ├── intermediate_results.json        # Progress checkpoint
│   ├── domain_report.txt                # Human-readable summary
│   ├── detailed_results.csv             # Spreadsheet-ready data
│   └── aggregate_statistics.json        # Machine-readable stats
├── skab/
│   ├── intermediate_results.json
│   ├── domain_report.txt
│   ├── detailed_results.csv
│   └── aggregate_statistics.json
└── final_summary.txt                    # Overall testbed summary
```

### Files Created:

#### 3.1 `intermediate_results.json`
- **Purpose**: Progress checkpoint (saved after each dataset)
- **Format**: JSON
- **Contains**: All raw results for the domain
- **Use**: Resume if process is interrupted

#### 3.2 `domain_report.txt`
- **Purpose**: Human-readable summary for the domain
- **Format**: Plain text with formatting
- **Contains**:
  ```
  ========================================
  RAMSeS Testbed Report - Domain: servermachinedataset
  Generated: 2025-12-25 15:30:00
  ========================================
  
  OVERALL STATISTICS
  ========================================
  Total Datasets Processed: 2
  Total Computational Time: 5400.50s (90.01 min)
  Average Runtime per Dataset: 2700.25s
  Average Memory Usage: 950.50 MB
  Average Peak Memory: 1234.75 MB
  
  AVERAGE MODULE COMPUTATIONAL OVERHEAD
  ========================================
  Genetic Algorithm         :    2400.5000s
  Thompson Sampling         :     180.2500s
  GAN Robustness           :     450.7500s
  Monte Carlo              :     320.1250s
  ...
  
  AVERAGE F1 SCORES
  ========================================
  Genetic Algorithm        : 0.9234
  Thompson Sampling        : 0.9156
  Final Decision (Selected): 0.9287
  ...
  ```

#### 3.3 `detailed_results.csv`
- **Purpose**: Spreadsheet-ready detailed data
- **Format**: CSV (Excel/pandas compatible)
- **Contains**: One row per dataset with all metrics
- **Columns**:
  ```
  dataset_file, entity, total_runtime, avg_memory_mb, peak_memory_mb,
  ga_f1, ga_pr_auc, thompson_f1, gan_f1, monte_carlo_f1,
  final_selected, final_f1, ...
  ```

#### 3.4 `aggregate_statistics.json`
- **Purpose**: Machine-readable summary statistics
- **Format**: JSON
- **Contains**: 
  ```json
  {
    "total_datasets": 2,
    "total_runtime": 5400.50,
    "avg_runtime": 2700.25,
    "avg_memory_mb": 950.50,
    "avg_module_times": {
      "genetic_algorithm": 2400.50,
      "thompson_sampling": 180.25,
      ...
    },
    "avg_f1_scores": {
      "genetic_algorithm": 0.9234,
      "thompson_sampling": 0.9156,
      ...
    }
  }
  ```

### Created by:
`run_testbed_comprehensive.py` (specified by `--output-dir testbed_results`)

### Check results:
```bash
ls -lh testbed_results/servermachinedataset/
cat testbed_results/servermachinedataset/domain_report.txt
```

---

## Quick Access Commands

### 1. Monitor Progress (Real-time):
```bash
# Watch the log file as it updates
tail -f testbed_run_*.log

# Watch last 50 lines
tail -n 50 -f testbed_run_*.log

# Search for specific events
grep -i "generation\|completed" testbed_run_*.log
```

### 2. Check Individual Results:
```bash
# List all completed datasets
ls -d myresults/comprehensive/*/*/

# View specific result
cat myresults/comprehensive/servermachinedataset/machine-1-1/comprehensive_results_*
```

### 3. View Aggregated Results:
```bash
# View domain summary
cat testbed_results/servermachinedataset/domain_report.txt

# Open CSV in Excel/LibreOffice
libreoffice testbed_results/servermachinedataset/detailed_results.csv

# View JSON stats
cat testbed_results/servermachinedataset/aggregate_statistics.json | python -m json.tool
```

### 4. Check Progress:
```bash
# Count completed datasets
ls myresults/comprehensive/servermachinedataset/*/comprehensive_results_* | wc -l

# Check what's running
ps aux | grep "python app.py"

# Check log file size (growing = still running)
ls -lh testbed_run_*.log
```

---

## Summary Table

| Output Type | Location | Format | When Created | Purpose |
|-------------|----------|--------|--------------|---------|
| Execution Log | `testbed_run_*.log` | Text | Immediately | Real-time progress monitoring |
| Individual Results | `myresults/comprehensive/{domain}/{entity}/` | Text | During execution | Detailed per-dataset metrics |
| Intermediate JSON | `testbed_results/{domain}/intermediate_results.json` | JSON | After each dataset | Progress checkpoint |
| Domain Report | `testbed_results/{domain}/domain_report.txt` | Text | After domain completes | Human-readable summary |
| Detailed CSV | `testbed_results/{domain}/detailed_results.csv` | CSV | After domain completes | Spreadsheet analysis |
| Aggregate Stats | `testbed_results/{domain}/aggregate_statistics.json` | JSON | After domain completes | Machine-readable stats |

---

## Example: After Running test_smd_small.csv

After running:
```bash
./run_testbed.sh testbed/file_list/test_smd_small.csv
```

You'll have:

```
RAMSeS/
├── testbed_run_20251225_141315.log          # Full execution log
├── myresults/
│   └── comprehensive/
│       └── servermachinedataset/
│           ├── machine-1-1/
│           │   └── comprehensive_results_servermachinedataset_machine-1-1_iter0.txt
│           └── machine-1-2/
│               └── comprehensive_results_servermachinedataset_machine-1-2_iter0.txt
└── testbed_results/
    └── servermachinedataset/
        ├── intermediate_results.json
        ├── domain_report.txt
        ├── detailed_results.csv
        └── aggregate_statistics.json
```

**Total outputs: 6 files + 1 log file = 7 files**

---

## Important Notes

1. **Intermediate results are saved progressively** - You don't lose data if the process is interrupted
2. **Individual results exist even if testbed crashes** - Each dataset result is saved immediately
3. **Log file shows everything** - If something goes wrong, check `testbed_run_*.log`
4. **CSV can be imported to Excel/Python** - For custom analysis
5. **JSON files are machine-readable** - For automated processing

---

## Where to Find Your Results RIGHT NOW

Based on your current run (started 14:13):

```bash
# Log file (live updates)
tail -f ~/local-storage/projects/RAMSeS/testbed_run_20251225_*.log

# Individual result (will appear when dataset finishes)
ls ~/local-storage/projects/RAMSeS/myresults/comprehensive/servermachinedataset/machine-1-1/

# Summary (will appear when all datasets in domain finish)
ls ~/local-storage/projects/RAMSeS/testbed_results/servermachinedataset/
```
