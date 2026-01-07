# Analysis Scripts - README

## 📂 Available Scripts

This directory contains optimized shell scripts for running online phase analysis on three different datasets.

### 🎯 Individual Dataset Scripts

| Script | Dataset | Runtime | Dimensions | Best For |
|--------|---------|---------|------------|----------|
| `run_skab_all_analyses.sh` | SKAB | 70 min | 11D multivariate | ⭐ Quick validation |
| `run_ucr_all_analyses.sh` | UCR | 75 min | 1D univariate | Generalization |
| `run_smd_all_analyses.sh` | SMD | 3-4 hrs | 39D multivariate | Scalability |

### 🚀 Master Script

| Script | Description | Runtime |
|--------|-------------|---------|
| `run_all_datasets.sh` | Runs all three datasets sequentially | 5-6 hours |

---

## ⚡ Quick Start

### Run One Dataset (Recommended: Start with SKAB)

```bash
cd /home/maxoud/local-storage/projects/RAMSeS
source /raid0_ssd1/anaconda3/etc/profile.d/conda.sh
conda activate RAMS

# Run SKAB (70 minutes)
./shells/run_skab_all_analyses.sh
```

### Run All Datasets (Recommended: Run overnight in screen)

```bash
cd /home/maxoud/local-storage/projects/RAMSeS

# Start screen session
screen -S complete_analysis

# Activate environment
source /raid0_ssd1/anaconda3/etc/profile.d/conda.sh
conda activate RAMS

# Run master script
./shells/run_all_datasets.sh

# Detach from screen: Ctrl+A, then D
# Reattach later: screen -r complete_analysis
```

---

## 📊 What Each Script Does

All scripts run **THREE analyses** per dataset:

1. **Adaptive Analysis** (R1.O2, R1.O3)
   - Tests update intervals: [5, 10, 20, None]
   - Injects regime shifts at 25%, 50%, 75%
   - Measures F1 recovery
   - **Output:** `results_*/adaptive/`

2. **Scalability Analysis** (R1.O3)
   - Tests pool sizes: [3, 5, 8] models
   - Measures F1 vs pool size
   - Shows diminishing returns
   - **Output:** `results_*/scalability/`

3. **Window Size Sensitivity** (R2.O5)
   - Tests window sizes: [128, 256, 512, 1024]
   - Measures F1 vs latency
   - Finds optimal window size
   - **Output:** `results_*/window_size/`

---

## 📂 Output Structure

```
results_skab/
├── adaptive/adaptive_analysis/SKAB/*/
│   ├── adaptive_summary.txt              ← Human-readable results
│   ├── adaptive_f1_over_time.png         ← Main plot for paper
│   └── adaptive_branch_comparison.png
│
├── scalability/scalability_analysis/SKAB/*/
│   ├── scalability_summary.txt
│   └── scalability_plot.png              ← F1 vs pool size
│
└── window_size/window_size_analysis/SKAB/*/
    ├── window_size_summary.txt
    └── window_size_plot.png

# Same structure for results_ucr/ and results_smd/
```

---

## ⏱️ Time Estimates

### By Dataset

| Dataset | Adaptive | Scalability | Window Size | **Total** |
|---------|----------|-------------|-------------|-----------|
| **SKAB** | 20 min | 25 min | 25 min | **70 min** |
| **UCR** | 25 min | 25 min | 25 min | **75 min** |
| **SMD** | 60-90 min | 60-90 min | 60-90 min | **3-4 hrs** |

### All Three Datasets

**Total:** 5-6 hours (via `run_all_datasets.sh`)

---

## 🔧 Configuration

Each script uses dataset-specific configurations optimized for:
- Number of windows (adjusted for dataset size)
- Number of entities (subset for manageable runtime)
- Update intervals (adapted to data characteristics)

**Dataset lists:** `../dataset_lists/`
- `skab_full.csv` - 16 SKAB entities
- `smd_sample.csv` - 8 SMD machines  
- `ucr_sample.csv` - 5 UCR time series

---

## 📝 View Results

```bash
# View all summaries
find ../results_* -name '*_summary.txt' -exec cat {} \;

# List all plots
find ../results_* -name '*.png' -type f

# Compare adaptive analysis across datasets
cat ../results_skab/adaptive/adaptive_analysis/*/*/adaptive_summary.txt
cat ../results_ucr/adaptive/adaptive_analysis/*/*/adaptive_summary.txt
cat ../results_smd/adaptive/adaptive_analysis/*/*/adaptive_summary.txt
```

---

## 🎯 Recommended Workflow

### Day 1: Quick Validation
```bash
./shells/run_skab_all_analyses.sh  # 70 minutes
```

### Night 1: Complete Evidence
```bash
screen -S complete
./shells/run_all_datasets.sh  # 5-6 hours
# Ctrl+A, D to detach
```

### Day 2: Extract Results
```bash
screen -r complete  # Reattach
find ../results_* -name '*.png'  # View plots
```

---

## 📖 Documentation

- **`../docs/DATASET_GUIDE.md`** - Complete detailed guide
- **`../docs/QUICK_START_DATASETS.md`** - Quick reference
- **`../docs/ORGANIZATION_SUMMARY.md`** - Organization summary

---

## ✅ Summary

- ✅ 4 executable scripts ready to run
- ✅ Each runs 3 analyses per dataset
- ✅ Optimized for dataset characteristics
- ✅ All results saved in `results_*/`

**Start here:** `./shells/run_skab_all_analyses.sh` (70 min)
