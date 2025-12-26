# Archived Results - SKAB Dataset

## Archive Date
December 26, 2025 00:27 UTC

## Reason for Archival
These results were computed using an **incorrect train/test split method** and are archived for reference only.

## The Issue
The SKAB dataset loading code used `sklearn.model_selection.train_test_split()` with random shuffling, which:
- ❌ **Broke temporal continuity** of time series data
- ❌ **Mixed past and future data** in training and testing sets
- ❌ **Created data leakage** through test set normalization
- ❌ **Made evaluation unrealistic** (training on random samples from entire time series)

## What Was Fixed
Changed to **sequential split** (consistent with SMD and Anomaly-Archive datasets):
- ✅ First 80% of time series → Training set
- ✅ Last 20% of time series → Testing set
- ✅ Maintains temporal order
- ✅ Realistic evaluation (predict future from past)
- ✅ Proper normalization (fit on train, transform test)

## Impact
Results in these archived directories are **overly optimistic** due to:
1. **Data leakage**: Model saw future information during training
2. **Unrealistic evaluation**: Random split doesn't reflect real-world deployment

New results computed after this fix will be:
- More challenging (lower F1 scores expected)
- More realistic and reliable
- Properly comparable to other datasets

## Archived Directories

### Main Results
- `skab_random_split_YYYYMMDD_HHMMSS/` - Comprehensive results

### Full Backup
- `skab_full_backup_YYYYMMDD_HHMMSS/` - All SKAB results including:
  - Thompson Sampling results
  - Robust aggregation results
  - GA ensemble results
  - Robustness testing (GAN, Monte Carlo, Off-by-threshold)
  - Testbed aggregated results

## Code Change
**File**: `Datasets/load.py`
**Function**: `load_csv_file()`
**Commit**: 2025-12-26

### Before (WRONG):
```python
X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)  # Random shuffle!
```

### After (CORRECT):
```python
train_end = int(n_timestamps * 0.8)
X_train = X[:, :train_end]  # First 80% sequential
X_test = X[:, train_end:]    # Last 20% sequential
```

---
**Do not use these archived results for publication or comparison!**
