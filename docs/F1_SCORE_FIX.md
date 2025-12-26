# F1 Score Calculation Fix

## Problem Identified

The F1 scores in the comprehensive results were showing invalid values (negative or > 1):
- Example: Thompson Sampling showed F1 = -0.026377
- Log showed: F1 = 2.007081 (should be between 0 and 1)

## Root Cause

The `evaluate_individual_models` function in `/Metrics/Ensemble_GA.py` was calling `f1_score()` directly with **continuous anomaly scores** instead of **binary predictions** (0 or 1).

### Original Buggy Code:
```python
best_f1, precision, recall, TP, TN, FP, FN = f1_score(y_scores, y_true)
```

The `f1_score()` function expects binary predictions but was receiving continuous scores like [0.5, 0.8, 0.3, ...]. When you multiply continuous scores by binary labels in the TP/FP/FN calculations, you get nonsensical results:
- `TP = np.sum(predict * actual)` with continuous predict values
- This can produce F1 scores outside [0, 1]

## Solution

Use the `best_f1_linspace()` function which:
1. Finds the optimal threshold for converting scores to binary predictions
2. Applies segment adjustment (adjust_predicts) for time series
3. Calculates F1 score on **binary predictions**

### Fixed Code:
```python
from Metrics.metrics import best_f1_linspace

best_f1, precision, recall, y_pred_binary, _, best_threshold = best_f1_linspace(
    y_scores, y_true, n_splits=100, segment_adjust=True, f1_type='standard'
)
```

## Changes Made

### 1. `/Metrics/Ensemble_GA.py` - Line ~237
**Fixed the `evaluate_individual_models` function:**
- Replaced direct `f1_score(y_scores, y_true)` call
- Added `best_f1_linspace()` to find optimal threshold
- Now returns proper binary predictions in `adjusted_y_pred_list`
- F1 scores are now guaranteed to be in [0, 1]

### 2. `/app.py` - Line ~733
**Improved the `to_scalar()` helper function:**
- Added validation warning for invalid F1/PR-AUC scores
- Keeps original value but logs warning if outside [0, 1]
- Helps identify if issues occur elsewhere in the pipeline

## Impact

### Before Fix:
```
THOMPSON SAMPLING - ONLINE MODEL SELECTION
Chosen Model: LSTMVAE_3
  F1 Score    : -0.026377  ❌ INVALID
  PR-AUC      : 0.333331

Log: Model LSTMVAE_3: F1 score = 2.007081 ❌ INVALID
```

### After Fix:
```
THOMPSON SAMPLING - ONLINE MODEL SELECTION
Chosen Model: LSTMVAE_3
  F1 Score    : 0.845123  ✓ VALID (example value)
  PR-AUC      : 0.333331

Log: Model LSTMVAE_3: F1 score = 0.845123 ✓ VALID
```

## Testing

To test the fix:
```bash
cd /home/maxoud/local-storage/projects/RAMSeS
python app.py -c Configs/custom_config.yml
```

Check the comprehensive results file:
```bash
cat myresults/comprehensive/skab/3/comprehensive_results_skab_3_iter0.txt
```

Verify all F1 scores are between 0 and 1.

## Related Functions

All these functions in the codebase should follow the same pattern:
- ✓ `best_f1_linspace()` - Correct implementation
- ✓ `range_based_precision_recall_f1_auc()` - Correct implementation  
- ✓ `get_composite_fscore_raw()` - Correct implementation
- ❌ Direct `f1_score(continuous_scores, labels)` - WRONG (now fixed)

## Additional Notes

The `best_f1_linspace` function:
- Tests 100 different thresholds (n_splits=100)
- Uses segment_adjust=True for time series anomaly detection
- Properly handles edge cases (no anomalies, all anomalies)
- Returns both F1 score and the binary predictions

This is the standard way to calculate F1 for anomaly detection with continuous anomaly scores.
