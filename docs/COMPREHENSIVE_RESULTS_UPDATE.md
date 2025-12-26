# Comprehensive Results Format Update

## Summary of Changes

The comprehensive results output has been updated to include chosen models and metrics (F1 and PR-AUC) for each module in the RAMSeS framework.

## Updated Output Format

### 1. Genetic Algorithm - Ensemble Selection
```
Chosen Model: [Ensemble models]
  F1 Score    : X.XXXXXX
  PR-AUC      : X.XXXXXX
```

### 2. Thompson Sampling - Online Model Selection
```
Chosen Model: MODEL_NAME
  F1 Score    : X.XXXXXX
  PR-AUC      : X.XXXXXX

Top-5 Models (Ranked):
  1. MODEL_1
  2. MODEL_2
  ...
```

### 3. GAN Robustness Test
```
Chosen Model: MODEL_NAME
  F1 Score    : X.XXXXXX
  PR-AUC      : X.XXXXXX

  Top-5 by F1:
    1. MODEL_1
    ...
```

### 4. Borderline Sensitivity Test
```
Chosen Model: MODEL_NAME
  F1 Score    : X.XXXXXX
  PR-AUC      : X.XXXXXX

  Top-5 by F1:
    1. MODEL_1
    ...
```

### 5. Monte Carlo Simulation
```
Chosen Model (F1)   : MODEL_NAME_F1
Chosen Model (PR-AUC): MODEL_NAME_PR
  Best F1 Score      : X.XXXXXX
  Best PR-AUC Score  : X.XXXXXX

  Top-5 by F1:
    1. MODEL_1
    ...
```

### 6. Framework Final Decision

The final decision section now shows both options and makes a clear choice based on F1 scores:

```
Single Model Option:
--------------------------------------------------
  Model       : MODEL_NAME
  F1 Score    : X.XXXXXX
  PR-AUC      : X.XXXXXX

Ensemble Option:
--------------------------------------------------
  Models      : [Ensemble models]
  Size        : N
  Meta-Model  : rf/lr/...
  F1 Score    : X.XXXXXX
  PR-AUC      : X.XXXXXX
  Fitness     : X.XXXXXX

Final Choice:
--------------------------------------------------
  ✓ ENSEMBLE SELECTED  (or SINGLE MODEL SELECTED)
    Reason: Ensemble F1 (X.XXXXXX) >= Single Model F1 (X.XXXXXX)
    Improvement: +X.XXXXXX (XX.XX%)
```

## Technical Changes Made

### 1. Updated `write_comprehensive_results()` in `app.py`

**Thompson Sampling Section:**
- Added "Chosen Model" with F1 and PR-AUC metrics
- Moved "Best Model" to "Chosen Model" at the top
- Shows metrics before the Top-5 list

**Robustness Tests (GAN, Borderline, Monte Carlo):**
- Added "Chosen Model" with F1 and PR-AUC metrics for each test
- Shows the best performing model with its scores
- Maintains the Top-5 rankings below

**Final Decision Section:**
- Restructured to show both Single Model and Ensemble options clearly
- Added explicit "Final Choice" section
- Shows comparison and improvement/advantage percentage
- Makes the decision based on F1 score comparison (>= for ensemble)

### 2. Updated Model Evaluation in `app.py`

**Added evaluations for:**
- Thompson Sampling best model (evaluates and captures F1 and PR-AUC)
- Monte Carlo best models (evaluates both F1 and PR-AUC best models)
- Final aggregated single model (gets actual F1 and PR-AUC scores)

**Updated results_dict population:**
- Added `f1` and `pr_auc` fields to Thompson Sampling results
- Added `best_f1` and `best_pr_auc` to Monte Carlo results
- Updated final_decision to use actual evaluated metrics instead of proxies

### 3. Decision Logic

The framework now:
1. Evaluates all chosen models from each module
2. Captures their F1 and PR-AUC scores
3. Compares ensemble F1 vs single model F1
4. Selects ensemble if F1 >= single model F1
5. Shows clear reasoning in the output

## Usage

Run your RAMSeS pipeline as usual:
```bash
python app.py -c Configs/your_config.yml
```

The comprehensive results will be generated with the new format in:
```
myresults/comprehensive/{dataset}/{entity}/comprehensive_results_{dataset}_{entity}_iter0.txt
```

## Benefits

1. **Complete Visibility**: Each module's best model and performance metrics are clearly shown
2. **Easy Comparison**: All models' F1 and PR-AUC scores are available for comparison
3. **Clear Decision**: The final choice shows explicit reasoning and quantified improvement
4. **Better Documentation**: Results files are self-contained and easy to understand
