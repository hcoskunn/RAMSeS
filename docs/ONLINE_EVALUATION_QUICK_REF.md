# Quick Reference: Using Online Evaluation

## Module: `online_evaluation.py`

### Import Functions

```python
from online_evaluation import (
    visualize_injected_anomalies,
    setup_sliding_windows,
    update_window_data,
    compute_misclassifications,
    run_online_evaluation,
    run_single_shot_evaluation,
)
```

---

## Function Reference

### 1. visualize_injected_anomalies()

**Purpose**: Create plots showing injected anomalies vs original data

**Usage**:
```python
plot_path = visualize_injected_anomalies(
    test_data=test_data,               # Dataset with anomalies
    test_data_before=test_data_orig,   # Original dataset
    anomaly_sizes=anomaly_magnitudes,   # Anomaly magnitude array
    dataset='smd',                      # Dataset name
    entity='machine-1-1',               # Entity name
    anomaly_list=['spikes']             # Types of anomalies injected
)
# Returns: Path to saved plot
```

**Output**: PNG file saved to `myresults/GA_Ens/{dataset}/{entity}/`

---

### 2. setup_sliding_windows()

**Purpose**: Configure sliding windows for online evaluation

**Usage**:
```python
data_windows, targets_windows, mask_windows, num_windows, window_size, stride = setup_sliding_windows(
    test_data_before=original_test_data,  # Original test data
    iterations=10,                        # Number of windows
    min_length=256                        # Minimum window size
)
```

**Returns**:
- `data_windows`: List of data windows
- `targets_windows`: List of target/label windows
- `mask_windows`: List of mask windows
- `num_windows`: Total number of windows created
- `window_size`: Calculated window size
- `stride`: Calculated stride/step size

---

### 3. update_window_data()

**Purpose**: Update dataset with specific window data

**Usage**:
```python
update_window_data(
    test_data=test_dataset,          # Dataset to update (modified in-place)
    data_windows=data_windows,        # List of data windows
    targets_windows=targets_windows,  # List of target windows
    new_mask=mask_windows,            # List of mask windows
    window_idx=5                      # Which window to use (0-based)
)
```

**Note**: Modifies `test_data` in place

---

### 4. compute_misclassifications()

**Purpose**: Calculate and save misclassification counts

**Usage**:
```python
compute_misclassifications(
    adjusted_y_pred_ind_current=predictions,  # Current model predictions
    test_data_copy=test_data,                 # Test data copy
    dataset='smd',                            # Dataset name
    entity='machine-1-1',                     # Entity name
    values=fitness_values,                    # Fitness function output
    full_aggregated='LSTMVAE_1',             # Current best model name
    best_ensemble=['LOF_1', 'RNN_2'],        # Current best ensemble
    iteration=3                               # Iteration number
)
```

**Output**: Text file saved to `myresults/robust_aggregated/{dataset}/{entity}/`

---

### 5. run_online_evaluation() ⭐

**Purpose**: Run complete online evaluation with sliding windows

**Usage**:
```python
results = run_online_evaluation(
    train_data=train_data,
    test_data=test_data,
    test_data_before=original_test_data,
    dataset='smd',
    entity='machine-1-1',
    trained_models=model_dict,
    algorithm_list_instances=model_names,
    selection_func=run_model_selection_algorithms_1,
    iterations=10,
    anomaly_list=['spikes', 'dips'],
    args=cmd_args,
    initial_results=None,  # Optional: provide first window results
    min_length=256
)

# Unpack results
(best_thompson, robust_agg, full_aggregated, best_ensemble,
 individual_predictions, base_predictions_train, base_predictions_test,
 y_true_train, y_true_test, meta_model_type) = results
```

**Parameters**:
- `iterations`: Number of sliding windows to process
- `selection_func`: Sequential or parallel model selection function
- `initial_results`: Optional pre-computed results for first window
- All other parameters: dataset, models, arguments, etc.

**Returns**: 10-item tuple with final iteration results

---

### 6. run_single_shot_evaluation()

**Purpose**: Run evaluation on full test set without sliding windows

**Usage**:
```python
results = run_single_shot_evaluation(
    train_data=train_data,
    test_data=test_data,
    dataset='smd',
    entity='machine-1-1',
    trained_models=model_dict,
    algorithm_list_instances=model_names,
    selection_func=run_model_selection_algorithms_1,
    anomaly_list=['spikes'],
    args=cmd_args
)

# Same 10-item tuple as run_online_evaluation
```

**Use When**: 
- `iterations=1` (default)
- Don't need sliding windows
- Want to process entire test set at once

---

## Command-Line Usage

### Single-Shot (Default)
```bash
python app.py \
  --dataset_path /path/to/data \
  --trained_model_path /path/to/models \
  --dataset smd \
  --entity machine-1-1
```

### Online Evaluation (10 Windows)
```bash
python app.py \
  --dataset_path /path/to/data \
  --trained_model_path /path/to/models \
  --dataset smd \
  --entity machine-1-1 \
  --iterations 10
```

### With Parallel Execution
```bash
python app.py \
  --dataset_path /path/to/data \
  --trained_model_path /path/to/models \
  --iterations 10 \
  --parallel
```

---

## Example: Custom Online Evaluation Script

```python
#!/usr/bin/env python
"""Custom online evaluation script"""

import sys
sys.path.append('/path/to/RAMSeS')

from Datasets.load import load_data
from app import (
    load_trained_models,
    run_model_selection_algorithms_1,
    algorithm_list_instances
)
from online_evaluation import run_online_evaluation
from Model_Selection.inject_anomalies import Inject
import copy

# Load data
train_data = load_data(dataset='smd', group='train', entities='machine-1-1',
                       downsampling=10, root_dir='/path/to/data')
test_data = load_data(dataset='smd', group='test', entities='machine-1-1',
                      downsampling=10, root_dir='/path/to/data')

# Load models
trained_models = load_trained_models(
    algorithm_list_instances,
    '/path/to/models/smd/machine-1-1'
)

# Inject anomalies
test_data_before = copy.deepcopy(test_data)
train_data, _ = Inject(train_data, ['spikes'])
test_data, _ = Inject(test_data, ['spikes'])

# Configure
args = {
    'population_size': 20,
    'generations': 50,
    'thompson_iterations': 50,
    'mc_simulations': 100,
    'noise_level': 0.1,
    'meta_model_type': 'rf',
    'mutation_rate': 0.1
}

# Run online evaluation
results = run_online_evaluation(
    train_data=train_data,
    test_data=test_data,
    test_data_before=test_data_before,
    dataset='smd',
    entity='machine-1-1',
    trained_models=trained_models,
    algorithm_list_instances=algorithm_list_instances,
    selection_func=run_model_selection_algorithms_1,
    iterations=5,
    anomaly_list=['spikes'],
    args=args,
    min_length=256
)

print(f"Best model: {results[0]}")
print(f"Best ensemble: {results[3]}")
```

---

## Output Files

### Visualizations
- **Location**: `myresults/GA_Ens/{dataset}/{entity}/`
- **Files**: `ensemble_scores_{dataset}_{entity}_Data_vs_anomalies_[...].png`
- **Content**: Plots showing original data, injected anomalies, and labels

### Misclassification Reports
- **Location**: `myresults/robust_aggregated/{dataset}/{entity}/`
- **Files**: `new_robust_aggregated_results_{dataset}_{entity}_{iteration}.txt`
- **Content**: 
  - Iteration number
  - Current best model
  - Misclassifications by single model
  - Current best ensemble
  - Misclassifications by ensemble

### Model Selection Results
- **Location**: `myresults/robust_aggregated/{dataset}/{entity}/`
- **Files**: `robust_aggregated_results_{dataset}_{entity}_{iteration}.txt`
- **Content**:
  - GAN robustness rankings
  - Borderline sensitivity rankings
  - Monte Carlo rankings
  - Robust aggregate rankings
  - Final aggregate with Thompson Sampling

---

## Tips & Best Practices

### 1. Window Configuration
- **Small datasets**: Use fewer iterations (2-5)
- **Large datasets**: Can use 10-20 iterations
- **Window size**: Automatically calculated as `total_size / iterations`
- **Minimum**: Respects `min_length` parameter (default: 256)

### 2. Performance
- **Sequential mode**: Better for debugging, easier to follow
- **Parallel mode**: Faster but uses more resources
- **Iterations**: More iterations = longer runtime but better adaptation

### 3. Memory Management
- Uses `copy.deepcopy()` extensively
- Large datasets with many iterations can use significant memory
- Consider reducing model instances if memory is limited

### 4. Anomaly Injection
- Supports multiple anomaly types: `['spikes', 'dips', 'noise', ...]`
- Injected before each window evaluation
- Same anomaly pattern for all windows (reproducible)

### 5. Results Tracking
- Each iteration produces separate output files
- Files numbered by iteration: `..._0.txt`, `..._1.txt`, etc.
- Compare across iterations to see adaptation

---

## Troubleshooting

### "No windows created"
- Check `iterations` parameter is > 0
- Verify test data has sufficient length
- Ensure `min_length` isn't larger than total data size

### "Model not found in trained_models"
- Verify all model instances exist in trained_models dict
- Check that best_ensemble models are loaded
- Ensure model names match exactly (case-sensitive)

### Memory errors
- Reduce number of iterations
- Use fewer model instances
- Process smaller dataset chunks
- Consider using single-shot evaluation instead

### Visualization not saved
- Check write permissions for output directory
- Verify matplotlib backend is configured
- Check disk space

---

## See Also

- **PROJECT_OVERVIEW.md** - Complete project documentation
- **BUGFIXES_APPLIED.md** - All bug fixes and improvements
- **USAGE_GUIDE.md** - General usage guide
- **ONLINE_EVALUATION_REFACTORING.md** - Detailed refactoring notes

---

**Quick command to run online evaluation with 5 windows:**
```bash
python app.py --iterations 5 --dataset_path /path/to/data --trained_model_path /path/to/models
```
