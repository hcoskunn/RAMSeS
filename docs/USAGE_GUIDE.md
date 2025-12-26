# Quick Usage Guide for Fixed app.py

## Basic Usage

### 1. Run with default dataset (skab/3)
```bash
python app.py \
  --dataset_path /path/to/Mononito \
  --trained_model_path /path/to/trained_models
```

### 2. Run with specific dataset and entity
```bash
python app.py \
  --dataset smd \
  --entity machine-1-1 \
  --dataset_path /path/to/Mononito \
  --trained_model_path /path/to/trained_models
```

### 3. Run with parallel execution
```bash
python app.py \
  --dataset_path /path/to/Mononito \
  --trained_model_path /path/to/trained_models \
  --parallel
```

### 4. Configure hyperparameters (if supported by get_args_from_cmdline)
```bash
python app.py \
  --dataset_path /path/to/Mononito \
  --trained_model_path /path/to/trained_models \
  --population_size 50 \
  --generations 100 \
  --thompson_iterations 100 \
  --mc_simulations 200 \
  --noise_level 0.15
```

## Configuration Constants

You can edit these at the top of `app.py`:

```python
DEFAULT_POPULATION_SIZE = 20        # GA population size
DEFAULT_GENERATIONS = 50            # GA generations
DEFAULT_THOMPSON_ITERATIONS = 50    # Thompson Sampling iterations
DEFAULT_MC_SIMULATIONS = 100        # Monte Carlo simulations
DEFAULT_NOISE_LEVEL = 0.1          # Gaussian noise std dev
DEFAULT_MUTATION_RATE = 0.1        # GA mutation rate
```

## Model Path Structure

Your trained models should be organized as:
```
{trained_model_path}/
└── {dataset}/
    └── {entity}/
        ├── CBLOF_1.pth
        ├── CBLOF_2.pth
        ├── DGHL_1.pth
        ├── LOF_1.pth
        ├── LSTMVAE_1.pth
        └── ... (all model instances)
```

Example:
```
/path/to/trained_models/
├── skab/
│   └── 3/
│       ├── CBLOF_1.pth
│       └── ...
└── smd/
    └── machine-1-1/
        ├── CBLOF_1.pth
        └── ...
```

## Output Structure

Results are saved to:
```
myresults/
├── robust_aggregated/{dataset}/{entity}/
│   ├── robust_aggregated_results_{dataset}_{entity}_{iteration}.txt
│   └── new_robust_aggregated_results_{dataset}_{entity}_{iteration}.txt
└── GA_Ens/{dataset}/{entity}/
    └── ensemble_scores_{dataset}_{entity}_Data_vs_anomalies_[...].png
```

## Algorithm List

The following algorithms are configured by default:

**Base algorithms**:
- CBLOF (4 instances)
- DGHL (4 instances)
- LOF (4 instances)
- LSTMVAE (4 instances)
- MD (1 instance)
- NN (3 instances)
- RNN (4 instances)
- RM (3 instances)

**Total: 27 model instances**

## Python API

You can also call the function directly in Python:

```python
from app import run_app, algorithm_list, algorithm_list_instances

# Run with defaults
run_app(algorithm_list, algorithm_list_instances)

# Run with specific dataset/entity
run_app(algorithm_list, algorithm_list_instances, 
        dataset='smd', entity='machine-1-1')

# Run with parallel execution
run_app(algorithm_list, algorithm_list_instances, 
        dataset='skab', entity='3', use_parallel=True)
```

## Troubleshooting

### Error: "Model directory does not exist"
- Check that `--trained_model_path` points to correct directory
- Ensure path structure: `{trained_model_path}/{dataset}/{entity}/`

### Error: "Failed to load models"
- Ensure all 27 model .pth files exist in the directory
- Check file names match exactly (e.g., `CBLOF_1.pth`, not `cblof_1.pth`)

### Error: "Failed to load training/test data"
- Check `--dataset_path` points to correct Mononito directory
- Verify dataset name and entity name are correct

### Window size errors
- Check `min_length` parameter
- Ensure dataset has sufficient data points
- Verify `iterations` parameter is positive

## What Changed from Original

1. ✅ No more hardcoded paths
2. ✅ No more global variables
3. ✅ No duplicate algorithm instances
4. ✅ Configurable hyperparameters
5. ✅ Better error messages
6. ✅ Support for different datasets/entities
7. ✅ Consistent rank aggregation ordering
8. ✅ Safe window size calculations
9. ✅ Optional parallel execution
10. ✅ Improved logging throughout

## Migration from Old Version

If you have scripts using the old version:

**Before**:
```python
trained_models = {}
run_app(algorithm_list, algorithm_list_instances)
```

**After**:
```python
# No global needed
run_app(algorithm_list, algorithm_list_instances, 
        dataset='skab', entity='3')
```

## Performance Tips

1. **Parallel execution**: Use `--parallel` for faster execution on multi-core systems
2. **Reduce simulations**: Lower `mc_simulations` for faster testing (default: 100)
3. **Reduce GA parameters**: Use smaller `population_size` and `generations` for testing
4. **GPU acceleration**: Ensure CUDA is available for deep learning models (DGHL, LSTMVAE, RNN)

## Next Steps

After fixing bugs, consider:
1. Adding unit tests
2. Creating configuration files (YAML/JSON)
3. Adding progress bars for long operations
4. Implementing checkpoint/resume functionality
5. Supporting batch processing of multiple entities
