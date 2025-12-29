# Hyperparameter Grid Simplification

## Summary

**Changed all hyperparameter grids to train ONLY ONE version of each model** instead of multiple versions with different hyperparameters.

## Changes Made

### Before:
- Multiple hyperparameter combinations per model
- Example: LSTMVAE had 4 versions (2 × 2 combinations)
- Total: Many models being trained unnecessarily

### After:
- **ONE** hyperparameter configuration per model
- **ONE** version of each model (_1 only)
- Faster training, less disk space, simpler model selection

## Detailed Changes

| Model | Parameter | Before | After | Versions Reduced |
|-------|-----------|--------|-------|------------------|
| **LSTMVAE** | `hidden_size` | [512, 256] | [256] | 4 → 1 |
| | `latent_size` | [256, 128] | [128] | |
| **NN** | `n_neighbors` | [1, 3, 5] | [3] | 3 → 1 |
| **DGHL** | `z_size` | [25, 50] | [50] | 4 → 1 |
| | `z_iters` | [25, 100] | [100] | |
| **RNN** | `input_size` | [32, 64] | [64] | 4 → 1 |
| | `state_hsize` | [128, 256] | [256] | |
| **LOF** | `contamination` | [0.1, 0.15, 0.2, 0.25] | [0.1] | 4 → 1 |
| **KDE** | `contamination` | [0.1, 0.15, 0.2, 0.25] | [0.1] | 4 → 1 |
| **ABOD** | `contamination` | [0.1, 0.15, 0.2, 0.25] | [0.1] | 4 → 1 |
| **SOS** | `contamination` | [0.1, 0.15, 0.2, 0.25] | [0.1] | 4 → 1 |
| **ALAD** | `contamination` | [0.1, 0.15, 0.2, 0.25] | [0.1] | 4 → 1 |
| **PYOD** | `contamination` | [0.1, 0.15, 0.2, 0.25] | [0.1] | 4 → 1 |
| **CBLOF** | `contamination` | [0.1, 0.15, 0.2, 0.25] | [0.1] | 4 → 1 |
| **COF** | `contamination` | [0.1, 0.15, 0.2, 0.25] | [0.1] | 4 → 1 |
| **RM** | `running_window_size` | [4, 16, 64] | [16] | 3 → 1 |
| **MD** | *(already had single values)* | - | - | 1 (no change) |

## Impact

### Training Time
- **Before**: Training 4 versions of LSTMVAE, 3 versions of NN, etc.
- **After**: Training ONLY 1 version of each model
- **Speedup**: ~3-4x faster training per model

### Disk Space
- **Before**: Multiple .pth, .meta, .png files per model type
  - Example: `LSTMVAE_1.pth`, `LSTMVAE_2.pth`, `LSTMVAE_3.pth`, `LSTMVAE_4.pth`
- **After**: Single file per model type
  - Example: Only `LSTMVAE_1.pth`
- **Savings**: ~70-75% disk space reduction

### Model Selection
- **Before**: Model selection algorithms had to evaluate many versions
  - `algorithm_list_instances` used only `_1` versions anyway!
- **After**: Cleaner, only the models actually used are trained
- **Consistency**: No confusion about which version to use

## Chosen Hyperparameters (Rationale)

| Model | Parameter | Value Chosen | Reason |
|-------|-----------|--------------|--------|
| LSTMVAE | `hidden_size` | 256 | Good balance (not too large, not too small) |
| | `latent_size` | 128 | Reasonable compression ratio |
| NN | `n_neighbors` | 3 | Standard choice for KNN |
| DGHL | `z_size` | 50 | Higher value (better representation) |
| | `z_iters` | 100 | More iterations (better convergence) |
| RNN | `input_size` | 64 | Larger input capacity |
| | `state_hsize` | 256 | Larger hidden state |
| ALL outlier models | `contamination` | 0.1 | Conservative estimate (10% anomalies) |
| RM | `running_window_size` | 16 | Middle ground |

## How to Retrain

If you want to retrain models with the new hyperparameters:

```bash
# Delete existing models
rm -rf Mononito/trained_models/servermachinedataset/machine-1-1/*.pth
rm -rf Mononito/trained_models/servermachinedataset/machine-1-1/*.meta
rm -rf Mononito/trained_models/servermachinedataset/machine-1-1/*.png

# Run training (will create only _1 versions)
python app.py --dataset smd --entity machine-1-1
```

## File Modified

- `/home/maxoud/local-storage/projects/RAMSeS/Model_Training/hyperparameter_grids.py`

All parameter grids now contain single values only, ensuring ONE model version per architecture.

## Next Steps

1. **Clean up existing trained models** (optional - remove _2, _3, _4 versions)
2. **Retrain if needed** (or keep using existing _1 versions)
3. **Monitor training time** - should be much faster!
4. **Check disk usage** - should be much lower!

## Notes

- The `algorithm_list_instances` in `app.py` was already set to use only `_1` versions
- This change aligns the training with what's actually used
- No functionality is lost - we're just not wasting resources on unused models
