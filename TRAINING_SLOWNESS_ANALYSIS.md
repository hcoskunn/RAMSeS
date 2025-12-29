# Training Slowness Investigation

## Your Concern

You mentioned that **training itself** is taking too much time, even though:
- ✅ Running in parallel mode
- ✅ Training only ONE model instance (not 19)
- ✅ Using the SMALLEST model (hidden=256, latent=128)

## Current Training Configuration

### Hyperparameters (from hyperparameter_grids.py)

**LSTMVAE** (your main model):
- `hidden_size`: 256 (smallest)
- `latent_size`: 128 (smallest)  
- `num_layers`: 4
- `window_size`: 64
- `window_step`: 64
- `max_steps`: **100** ⚠️
- `train_batch_size`: 256
- `learning_rate`: 0.0005

**Other models** (NN, DGHL, RNN, etc.):
- All using `max_steps`: **100**
- All using single hyperparameter values

## What Does max_steps=100 Mean?

`max_steps` = number of gradient descent updates during training

**Current**: 100 steps
**Commented out (original)**: 1000-5000 steps

### Training Time Estimate

For LSTMVAE with 100 steps:
- Per step: ~0.5-2 seconds (depends on data size, GPU/CPU)
- **Total: 50-200 seconds (~1-3 minutes)**

This is actually FAST for deep learning!

## But WHY Does It Feel Slow?

### Possibility 1: You're Training MULTIPLE Models ⚠️

Even though each model has only 1 hyperparameter config, you still train multiple model TYPES:

From `algorithm_list` in app.py:
```python
algorithm_list = ['DGHL', 'LSTMVAE', 'NN', 'RNN', 'MD', 'RM', 'LOF', 'CBLOF']
```

**That's 8 different models!**

Training time per entity:
- LSTMVAE: ~2-5 minutes
- RNN: ~2-5 minutes
- DGHL: ~2-5 minutes
- MD: ~2-5 minutes
- NN: ~30 seconds
- RM: ~10 seconds
- LOF: ~30 seconds
- CBLOF: ~30 seconds

**Total: 10-25 minutes per entity**

### Possibility 2: Large Dataset Size

If machine-1-1 has a LOT of data (e.g., 500,000 time points), then:
- Creating training batches takes time
- Each forward/backward pass processes more data
- Data loading becomes a bottleneck

### Possibility 3: CPU vs GPU

Check if models are training on CPU or GPU:

```python
# In hyperparameter_grids.py
'device': [None]  # This means CPU!
```

**⚠️ Training deep models (LSTMVAE, RNN, DGHL, MD) on CPU is 10-50x slower than GPU!**

### Possibility 4: No Model Caching

Looking at train.py lines 755-758:

```python
if not self.overwrite:
    if self.logging_obj.check_file_exists(...):
        print(f'Model LSTMVAE_{MODEL_ID+1} already trained!')
        continue
```

This checks if model EXISTS. But you cleared the cache, so:
- Old models may have been invalidated
- System is retraining from scratch
- Expected if you just cleared cache!

## Performance Comparison

### OLD Sequential (19 models total)
```
Training: 19 models × 2-5 min = 38-95 minutes
```

### NEW Simplified (8 models, 1 version each)
```
Training: 8 models × 2-5 min = 16-40 minutes
```

**Expected speedup: ~2.4x faster** ✅

But you're saying it's SLOWER? Let me check...

## Actual Timeline Check

```bash
# Two processes running machine-1-1:
PID     ELAPSED    
336023  01:28:44   # 88 minutes
1102331 00:58:55   # 59 minutes
```

**Process 1102331 has been running for 59 minutes**

If it's training 8 models × ~5 min each = 40 minutes for training
Plus evaluation/model selection = +20 minutes
**Total: ~60 minutes** ✅ This matches!

## The Real Problem: max_steps is TOO LOW!

### Current: max_steps = 100

This was REDUCED from original values:
- DGHL: 1000 → 100 (10x reduction)
- MD: 5000 → 100 (50x reduction!)  
- LSTMVAE: 1000 → 100 (10x reduction)

**Why this is a problem:**
1. Models don't converge properly
2. Poor training = poor accuracy
3. Model selection algorithms get confused
4. May need to RETRAIN multiple times

### Impact

**Undertraining** can actually SLOW THINGS DOWN because:
- Models perform poorly
- Ensemble selection takes longer (trying bad combinations)
- Thompson sampling needs more iterations to converge
- You might need to retrain with better hyperparameters later

## Recommendations

### Option 1: Increase max_steps ⭐ RECOMMENDED

Restore reasonable training steps:

```python
# In hyperparameter_grids.py

LSTMVAE_TRAIN_PARAM_GRID = {
    'max_steps': [500],  # Was 1000, now 100 → compromise at 500
}

DGHL_TRAIN_PARAM_GRID = {
    'max_steps': [500],  # Was 1000, now 100 → compromise at 500
}

MD_TRAIN_PARAM_GRID = {
    'max_steps': [1000],  # Was 5000, now 100 → compromise at 1000
}

RNN_TRAIN_PARAM_GRID = {
    'max_steps': [500],  # compromise
}
```

**Impact:**
- Training time: 16-40 min → 80-200 min (5x longer)
- BUT: Models actually learn properly
- Better accuracy, less debugging, no need to retrain

### Option 2: Use GPU Instead of CPU ⭐⭐⭐ BEST

If you have GPU available:

```python
# In hyperparameter_grids.py
LSTMVAE_PARAM_GRID = {
    'device': ['cuda'],  # or ['cuda:0'] for specific GPU
}
```

**Impact:**
- Training time: 16-40 min → 2-5 min (10x faster!)
- Can increase max_steps without time penalty
- Best of both worlds: fast AND well-trained

### Option 3: Kill Duplicate Process

You have TWO processes training machine-1-1:

```bash
kill 336023  # Kill the older one
```

This won't make training faster, but frees up resources.

### Option 4: Use Smaller Datasets for Testing

If you're just testing the system:
- Use a smaller entity
- Use fewer models
- Then scale up to full run

## Summary

**Training time breakdown for ONE entity:**
1. Training 8 models: 16-40 minutes (with max_steps=100)
2. Model selection (parallel): 10-20 minutes
3. Robustness tests: 5-10 minutes
4. **Total: 30-70 minutes per entity**

**This is actually REASONABLE!** Deep learning models take time to train.

**The real issue:** `max_steps=100` is probably TOO LOW for good model quality.

**Best fix:** Use GPU (`device='cuda'`) and increase max_steps to 500-1000.

## Quick Actions

```bash
# 1. Check if you have GPU
nvidia-smi

# 2. If yes, I can update hyperparameter_grids.py to use GPU
#    This will give 10-50x speedup!

# 3. Also increase max_steps to reasonable values
#    Training will be slower but models will actually work properly
```

Would you like me to:
1. **Enable GPU training** (if available)
2. **Increase max_steps** to 500-1000 for better model quality
3. **Both** (GPU + more steps = fast AND good models)
4. **Neither** (keep current settings)

Let me know!
