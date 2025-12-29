# Why Parallel Mode Might Be Slower - Analysis

## Summary

You're experiencing slower performance with:
- ✅ Parallel mode enabled (`--parallel true`)
- ✅ Only training ONE model version (reduced from 19 models)
- ✅ Training the SMALLEST model (256/128 instead of 512/256)

## Model Comparison

### OLD Sequential Training
| Model | Hidden | Latent | Parameters | File Size | Training Cost |
|-------|--------|--------|------------|-----------|---------------|
| LSTMVAE_1 | 512 | 256 | 7,839,782 | 60 MB | 3.95x |
| LSTMVAE_2 | 512 | 128 | 7,642,918 | 59 MB | 3.85x |
| LSTMVAE_3 | 256 | 256 | 2,085,158 | 16 MB | 1.05x |
| LSTMVAE_4 | 256 | 128 | 1,986,598 | 16 MB | **1.00x** |
| + 15 other models | - | - | - | - | - |
| **TOTAL** | - | - | - | **~300 MB** | **19 models** |

### NEW Parallel Training
| Model | Hidden | Latent | Parameters | File Size | Training Cost |
|-------|--------|--------|------------|-----------|---------------|
| LSTMVAE_1 | 256 | 128 | 1,986,598 | 16 MB | **1.00x** |
| **TOTAL** | - | - | - | **16 MB** | **1 model** |

**Conclusion**: The new model is the SMALLEST and FASTEST to train! 🎉

## So Why Is It Slower?

### 1. Parallel Mode Overhead ⚠️

The parallel implementation makes **5 deep copies** of test_data:

```python
# From app.py lines 540-548
test_data_ga = copy.deepcopy(test_data)
test_data_thompson = copy.deepcopy(test_data)
test_data_gan = copy.deepcopy(test_data)
test_data_borderline = copy.deepcopy(test_data)
test_data_montecarlo = copy.deepcopy(test_data)
```

**Cost of deepcopy**:
- If test_data is large (e.g., 100MB in memory), this creates 500MB total
- For machine-1-1 with 500k+ time points, this could take 5-30 seconds
- This happens BEFORE any parallel execution begins

### 2. Thread Contention

ThreadPoolExecutor with 5 workers may cause:
- **GIL (Global Interpreter Lock) contention** - Python threads fighting for CPU
- **Memory bandwidth saturation** - All 5 threads reading/writing data simultaneously
- **Cache thrashing** - Each thread's data evicting others from CPU cache

### 3. You're Comparing Different Stages! ⚠️⚠️⚠️

**Most Important**: Check WHAT is running:

```bash
# Currently running processes:
PID     ELAPSED    CMD
336023  01:28:44   python app.py --dataset servermachinedataset --entity machine-1-1 --parallel true
1102331 00:58:55   python app.py --dataset servermachinedataset --entity machine-1-1 --parallel true
```

You have **TWO processes** running machine-1-1 simultaneously!

### 4. Training Phase vs Evaluation Phase

The system has two main phases:

**Phase 1: Model Training** (fast, now 1 model instead of 19)
- ✅ This SHOULD be 19x faster!
- Old: Train 19 models (LSTMVAE_1-4, NN_1-3, LOF_1-4, etc.)
- New: Train 1 model (LSTMVAE_1 only)

**Phase 2: Model Selection Evaluation** (may be slower in parallel)
- GA (genetic algorithm)
- Thompson sampling
- GAN robustness
- Off-by-threshold
- Monte Carlo simulation

**If you're in Phase 2**, parallel mode overhead might outweigh benefits!

## Performance Analysis

### Sequential Mode
```
Total Time = Training_Time + (GA + Thompson + GAN + Borderline + MonteCarlo)
           = 19_models × T  + (5 × E)
           = 19T + 5E
```

### Parallel Mode (with overhead)
```
Total Time = Training_Time + DeepCopy_Overhead + max(GA, Thompson, GAN, Borderline, MonteCarlo)
           = 1_model × T  + 5×C + max(E, E, E, E, E)
           = T + 5C + E
```

Where:
- T = time to train one model (~5-10 minutes for LSTMVAE)
- E = time for one evaluation algorithm (~2-5 minutes)
- C = time to deepcopy test_data (~5-30 seconds)

### When Parallel is FASTER
```
T + 5C + E < 19T + 5E
5C + E < 18T + 5E
5C < 18T + 4E

If: 5C < 18T + 4E  → Parallel WINS
```

**Example**: If T=5min, E=2min, C=10sec:
```
5(10s) + 2min < 18(5min) + 5(2min)
50s + 2min < 90min + 10min
2.83min < 100min  → PARALLEL WINS by 97 minutes! ✅
```

### When Parallel is SLOWER ⚠️

If:
1. Training time (T) is very small (models already trained)
2. You're just re-running evaluation on existing models
3. test_data is HUGE (deepcopy takes minutes)

**Example**: Re-running evaluation only (T=0):
```
5C + E vs 5E
5(30s) + 2min vs 5(2min)
4.5min vs 10min  → PARALLEL WINS by 5.5 minutes ✅
```

Even with huge deepcopy cost, parallel still wins!

## Most Likely Explanation

### You Have TWO Processes Running! 🔴

```bash
ps -p 336023,1102331 -o pid,etime,cmd
PID     ELAPSED CMD
336023  01:28:44 python app.py --dataset servermachinedataset --entity machine-1-1 --parallel true
1102331 00:58:55 python app.py --dataset servermachinedataset --entity machine-1-1 --parallel true
```

**This means**:
- Process 1 started 1h28min ago (probably from OLD screen session)
- Process 2 started 58min ago (probably after cache clear)
- Both are competing for resources!
- Each uses 5 parallel workers = **10 workers total**
- On 64 cores, this shouldn't be bad, BUT:
  - They're both trying to write to same model files
  - They're both reading/writing same data
  - Memory bandwidth is shared

## Recommendations

### Option 1: Kill Old Processes ⭐ RECOMMENDED
```bash
# Stop the older process
kill 336023

# Let the newer one (1102331) continue
# This one has the new hyperparameter grids
```

### Option 2: Use Sequential Mode for Evaluation Only
If models are already trained and you're just re-evaluating:
```bash
python app.py --dataset servermachinedataset --entity machine-1-1  # no --parallel flag
```

Sequential mode avoids the 5× deepcopy overhead.

### Option 3: Optimize Parallel Mode
Reduce deepcopy overhead by:
1. Use shallow copies where possible (risky - need careful analysis)
2. Use multiprocessing instead of threading (avoids GIL)
3. Reduce number of parallel workers from 5 to 3

### Option 4: Measure Each Phase Separately
```bash
# Add timing logs to see WHERE the time goes:
# - Model training time
# - Deepcopy time
# - GA time
# - Thompson time
# - etc.
```

## Quick Test

To verify what's slow, check:

```bash
# 1. Are models already trained?
ls -lh Mononito/trained_models/servermachinedataset/machine-1-1/*.pth

# 2. If yes, you're in evaluation phase
# 3. Check memory size of test_data
# 4. Time the deepcopy manually:

python3 -c "
import time
import copy
import pickle
# Load your test_data
# test_data = ... load it ...
start = time.time()
test_copy = copy.deepcopy(test_data)
print(f'Deepcopy took: {time.time()-start:.2f} seconds')
"
```

## Bottom Line

**Training is definitely faster** (1 model vs 19 models).

**Evaluation might be slower** due to:
1. ⚠️ **TWO PROCESSES RUNNING** (kill the old one!)
2. Deepcopy overhead (5× copies of large data)
3. Thread contention (GIL, memory bandwidth)
4. You're comparing apples to oranges (different stages)

**Kill the old process (PID 336023) and re-measure!**
