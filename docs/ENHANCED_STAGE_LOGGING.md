# Enhanced Stage-by-Stage Logging

## Overview
The RAMSeS execution has been enhanced with **detailed stage tracking** to show exactly what's happening at any moment.

## 7 Main Stages

### 🚀 STAGE 1/7: Loading Training Data
**What happens**: Loads training dataset from disk
**Time**: ~5-10 seconds
**Output**:
```
📂 STAGE 1/7: Loading Training Data...
✓ Training data loaded: 1 entity(ies)
```

---

### 📂 STAGE 2/7: Loading Test Data
**What happens**: Loads test dataset from disk
**Time**: ~5-10 seconds
**Output**:
```
📂 STAGE 2/7: Loading Test Data...
✓ Test data loaded: 1 entity(ies)
```

---

### 🔧 STAGE 3/7: Training/Loading Models
**What happens**: 
1. Checks which models already exist
2. Trains missing models (if any)
3. Loads all trained models into memory

**Time**: 
- If all models exist: ~5-10 seconds
- If training missing models: ~15-30 minutes (depends on how many are missing)

**Output**:
```
🔧 STAGE 3/7: Training/Loading Models...
  → Checking existing models and training missing ones...
  Model RNN_1 already trained!  ← Skipping existing
  Training RNN_2...              ← Training new model
  ...
  ✓ Model training phase complete
  → Loading trained models from disk...
  WARNING: Model RNN_3 not found, skipping
  ✓ Loaded 14 models: LOF_1, LOF_2, ..., DGHL_2
```

---

### 💉 STAGE 4/7: Injecting Synthetic Anomalies
**What happens**: Injects synthetic anomalies into test data for evaluation
**Time**: ~1-2 seconds
**Output**:
```
💉 STAGE 4/7: Injecting Synthetic Anomalies...
✓ Injected anomalies: ['spikes']
```

---

### 📊 STAGE 5/7: Preparing Data and Visualization
**What happens**: 
- Creates visualization of injected anomalies
- Sets up sliding windows for evaluation

**Time**: ~5-10 seconds
**Output**:
```
📊 STAGE 5/7: Preparing Data and Visualization...
✓ Visualization saved to myresults/GA_Ens/...
```

---

### 🔍 STAGE 6/7: Running Model Selection Algorithms
**What happens**: This is the LONGEST stage with 6 sub-stages

**Time**: ~60-90 minutes (most of the execution time!)

#### Sub-stages:

##### 📊 Sub-stage 6.1: Genetic Algorithm (GA)
**What happens**: 
1. Evaluates each individual model (this is SLOW for LOF models!)
2. Runs 20 generations to find best ensemble

**Time**: ~30-50 minutes
- Individual model evaluation: ~20-30 minutes (LOF takes 3 min per model!)
- 20 GA generations: ~10-20 minutes

**Output**:
```
📊 Sub-stage 6.1: Genetic Algorithm (GA) - Finding best ensemble...
  This will evaluate individual models and run 20 generations

# Individual model evaluation (SLOWEST PART!)
Model LOF_1: F1 score = 0.9338... ← 3 minutes
Model LOF_2: F1 score = 0.9338... ← 3 minutes
Model LOF_3: F1 score = 0.9338... ← 3 minutes
Model LOF_4: F1 score = 0.9338... ← 3 minutes
Model NN_1: F1 score = 0.6228...  ← 30 seconds
...

# Genetic algorithm generations
Generation 1
Evaluated fitness for ensemble: ['LOF_1', 'RNN_1', ...]
Generation 2
...
Generation 20

✓ [GA] Best ensemble=['LOF_1', 'RNN_1', ...] | F1=0.9234 | Time=2400s
```

##### 📊 Sub-stage 6.2: Thompson Sampling
**What happens**: Online model selection using Thompson Sampling
**Time**: ~2-3 minutes
**Output**:
```
📊 Sub-stage 6.2: Thompson Sampling - Online model selection...
✓ [Thompson] Top-5: ['LOF_1', 'RNN_1', ...] | Time=180s
```

##### 📊 Sub-stage 6.3: GAN Robustness Testing
**What happens**: Tests model robustness against GAN-generated adversarial examples
**Time**: ~5-8 minutes
**Output**:
```
📊 Sub-stage 6.3: GAN Robustness Testing...
✓ [GAN] F1 names top-5: ['LOF_1', ...] | Time=450s
  [GAN] PR names top-5: ['LOF_2', ...]
```

##### 📊 Sub-stage 6.4: Off-by-Threshold Testing
**What happens**: Tests sensitivity to threshold changes
**Time**: ~3-5 minutes
**Output**:
```
📊 Sub-stage 6.4: Off-by-Threshold Testing...
✓ [Borderline] F1 names top-5: ['LOF_1', ...] | Time=320s
  [Borderline] PR names top-5: ['LOF_2', ...]
```

##### 📊 Sub-stage 6.5: Monte Carlo Simulation
**What happens**: Stress tests models with noise
**Time**: ~3-5 minutes
**Output**:
```
📊 Sub-stage 6.5: Monte Carlo Simulation...
✓ [MonteCarlo] F1 names top-5: ['LOF_1', ...] | Time=320s
  [MonteCarlo] PR names top-5: ['LOF_2', ...]
```

##### 📊 Sub-stage 6.6: Rank Aggregation
**What happens**: Combines all rankings to make final decision
**Time**: ~10-20 seconds
**Output**:
```
📊 Sub-stage 6.6: Rank Aggregation...
✓ [Aggregation] Time=15s
```

---

### 📝 STAGE 7/7: Writing Comprehensive Results
**What happens**: Writes all results to text file
**Time**: ~1-2 seconds
**Output**:
```
📝 STAGE 7/7: Writing Comprehensive Results...
✓ Results written to: myresults/comprehensive/...
🎉 EXECUTION COMPLETE! Total Time: 5400.45s (90.01 min)
```

---

## Progress Tracking

### While Running:
You'll see periodic status updates every 30 seconds:
```
⏱ Status: Running for 189s, Memory: 99.8 MB
⏱ Status: Running for 371s, Memory: 99.8 MB
⏱ Status: Running for 553s, Memory: 99.8 MB
```

### To monitor:
```bash
# Watch log file in real-time
tail -f testbed_run_*.log

# Show only stage messages
tail -f testbed_run_*.log | grep -E "STAGE|Sub-stage|✓"

# Show timing information
tail -f testbed_run_*.log | grep -E "STAGE|Time="
```

---

## Time Breakdown Example (SMD dataset)

| Stage | Sub-stage | Time | % of Total |
|-------|-----------|------|------------|
| 1 | Loading Training Data | 5s | <1% |
| 2 | Loading Test Data | 5s | <1% |
| 3 | Training/Loading Models | 10s* | <1% |
| 4 | Injecting Anomalies | 2s | <1% |
| 5 | Preparing Visualization | 5s | <1% |
| **6** | **Model Selection** | **~5400s** | **~99%** |
| 6.1 | - Individual Evaluation | 1200s | 22% |
| 6.1 | - GA Generations | 1200s | 22% |
| 6.2 | - Thompson Sampling | 180s | 3% |
| 6.3 | - GAN Robustness | 450s | 8% |
| 6.4 | - Off-by-Threshold | 320s | 6% |
| 6.5 | - Monte Carlo | 320s | 6% |
| 6.6 | - Rank Aggregation | 15s | <1% |
| 7 | Writing Results | 2s | <1% |
| **TOTAL** | | **~5430s** | **100%** |

*\*Assuming models already trained. If training from scratch, add 60-90 minutes.*

---

## Why Individual Model Evaluation is Slow

The **individual model evaluation in Stage 6.1** is the slowest part because:

1. **LOF models are O(n²) complexity**
   - Must compute k-nearest neighbors for EVERY data point
   - SMD datasets have ~450,000 data points with 2368 features
   - Each LOF model takes ~3 minutes

2. **4 LOF models = 12 minutes total**
   - This is unavoidable with the current LOF implementation
   - Necessary to properly evaluate LOF performance

3. **Why you're stuck at "LOF_1", "LOF_2", "LOF_3"**:
   ```
   14:16:24 - Model LOF_1: F1 score = 0.9338...  ← 3 min
   14:19:25 - Model LOF_2: F1 score = 0.9338...  ← 3 min  
   14:22:28 - Model LOF_3: F1 score = 0.9338...  ← 3 min
   14:25:xx - Model LOF_4: F1 score = ...        ← 3 min (in progress!)
   ```

**This is normal!** The system is working correctly, just slowly due to LOF complexity.

---

## What to Look For

### Everything is fine if you see:
✅ Stage numbers progressing (1/7 → 2/7 → ... → 7/7)
✅ Regular status updates every 30 seconds
✅ Model names appearing (LOF_1, LOF_2, etc.)
✅ Generation numbers increasing (Generation 1, 2, 3, ...)
✅ Memory staying relatively stable (~100 MB)

### Something might be wrong if you see:
❌ No updates for more than 5 minutes
❌ Memory continuously growing
❌ Same generation number for > 10 minutes
❌ Error messages or tracebacks

---

## Quick Reference

### Where are we now?
```bash
# Show last stage message
tail -f testbed_run_*.log | grep -E "STAGE [0-9]/7|Sub-stage"
```

### How long has each stage taken?
```bash
# Show all timing messages
grep -E "✓.*Time=" testbed_run_*.log
```

### What models are being evaluated?
```bash
# Show model evaluation progress
grep -E "Model.*F1 score" testbed_run_*.log
```

### Which generation are we on?
```bash
# Show GA progress
grep "Generation" testbed_run_*.log | tail -5
```

---

## Expected Timeline (SMD dataset)

```
Time    Stage                           What You'll See
------  ------------------------------  ----------------------------------
00:00   🚀 STARTING                     Startup message
00:05   📂 STAGE 1: Loading Train      Loading training data...
00:10   📂 STAGE 2: Loading Test       Loading test data...
00:20   🔧 STAGE 3: Models             Checking/loading models...
00:30   💉 STAGE 4: Anomalies          Injecting anomalies...
00:35   📊 STAGE 5: Visualization      Preparing data...
00:45   🔍 STAGE 6: Model Selection    Starting longest stage...
00:50   📊 Sub-stage 6.1: GA           Evaluating individual models...
01:00   - Model LOF_1                  First LOF model (SLOW!)
04:00   - Model LOF_2                  Second LOF model
07:00   - Model LOF_3                  Third LOF model
10:00   - Model LOF_4                  Fourth LOF model
13:00   - Other models                 Faster models (NN, RNN, etc.)
20:00   - Generation 1                 GA starting...
25:00   - Generation 5                 
30:00   - Generation 10                Halfway through GA
40:00   - Generation 20                GA complete
45:00   📊 Sub-stage 6.2: Thompson     Thompson Sampling
48:00   📊 Sub-stage 6.3: GAN          GAN testing
55:00   📊 Sub-stage 6.4: Borderline   Threshold testing
60:00   📊 Sub-stage 6.5: Monte Carlo  Noise testing
65:00   📊 Sub-stage 6.6: Aggregation  Combining results
67:00   📝 STAGE 7: Writing Results    Almost done!
68:00   🎉 COMPLETE!                   Finished!
```

---

## Summary

**Enhanced logging now shows:**
- ✅ 7 main stages with emoji indicators
- ✅ 6 sub-stages within Stage 6
- ✅ Individual model evaluation progress
- ✅ Generation-by-generation GA progress
- ✅ Timing for each stage
- ✅ Clear completion markers

**You'll always know:**
- 📍 Where you are in the process
- ⏱️ How long each stage is taking
- 🎯 What's coming next
- ✅ When each stage completes

**No more wondering "Is it stuck?" - you'll see continuous progress!** 🚀
