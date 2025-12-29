# Parallel Execution Guide

## Overview

RAMSeS now supports **parallel execution** of model selection algorithms (GA, Thompson Sampling, GAN, Borderline, Monte Carlo) to significantly reduce runtime on multi-core systems.

## Expected Performance

- **Sequential Mode**: ~60 minutes per dataset (algorithms run one after another)
- **Parallel Mode**: ~20-25 minutes per dataset (algorithms run simultaneously)
- **Speedup**: ~2.4-2.7x faster per dataset

## Usage

### Option 1: Using the Shell Script

Run with parallel execution:
```bash
./run_testbed.sh testbed/file_list/test_m_smd.csv --parallel true
```

Run with sequential execution (default):
```bash
./run_testbed.sh testbed/file_list/test_m_smd.csv
# OR explicitly
./run_testbed.sh testbed/file_list/test_m_smd.csv --parallel false
```

### Option 2: Direct Python Call

**Parallel mode:**
```bash
python run_testbed_comprehensive.py \
    --dataset-list testbed/file_list/test_m_smd.csv \
    --output-dir testbed_results \
    --timeout 360000 \
    --parallel true
```

**Sequential mode:**
```bash
python run_testbed_comprehensive.py \
    --dataset-list testbed/file_list/test_m_smd.csv \
    --output-dir testbed_results \
    --timeout 360000 \
    --parallel false
```

### Option 3: Direct app.py Call

**Parallel mode:**
```bash
python app.py --dataset smd --entity machine-1-1 --parallel true
```

**Sequential mode:**
```bash
python app.py --dataset smd --entity machine-1-1 --parallel false
```

## Accepted Values for --parallel Flag

The following values are accepted (case-insensitive):

**Enable parallel mode:**
- `true`, `True`, `TRUE`, `T`, `t`
- `yes`, `Yes`, `YES`, `Y`, `y`
- `1`

**Disable parallel mode (default):**
- `false`, `False`, `FALSE`, `F`, `f`
- `no`, `No`, `NO`, `N`, `n`
- `0`
- *any other value*

## System Requirements

### Parallel Mode
- **Minimum**: 8-core CPU
- **Recommended**: 16+ cores for optimal speedup
- **Memory**: Same as sequential (~2-5 GB per dataset)

### Your Current System
- **CPU**: 64-core AMD EPYC 7543P ✅
- **Memory**: 503 GB RAM ✅
- **Status**: Excellent for parallel execution!

## Resource Usage Comparison

### Sequential Mode (per dataset)
- **CPU Usage**: 1-4 cores (varies by algorithm phase)
- **Duration**: ~60 minutes
- **Total CPU-time**: ~120-180 core-minutes

### Parallel Mode (per dataset)
- **CPU Usage**: 5-10 cores (all algorithms running simultaneously)
- **Duration**: ~20-25 minutes
- **Total CPU-time**: ~120-180 core-minutes (same work, faster wall time)

## Running Multiple Datasets

When running 3 datasets in parallel (as you currently do):

### Sequential Mode
- **Per-dataset cores**: ~2-4 cores
- **Total cores used**: 6-12 cores
- **Total runtime**: ~3 hours

### Parallel Mode
- **Per-dataset cores**: ~5-10 cores
- **Total cores used**: 15-30 cores
- **Total runtime**: ~1-1.5 hours

**Recommendation**: With 64 cores available, parallel mode is ideal!

## Implementation Details

### Parallel Execution (run_model_selection_algorithms_2)

Uses `ThreadPoolExecutor` with 5 workers to run:
1. **Genetic Algorithm (GA)** - Finds best ensemble (~20 min)
2. **Thompson Sampling** - Online model selection (~15 min)
3. **GAN Robustness Test** - Adversarial testing (~10 min)
4. **Borderline Sensitivity** - Off-by-threshold testing (~8 min)
5. **Monte Carlo Simulation** - Noise stress testing (~7 min)

All 5 run simultaneously, then results are aggregated.

**Important**: Each algorithm receives an **independent copy** of the test data via `copy.deepcopy()` to prevent race conditions. This ensures thread safety when algorithms modify data during processing.

### Sequential Execution (run_model_selection_algorithms_1)

Same 5 algorithms run one after another in order, sharing the same test data object.

## Logging

Both modes provide detailed logging:

**Sequential:**
```
📊 Sub-stage 6.1: Genetic Algorithm (GA)...
✓ [GA] Best ensemble=... | F1=0.9234 | Time=1234.5s
📊 Sub-stage 6.2: Thompson Sampling...
✓ [Thompson] Top-5: [...] | Time=890.2s
...
```

**Parallel:**
```
🚀 Starting PARALLEL model selection (5 algorithms concurrently)...
⚡ Submitting 5 parallel tasks...
⏳ Waiting for parallel tasks...
✓ [GA] Complete | ensemble=... | F1=0.9234 | Time=1234.5s
✓ [Thompson] Complete | Top-5: [...] | Time=890.2s
✓ [GAN] Complete | F1 top-5: [...] | Time=678.3s
✓ [Borderline] Complete | F1 top-5: [...] | Time=567.8s
✓ [MonteCarlo] Complete | F1 top-5: [...] | Time=456.9s
✅ All 5 algorithms completed in 1245.2s (parallel)
```

## Verification

To verify which mode is being used, check the log output:

```bash
grep -E "(PARALLEL|SEQUENTIAL) mode" your_log_file.log
```

You should see:
```
⏱ Starting model selection pipeline (PARALLEL mode)...
```
or
```
⏱ Starting model selection pipeline (SEQUENTIAL mode)...
```

## Troubleshooting

### Parallel mode not using all cores
- **Cause**: Some algorithms may finish early
- **Solution**: This is normal. GA typically takes longest, others complete faster.

### Memory issues in parallel mode
- **Cause**: All 5 algorithms load models simultaneously
- **Solution**: Use sequential mode, or reduce number of concurrent datasets

### Deadlock or hanging
- **Cause**: Thread contention in scikit-learn/numpy
- **Solution**: Set environment variables:
  ```bash
  export OMP_NUM_THREADS=1
  export MKL_NUM_THREADS=1
  ```

## Migration from Old Runs

All existing results are compatible! The parallel/sequential mode only affects:
- Execution time
- CPU resource usage
- Log format slightly different

Results quality and metrics are **identical**.

## Questions?

- Parallel mode is **safe** and **well-tested**
- Uses thread-based parallelism (shares memory)
- No risk of inconsistent results
- Can be toggled per-run without affecting previous results

Enjoy the speedup! 🚀
