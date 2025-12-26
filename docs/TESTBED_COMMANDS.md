# RAMSeS Testbed Commands Guide

## Overview
The testbed has been enhanced with:
- ✅ **Flexible model loading** - Skips missing models without errors
- ✅ **Comprehensive logging** - Shows progress, timing, and status for each step
- ✅ **Smart caching** - Reuses existing results to avoid recomputation
- ✅ **Dynamic model paths** - Correctly loads models based on dataset and entity

## Quick Start Commands

### 1. Test with Small SMD Dataset (2 datasets - RECOMMENDED FOR TESTING)
```bash
cd /home/maxoud/local-storage/projects/RAMSeS
./run_testbed.sh testbed/file_list/test_smd_small.csv
```

### 2. Run Full SMD Testbed (All SMD datasets)
```bash
cd /home/maxoud/local-storage/projects/RAMSeS
./run_testbed.sh testbed/file_list/test_m_smd.csv
```

### 3. Run SKAB Testbed
```bash
cd /home/maxoud/local-storage/projects/RAMSeS
./run_testbed.sh testbed/file_list/test_m_skab.csv
```

### 4. Run with Custom Timeout (e.g., 2 hours = 7200 seconds)
```bash
cd /home/maxoud/local-storage/projects/RAMSeS
python run_testbed_comprehensive.py \
    --dataset-list testbed/file_list/test_m_smd.csv \
    --timeout 7200
```

### 5. Direct Python Command
```bash
cd /home/maxoud/local-storage/projects/RAMSeS
python run_testbed_comprehensive.py \
    --dataset-list testbed/file_list/test_smd_small.csv \
    --output-dir testbed_results \
    --timeout 3600
```

## What the Enhanced Logging Shows

### For Each Dataset:
```
================================================================================
STARTING PROCESSING: machine-1-1.txt from domain servermachinedataset
================================================================================
  Dataset: machine-1-1.txt
  Domain:  servermachinedataset
  Entity:  machine-1-1
  Checking for existing results: myresults/comprehensive/...
  
→ No existing results found, running computation...
→ Starting app.py for servermachinedataset/machine-1-1...
  Command: python app.py --dataset servermachinedataset --entity machine-1-1
  Timeout: 3600s
  Started at: 2025-12-25 12:00:00
  → Launching subprocess...
  → Process started (PID: 12345)
  
    # Model loading progress
    WARNING:__main__:Model RNN_2 not found, skipping
    WARNING:__main__:Model RNN_3 not found, skipping
    INFO:__main__:Loaded 14 trained models from .../machine-1-1
    INFO:__main__:Using 14 loaded models: LOF_1, LOF_2, ...
    
    # Genetic Algorithm progress
    Generation 1
    Evaluated fitness for ensemble: ['LOF_1', 'RNN_1', 'CBLOF_2']
    Generation 2
    ...
    
  ⏱ Status: Running for 180s, Memory: 1234.5 MB  # Every 30 seconds
  
  → Process finished with return code: 0
  → Total lines captured: 5432
✓ EXECUTION COMPLETED: machine-1-1.txt in 300.45s
  → Parsing results from: myresults/comprehensive/...
  ✓ Results parsed successfully
  → Memory stats: Avg=980.3 MB, Peak=1234.5 MB

✓ COMPLETED SUCCESSFULLY: machine-1-1.txt
  Total Runtime: 300.45s
  Peak Memory: 1234.5 MB
================================================================================
```

### For Cached Results:
```
================================================================================
STARTING PROCESSING: machine-1-2.txt from domain servermachinedataset
================================================================================
  Dataset: machine-1-2.txt
  Domain:  servermachinedataset
  Entity:  machine-1-2
  Checking for existing results: myresults/comprehensive/...
✓ Results already exist for machine-1-2.txt, skipping computation
  Parsing existing results from: myresults/comprehensive/...
✓ COMPLETED (from cache): machine-1-2.txt
  Runtime: 285.32s
================================================================================
```

### Domain Summary:
```
╔══════════════════════════════════════════════════════════════════════════════╗
║ DOMAIN COMPLETE: servermachinedataset                                       ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Total Datasets:    2                                                        ║
║  Successful:        2                                                        ║
║  Failed:            0                                                        ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

## Key Features

### 1. Missing Models Handling
- If models are missing (e.g., RNN_2, RNN_3, DGHL_3, DGHL_4):
  - ⚠️ **WARNING** logged for each missing model
  - ✅ Continues with available models
  - 📊 Shows count: "Loaded 14 trained models" instead of "Loaded 19 trained models"
  - 🔄 Genetic algorithm uses only loaded models

### 2. Progress Tracking
- ⏱️ **Every 30 seconds**: Status update with elapsed time and memory
- 📝 **Important events logged**:
  - Model loading (which models loaded, which skipped)
  - Generation progress
  - F1 scores
  - Ensemble selection
  - Final decision

### 3. Error Handling
- ❌ **Timeouts**: Kills process after timeout, logs last 10 lines
- ❌ **Failures**: Logs last 20 lines of output for debugging
- ⚠️ **Warnings**: Logs errors/warnings from app.py

### 4. Output Files
- **Log file**: `testbed_run_YYYYMMDD_HHMMSS.log` (timestamped)
- **Results**: `testbed_results/` directory
- **Intermediate**: Saved after each successful dataset

## Monitoring Running Testbed

### Watch progress in real-time:
```bash
tail -f testbed_run_*.log
```

### Check for errors:
```bash
grep -i "error\|failed\|warning" testbed_run_*.log
```

### Check successful completions:
```bash
grep "✓ COMPLETED" testbed_run_*.log
```

### Count progress:
```bash
grep "COMPLETED SUCCESSFULLY" testbed_run_*.log | wc -l
```

## File Locations

### Input Files:
- `testbed/file_list/test_smd_small.csv` - 2 SMD datasets (for testing)
- `testbed/file_list/test_m_smd.csv` - All SMD datasets
- `testbed/file_list/test_m_skab.csv` - All SKAB datasets

### Output Files:
- `testbed_results/` - Final aggregated results
- `myresults/comprehensive/{domain}/{entity}/` - Individual results
- `testbed_run_*.log` - Execution logs

### Models:
- `Mononito/trained_models/{dataset}/{entity}/` - Trained models

## Troubleshooting

### Issue: "Model XXX not found"
**Solution**: This is normal! The testbed will skip missing models and continue.

### Issue: Process seems stuck
**Solution**: Check the log file - status updates every 30 seconds show it's still running.

### Issue: Timeout
**Solution**: Increase timeout with `--timeout 7200` (2 hours) or higher.

### Issue: Out of memory
**Solution**: 
1. Close other applications
2. Run fewer datasets at once (use test_smd_small.csv)

## Example Complete Run

```bash
# 1. Navigate to RAMSeS directory
cd /home/maxoud/local-storage/projects/RAMSeS

# 2. Run small test (2 datasets)
./run_testbed.sh testbed/file_list/test_smd_small.csv

# 3. Monitor progress (in another terminal)
tail -f testbed_run_*.log

# 4. After completion, check results
ls -lh testbed_results/
```

## Expected Behavior

### For machine-1-1 (partially trained):
- ✅ Loads 14 models: LOF (1-4), NN (1-3), RNN_1, CBLOF (1-4), DGHL (1-2)
- ⚠️ Skips 5 models: RNN_2, RNN_3, RNN_4, DGHL_3, DGHL_4
- ✅ Runs successfully with 14 models

### For machine-1-2 (if trained):
- ✅ Loads available models
- ✅ Runs successfully

### Timing Expectations:
- **With cached results**: ~0.01 seconds per dataset
- **First run (no cache)**: ~3-5 minutes per dataset
- **Training included**: ~10-30 minutes per dataset (if models don't exist)

## Summary

**RECOMMENDED COMMAND TO START:**
```bash
cd /home/maxoud/local-storage/projects/RAMSeS
./run_testbed.sh testbed/file_list/test_smd_small.csv
```

This will:
- ✅ Run 2 SMD datasets with full debugging
- ✅ Skip missing models automatically
- ✅ Show detailed progress
- ✅ Create timestamped log file
- ✅ Complete in ~6-10 minutes (or ~0.02s if cached)
