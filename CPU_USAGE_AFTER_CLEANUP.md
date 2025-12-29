# CPU Usage After Killing SMD Process

**Updated**: December 26, 2025 (after killing duplicate SMD processes)

## Current Status

### Total System Capacity
- **Total cores**: 64
- **CPU utilization**: 98.9% user + 1.1% system = **100% BUSY**
- **Idle**: 0.0% = **0 cores free** 🔴
- **Status**: FULLY LOADED

---

## Active Processes

### 1. ramses-anomaly-archive (PID 1121046) 🟢
- **CPU**: 4264% ≈ **~42-43 cores**
- **Memory**: 3.1 GB
- **Runtime**: 1041 minutes = **17 hours 21 minutes**
- **Command**: `python app.py --dataset anomaly_archive --entity 077_UCR_Anomaly_DISTORTEDresperation11_58000_110800_110801 --parallel false`
- **Screen**: ramses-anomaly-archive
- **Mode**: Sequential (--parallel false)

**Status**: Still processing the same UCR Anomaly Archive entity for over 17 hours! 😱

### 2. New SMD Process (PID 1124916) 🟢
- **CPU**: 1467% ≈ **~15 cores**
- **Command**: `python app.py --dataset servermachinedataset --entity machine-1-7 --parallel true`
- **Status**: Now processing machine-1-7 (you killed the machine-1-1 processes)
- **Mode**: Parallel enabled

### 3. fatemeh's process (PID 213622) ⚪
- **CPU**: ~100% ≈ **1 core**
- **User**: fatemeh (not yours)
- **Status**: Untouched as requested ✅

### 4. Other processes
- **VS Code, background services**: ~6 cores

---

## Core Distribution

| Process | Cores | Percentage | Owner | Status |
|---------|-------|------------|-------|--------|
| **ramses-anomaly-archive** | **~43** | **67%** | maxoud | Running 17h 21m |
| **SMD machine-1-7** | **~15** | **23%** | maxoud | New process |
| fatemeh's job | ~1 | 2% | fatemeh | Untouched |
| Other/Background | ~5 | 8% | system | Various |
| **Available** | **0** | **0%** | - | **NONE** |

---

## Key Observations

### ✅ Good News
1. **You successfully killed the old SMD processes** (machine-1-1)
2. **New SMD process started** on machine-1-7 with parallel mode
3. **fatemeh's process untouched** as requested
4. **System is stable** - no crashes

### ⚠️ Concerns
1. **ramses-anomaly-archive is using 43 cores!** (increased from 34 cores earlier)
   - This is MORE than before!
   - Sequential mode but still using lots of cores
   - Been running for **17+ hours on ONE entity**

2. **No cores available** for new work
   - System is 100% utilized
   - Any new job will compete for resources

3. **UCR Anomaly Archive is VERY SLOW**
   - 17 hours for one entity
   - 250+ entities total
   - **Estimated total time: 17h × 250 = 4,250 hours = 177 days!** 😱😱😱

---

## Why Is ramses-anomaly-archive Using 43 Cores?

Even though it's running with `--parallel false`, it's still using 43 cores because:

1. **Internal parallelism**: Many Python libraries (numpy, scipy, sklearn, PyOD) automatically use multiple cores
2. **Model training**: Deep learning models spawn multiple threads
3. **No core limit set**: The process can use as many cores as available
4. **After killing SMD**: More cores became available, so the process grabbed them!

---

## Comparison: Before vs After Killing SMD

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| anomaly-archive cores | ~34 | ~43 | **+9 cores** 🔺 |
| SMD cores | ~18 | ~15 | **-3 cores** (new process) |
| Total occupied | ~63 | ~64 | **+1 core** |
| Free cores | ~1 | 0 | **0 free now!** 🔴 |

**The anomaly-archive process EXPANDED to use the freed cores!**

---

## Recommendations

### Option 1: Let ramses-anomaly-archive Continue ⏳
- **Pro**: Don't lose 17+ hours of progress
- **Con**: Will take 177 days total at this rate (unacceptable!)
- **Decision**: Only if you're OK waiting months

### Option 2: Stop ramses-anomaly-archive 🛑
```bash
# Attach to screen
screen -r ramses-anomaly-archive

# Press Ctrl+C to stop

# Or kill directly
kill 1121046
```
**Why**: 177 days is way too long!

### Option 3: Limit Core Usage ⚙️
Use `taskset` to limit cores:
```bash
# Limit to 20 cores instead of 43
taskset -cp 0-19 1121046
```
This would:
- Free up ~23 cores for other work
- Slow down anomaly-archive slightly
- Allow other jobs to run

### Option 4: Enable Parallel Mode for Anomaly Archive ⚡
The entity processing is in sequential mode. To speed up:
```bash
# Stop current run
screen -r ramses-anomaly-archive
# Ctrl+C

# Restart with parallel mode
cd RAMSeS
./run_testbed.sh --dataset-list testbed/file_list/test_u_ucr_anomaly_archive.csv --parallel true
```

This could reduce per-entity time from 17h to 6-8h.

### Option 5: Process Subset of UCR Entities 📋
Instead of all 250+ entities, select a subset:
- Top 50 most interesting entities
- Representative sample
- Reduce total time by 80%

---

## Immediate Action Required?

**Question**: Is 177 days acceptable for the UCR Anomaly Archive testbed?

- **If YES**: Let it run, no action needed
- **If NO**: You need to either:
  1. Stop it and use a subset of entities
  2. Enable parallel mode to speed it up
  3. Limit its core usage so other work can run

**My recommendation**: Stop it and use a subset of 50-100 UCR entities instead of all 250+.

---

## Summary

🔴 **ramses-anomaly-archive is now using 43 cores (67% of system)**
⏱️ **Running for 17+ hours on ONE entity**
📊 **Estimated total time: 177 days for all UCR entities**
✅ **SMD processes cleaned up successfully**
✅ **fatemeh's process untouched**
⚠️ **System 100% utilized, no free cores**

**Next step**: Decide if you want to stop anomaly-archive or let it continue for 6 months!
