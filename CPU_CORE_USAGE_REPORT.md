# CPU Core Usage Report

**Date**: December 26, 2025

## Core Usage Summary

### Total System
- **Total CPU cores**: 64
- **CPU utilization**: 97.8% user + 1.7% system = **99.5% BUSY**
- **Idle**: 0.6% (~0-1 cores free)
- **Python processes total**: 5138.5% CPU (equivalent to ~51 cores)

### Status: **SYSTEM IS FULLY LOADED** 🔴

Only **~1 core free** out of 64 cores!

---

## Top CPU Consumers

### 1. **ramses-anomaly-archive** (PID 1121046)
- **CPU**: 3453% (34-35 cores!)
- **Memory**: 3.0 GB
- **Runtime**: 7 hours 44 minutes
- **Command**: `python app.py --dataset anomaly_archive --entity 077_UCR_Anomaly_DISTORTEDresperation11_58000_110800_110801 --parallel false`
- **Screen**: ramses-anomaly-archive (2421830)
- **Status**: Running testbed via run_testbed.sh
- **Note**: Running in SEQUENTIAL mode (--parallel false)

### 2. **servermachinedataset** (PID 1119315)
- **CPU**: 1756% (17-18 cores)
- **Memory**: ~1.5 GB
- **Command**: `python app.py --dataset servermachinedataset --entity machine-1-1`
- **Status**: Still running (after you killed the duplicate)

### 3. **Other user process** (PID 213622)
- **CPU**: 99.9% (1 core)
- **User**: fatemeh
- **Command**: baran_not_enough_labels_lake.py
- **Status**: Not yours, another user's job

---

## Screen Session: ramses-anomaly-archive

### Process Tree
```
screen(2421830) ramses-anomaly-archive
  └─ bash(2421831)
      └─ run_testbed.sh(1120663)
          └─ python run_testbed_comprehensive.py(1120668)
              └─ python app.py(1121046) ← Main process using 34 cores
                  └─ 20+ threads
```

### What It's Doing
- Running UCR Anomaly Archive testbed
- Current entity: `077_UCR_Anomaly_DISTORTEDresperation11_58000_110800_110801`
- Using **SEQUENTIAL mode** (--parallel false)
- Has been running for **7 hours 44 minutes** on this entity alone!

### Why It's Using So Many Cores (Sequential Mode)
Even though it says `--parallel false`, it's still using 34 cores because:
1. Some algorithms internally use parallelism (numpy, sklearn, etc.)
2. PyOD models may use multiple threads
3. Data processing uses multiprocessing
4. Operating system thread scheduling

---

## Core Allocation Breakdown

| Process | Cores | Percentage | Status |
|---------|-------|------------|--------|
| ramses-anomaly-archive | ~34 | 53% | Running for 7h44m |
| servermachinedataset | ~18 | 28% | Running |
| fatemeh's process | ~1 | 2% | Other user |
| VS Code / Other | ~10 | 16% | Background |
| **Available** | **~1** | **1%** | Almost none |

---

## Time Estimates

### ramses-anomaly-archive Progress
- **Runtime so far**: 7 hours 44 minutes on ONE entity
- **Entity**: 077_UCR_Anomaly_DISTORTEDresperation11_58000_110800_110801
- **UCR Anomaly Archive**: Has 250+ entities
- **Sequential mode**: Processing one entity at a time

**Estimated total time for all UCR entities**:
```
If each entity takes ~7-8 hours:
250 entities × 7 hours = 1,750 hours = 73 days! 😱
```

This is VERY SLOW! The UCR Anomaly Archive is HUGE.

---

## Recommendations

### Immediate Actions

#### Option 1: Let It Run ⏳
- Keep current jobs running
- System is at max capacity
- Wait for jobs to finish
- No new jobs can start efficiently

#### Option 2: Check ramses-anomaly-archive Progress 🔍
```bash
# Attach to screen to see progress
screen -r ramses-anomaly-archive

# Check output/logs
tail -f output/anomaly_archive/077_UCR_Anomaly_DISTORTEDresperation11_58000_110800_110801/*.log
```

#### Option 3: Enable Parallel Mode for Anomaly Archive ⚡
The anomaly archive is running in **SEQUENTIAL mode**. If you enable parallel:
- Could reduce per-entity time by 2-3x
- But will use similar number of cores
- Total time: 73 days → 25-35 days

**To enable parallel mode**, you'd need to restart with `--parallel true`

#### Option 4: Stop Anomaly Archive (Risky) 🛑
If 73 days is too long:
```bash
# Stop the process
screen -r ramses-anomaly-archive
# Press Ctrl+C

# Or kill it
kill 1121046
```

**But**: You'll lose 7h44m of progress on current entity!

### Long-term Solutions

1. **Use Subset of UCR Anomaly Archive**
   - Don't process all 250 entities
   - Select top 50-100 most interesting ones
   - Save 80% of time

2. **Use Distributed Computing**
   - Run different entities on different machines
   - Cloud computing (AWS, GCP)
   - University cluster if available

3. **Optimize Per-Entity Time**
   - Reduce max_steps further (but quality suffers)
   - Skip some robustness tests
   - Use fewer models

4. **Parallelize Across Entities**
   - Modify testbed to process multiple entities simultaneously
   - Use 64 cores to run 2-3 entities in parallel
   - Each uses 20-25 cores

---

## Current Status Summary

### ✅ What's Working
- System is running at full capacity
- All jobs are progressing
- No crashes or errors visible

### ⚠️ Concerns
- **Only 1 core free** - system is maxed out
- **UCR Anomaly Archive will take 73 days** at current rate
- **No capacity for new jobs**

### 🎯 Recommendations
1. **Monitor ramses-anomaly-archive** - attach to screen and check progress
2. **Consider stopping if 73 days is too long**
3. **Or let it run over holiday break** while you're away
4. **Plan to use parallel mode** for future runs
5. **Consider subset of UCR entities** instead of all 250

---

## Quick Commands

```bash
# Check progress in ramses-anomaly-archive
screen -r ramses-anomaly-archive

# Detach without stopping: Ctrl+A then D

# Check all running Python processes
ps aux | grep "app.py" | grep -v grep

# Monitor CPU usage in real-time
htop  # (you're already running this)

# Check testbed progress
tail -f output/anomaly_archive/*/app.log
```

Would you like to:
1. Continue and let everything run?
2. Stop ramses-anomaly-archive (too slow)?
3. Check progress in more detail?
4. Adjust settings to speed up?
