# Current CPU Status - UCR Anomaly Archive Running

**Updated**: December 26, 2025
**Action**: Killed SMD machine-1-7 process (PID 1124916)
**Decision**: Keep UCR Anomaly Archive running ✅

---

## Current System Status

### Overall CPU
- **Total cores**: 64
- **CPU utilization**: 96.6% user
- **Idle**: 2.2% ≈ **~1-2 cores FREE** 🟢
- **Status**: Nearly full, but some breathing room

---

## Active Processes

### 1. 🎯 ramses-anomaly-archive (PID 1121046) - PRIMARY JOB
- **CPU**: 4377% ≈ **~43-44 cores** (67% of system)
- **Memory**: 3.1 GB
- **Runtime**: 1380 minutes = **23 hours 0 minutes**
- **Command**: `python app.py --dataset anomaly_archive --entity 077_UCR_Anomaly_DISTORTEDresperation11_58000_110800_110801 --parallel false`
- **Screen**: ramses-anomaly-archive
- **Status**: Still processing same UCR entity (running for almost 1 full day!)

### 2. 🆕 New SMD Process (PID 1133627) - JUST STARTED
- **CPU**: 1339% ≈ **~13 cores** (20% of system)
- **Memory**: 2.0 GB
- **Runtime**: 14 minutes (just started!)
- **Command**: `python app.py --dataset servermachinedataset --entity machine-1-8 --parallel true`
- **Status**: New entity started after you killed machine-1-7

### 3. ⚪ fatemeh's process (PID 213622)
- **CPU**: ~1 core
- **Status**: Still running, untouched ✅

### 4. 🔧 Background/System
- **CPU**: ~6 cores
- **Services**: VS Code, system services, etc.

---

## Core Distribution

| Process | Cores | Percentage | Status |
|---------|-------|------------|--------|
| **UCR Anomaly Archive** | **~44** | **69%** | 23 hours runtime |
| **SMD machine-1-8** | **~13** | **20%** | Just started (14 min) |
| fatemeh's job | ~1 | 2% | Untouched |
| Other/Background | ~4 | 6% | System |
| **FREE** | **~2** | **3%** | Available |

---

## Key Observations

### ✅ Good Status
1. **UCR Anomaly Archive is your priority** - running with 44 cores
2. **~2 cores free** - some headroom for system tasks
3. **New SMD process auto-started** on machine-1-8 (the testbed continues)
4. **System is stable** - no overload issues

### ⏱️ Timeline Update
**UCR Anomaly Archive Progress:**
- **Current entity**: 077_UCR_Anomaly_DISTORTEDresperation11_58000_110800_110801
- **Time on this entity**: 23 hours
- **Still not finished!** (this entity is taking a VERY long time)

**Estimated completion:**
- If this entity takes 24-30 hours total
- And you have 250+ entities
- **Total time: 24h × 250 = 6,000 hours = 250 days (~8 months)** 😱

---

## What's Happening with SMD?

You killed machine-1-7 (PID 1124916), but a **new process started** for machine-1-8:
- This means the testbed runner (`run_testbed_comprehensive.py`) is still running
- It automatically moves to the next entity when one is killed
- It will keep spawning new processes for each SMD entity

### If You Want to Stop SMD Completely:
You need to kill the **parent testbed runner** process, not just the individual app.py:

```bash
# Find the testbed runner
ps aux | grep "run_testbed" | grep smd

# Kill the parent process (run_testbed_comprehensive.py)
# This will stop it from spawning new entities
```

### If You're OK with SMD Running:
- Let it continue with ~13 cores
- It processes entities in parallel mode (faster)
- Won't interfere much with UCR Anomaly Archive (44 cores vs 13 cores)

---

## Recommendations

### Option 1: Keep Current Setup ⭐ RECOMMENDED
- **UCR Anomaly Archive**: 44 cores (your priority)
- **SMD**: 13 cores (continues in background)
- **Free**: 2 cores (system breathing room)
- **Pro**: Both jobs progress
- **Con**: UCR will still take ~8 months

### Option 2: Stop SMD Entirely, Give All Cores to UCR
```bash
# Find SMD testbed runner
ps aux | grep "run_testbed_comprehensive.py.*smd" | grep -v grep

# Kill it (will show you the PID)
kill <SMD_TESTBED_RUNNER_PID>
```
**Result**: UCR gets ~57 cores instead of 44 cores
**Speedup**: Marginal (maybe 20-30% faster)
**Worth it?**: Probably not - UCR is already using 44 cores efficiently

### Option 3: Limit UCR Cores, Free Up More for SMD
```bash
# Limit UCR to 30 cores
taskset -cp 0-29 1121046
```
**Result**: 
- UCR: 30 cores (slower but still works)
- SMD: Can use ~25-30 cores (faster)
- Better balance between jobs

### Option 4: Use a Subset of UCR Entities 🎯 BEST LONG-TERM
**Problem**: 250 days for all UCR entities is too long!

**Solution**: 
1. Stop current UCR run after this entity finishes
2. Create a subset file with 50-100 most important entities
3. Restart with subset
4. Reduces time from 250 days to 50-100 days

---

## Current Status: System is Healthy ✅

Your system is running fine:
- ✅ UCR Anomaly Archive has priority (44 cores)
- ✅ SMD continues in background (13 cores)
- ✅ ~2 cores free for system tasks
- ✅ No resource contention or crashes
- ⏱️ UCR will take ~8 months at current rate

---

## What to Monitor

### Check UCR Progress
```bash
# Attach to screen
screen -r ramses-anomaly-archive

# See current entity being processed
# Check logs for progress

# Detach: Ctrl+A then D
```

### Check if Entity Finally Finishes
This entity (077_UCR_Anomaly_DISTORTEDresperation11_58000_110800_110801) has been running for **23 hours**. 

When it finishes:
- Note the total time
- Next entity will start automatically
- You can estimate better timeline

### Check Free Cores
```bash
# Quick check
top -bn1 | grep "Cpu(s)"

# Detailed
htop  # (you're already running this)
```

---

## Summary

✅ **UCR Anomaly Archive is running with 44 cores (priority job)**
✅ **System has ~2 cores free (healthy)**
🔄 **New SMD process auto-started (machine-1-8) using 13 cores**
⏱️ **UCR entity runtime: 23 hours and counting...**
📊 **Estimated total UCR time: ~8 months** (if you run all 250+ entities)

**Your current setup is good!** The system is running smoothly with UCR as priority. The only concern is the **8-month timeline** - consider using a subset of UCR entities in future runs.

**No action needed right now** - let it run and monitor progress! 🎯
