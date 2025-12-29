# Active Screen Sessions Summary

Generated: December 26, 2025

## 🖥️ Screen Sessions Overview

### 1. 📺 ramses-anomaly-archive (Screen PID: 2421830)
- **Command**: `./run_testbed.sh testbed/file_list/test_u_ucr_anomaly_archive.csv`
- **Parallel Mode**: FALSE (sequential)
- **Current Entity**: `078_UCR_Anomaly_DISTORTEDresperation1_100000_110260_110412`
- **Process PID**: 44510
- **CPU Usage**: 3003% (30 cores)
- **Memory**: 3.4 GB
- **Runtime**: 1243 hours 34 minutes (running since start)
- **Status**: ✅ ACTIVE - Processing UCR Anomaly Archive

### 2. 📺 ramses-smd (Screen PID: 2167857)
- **Command**: `./run_testbed.sh testbed/file_list/test_u_ucr_anomaly_archive.csv`
- **Parallel Mode**: FALSE (sequential)
- **Current Entity**: `077_UCR_Anomaly_DISTORTEDresperation11_58000_110800_110801`
- **Process PID**: 3603338
- **CPU Usage**: 1397% (14 cores)
- **Memory**: 3.3 GB
- **Runtime**: 56 minutes 50 seconds
- **Started**: 19:35 (Dec 26)
- **Status**: ✅ ACTIVE - Processing UCR Anomaly Archive (NOT SMD!)

### 3. 📺 ramses-skab (Screen PID: 2168831)
- **Command**: None currently running
- **Status**: 💤 IDLE - No active testbed process

## 🆕 Recently Started (Not in named screens)

### A. PID 3491606 (pts/16)
- **Command**: `./run_testbed.sh testbed/file_list/test_m_smd.csv --parallel true`
- **Parallel Mode**: **TRUE** ⚡
- **Current Entity**: `machine-1-1`
- **Process PID**: 3492679
- **CPU Usage**: 1284% (13 cores)
- **Memory**: 1.5 GB
- **Runtime**: 62 minutes 5 seconds
- **Started**: 19:34
- **Status**: ✅ ACTIVE - Processing SMD with PARALLEL mode

### B. PID 3779429 (pts/14)
- **Command**: `python app.py --dataset servermachinedataset --entity machine-1-1 --parallel true`
- **Parallel Mode**: **TRUE** ⚡
- **CPU Usage**: 295% (3 cores)
- **Memory**: 2.0 GB
- **Runtime**: 73 minutes 23 seconds
- **Started**: 19:14
- **Status**: ✅ ACTIVE - Single entity test

### C. PID 4000004 (pts/16)
- **Command**: `python app.py --dataset servermachinedataset --entity machine-1-1 --parallel true`
- **Parallel Mode**: **TRUE** ⚡
- **CPU Usage**: 539% (5 cores)
- **Memory**: 1.9 GB
- **Runtime**: 125 minutes 44 seconds
- **Started**: 19:16
- **Status**: ✅ ACTIVE - Single entity test

## 📊 Summary

| Screen Name | Dataset | Parallel | CPU Cores | Memory | Runtime | Status |
|-------------|---------|----------|-----------|--------|---------|--------|
| ramses-anomaly-archive | UCR Anomaly Archive | No | 30 | 3.4 GB | 20+ days | ✅ Active |
| ramses-smd | UCR Anomaly Archive | No | 14 | 3.3 GB | ~1 hour | ✅ Active |
| ramses-skab | - | - | - | - | - | 💤 Idle |
| (unnamed) pts/16 | SMD | **Yes** ⚡ | 13 | 1.5 GB | ~1 hour | ✅ Active |
| (unnamed) pts/14 | SMD | **Yes** ⚡ | 3 | 2.0 GB | ~1.2 hours | ✅ Active |
| (unnamed) pts/16 | SMD | **Yes** ⚡ | 5 | 1.9 GB | ~2 hours | ✅ Active |

## 🎯 Key Findings

1. **ramses-smd screen is MISNAMED!** 
   - The screen is named "ramses-smd" but it's actually running UCR Anomaly Archive
   - The actual SMD testbed is running in an unnamed terminal (pts/16)

2. **Three parallel SMD tests running simultaneously**
   - All testing `machine-1-1` entity
   - Using parallel mode (new feature)
   - Total CPU: 13 + 3 + 5 = 21 cores combined

3. **Two UCR Anomaly Archive runs**
   - Both using sequential mode (old code)
   - Using 30 + 14 = 44 cores combined
   - One has been running for 20+ days!

## 💡 Recommendations

1. **Rename screens for clarity**:
   ```bash
   # The "ramses-smd" screen should be renamed:
   screen -S ramses-smd -X sessionname ramses-ucr-2
   ```

2. **Consolidate SMD parallel tests**:
   - You have 3 SMD processes all testing machine-1-1
   - Consider killing duplicates and keeping just one

3. **Check ramses-skab**:
   - This screen is idle - might want to start a SKAB testbed there

4. **Monitor the long-running UCR process**:
   - PID 44510 has been running for 20+ days
   - Might want to check if it's stuck or making progress
