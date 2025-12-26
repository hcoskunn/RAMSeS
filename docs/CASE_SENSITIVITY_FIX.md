# Case Sensitivity Bug Fix and Entity Extraction

## Problem 1: Case Sensitivity

Testbed was taking 2+ hours to process 3 datasets that should have completed in seconds.

## Root Cause

**Case sensitivity mismatch** between result storage and result checking:

- **Storage**: Results were written to `myresults/comprehensive/skab/5/comprehensive_results_skab_5_iter0.txt` (lowercase domain)
- **Checking**: Testbed looked for `myresults/comprehensive/SKAB/5/comprehensive_results_SKAB_5_iter0.txt` (uppercase domain)
- **Result**: Testbed couldn't find existing results → re-ran app.py from scratch every time

## Timeline

- **User reported**: "2+ hours for 3 datasets vs 3 minutes manual execution"
- **Initial investigation**: Suspected duplicate GA execution → ruled out (only 1 call)
- **Second investigation**: Found sleep overhead (100ms per loop iteration) → fixed but not main issue
- **User insight**: "saving the results" → led to deeper investigation
- **Discovery**: Tested actual execution → found it was re-running app.py despite existing results
- **Final diagnosis**: Case sensitivity bug in line 232 of `run_testbed_comprehensive.py`

## Fix

Changed line 231-233 in `run_testbed_comprehensive.py`:

```python
# Before:
results_file = f"myresults/comprehensive/{domain}/{entity}/comprehensive_results_{domain}_{entity}_iter0.txt"

# After:
domain_lower = domain.lower()
results_file = f"myresults/comprehensive/{domain_lower}/{entity}/comprehensive_results_{domain_lower}_{entity}_iter0.txt"
```

## Verification

**Before fix**:
```bash
$ timeout 60 python run_testbed_comprehensive.py ...
# Ran for 60+ seconds, still executing GA generations
Command exited with code 124  # Timeout
```

**After fix**:
```bash
$ timeout 30 python run_testbed_comprehensive.py ...
Results already exist for SKAB_5.csv, parsing existing results...
Results already exist for SKAB_2.csv, parsing existing results...
Results already exist for SKAB_14.csv, parsing existing results...
Elapsed time: 0.01s  # ✅ INSTANT!
```

## Impact

- **Performance gain**: 2+ hours → 0.01 seconds (720,000× faster)
- **For new results**: First run still takes ~3-5 minutes (expected)
- **For existing results**: Now correctly skips execution

## Related Fixes

Also fixed sleep overhead in monitoring loop:
- Changed from unconditional `time.sleep(0.1)` to conditional `time.sleep(0.01)`
- Reduces overhead for new runs from ~90s to ~1-2s per dataset

## Lesson Learned

**Always test the actual behavior, not just the code logic!**

The code *looked* correct (existing results check was in place), but only by running it did we discover:
1. It wasn't skipping execution
2. The path construction had a case mismatch
3. User's intuition was right - it was related to result handling (just not the way we initially thought)

## Problem 2: Entity Extraction for SMD Datasets

### Issue
After fixing the case sensitivity bug, SMD datasets failed with:
```
NotADirectoryError: [Errno 20] Not a directory: '.../servermachinedataset/machine-1-1.txt'
```

### Root Cause
Entity extraction regex `r'_(\d+)\.csv$'` was designed for SKAB format (`SKAB_5.csv`), but SMD uses different format (`machine-1-1.txt`):
- SKAB: `SKAB_5.csv` → entity should be `5`
- SMD: `machine-1-1.txt` → entity should be `machine-1-1` (full filename without extension)

The fallback `dataset_file.replace('.csv', '')` only removed `.csv` extension, leaving `.txt` files with the extension.

### Fix
Updated entity extraction in `run_testbed_comprehensive.py` line 223-230:

```python
# Before:
entity_match = re.search(r'_(\d+)\.csv$', dataset_file)
if entity_match:
    entity = entity_match.group(1)
else:
    entity = dataset_file.replace('.csv', '')

# After:
entity_match = re.search(r'_(\d+)\.csv$', dataset_file)
if entity_match:
    entity = entity_match.group(1)
else:
    # Remove extension (.csv, .txt, etc.)
    entity = re.sub(r'\.(csv|txt)$', '', dataset_file, flags=re.IGNORECASE)
```

### Verification
```python
# SKAB: SKAB_5.csv -> entity: 5 ✅
# SMD: machine-1-1.txt -> entity: machine-1-1 ✅
# Command: python app.py --dataset servermachinedataset --entity machine-1-1 ✅
```

## Problem 3: SMD Dataset Memory Corruption

### Issue
After fixing entity extraction and data loader, SMD datasets crash with memory corruption:
```
malloc(): smallbin double linked list corrupted
app.py failed for machine-1-1.txt with exit code -6
```

### Root Cause
**NOT a testbed issue** - This is a bug in the RAMSeS framework itself (likely in Thompson Sampling module based on progress bars showing ~7% when crash occurs).

The error indicates memory management problems when processing large SMD datasets:
- SKAB datasets: ~500KB, work fine
- SMD datasets: ~9MB, trigger memory corruption

### Status
⚠️ **Framework Issue** - Requires debugging the core RAMSeS algorithms (Thompson Sampling, or possibly data preprocessing for large datasets).

### Fixes Applied to Testbed
1. ✅ **Datasets/load.py line 125-148**: Added `.txt` file support to `load_any_dataset()`
2. ✅ **run_testbed_comprehensive.py line 230**: Fixed entity extraction for non-CSV files

### Testbed Now Works For:
- ✅ SKAB datasets (CSV format, ~500KB)
- ❌ SMD datasets (TXT format, ~9MB) - **Framework crashes, not testbed issue**

### Recommendation
The SMD crashes need to be fixed in the core RAMSeS framework before the testbed can successfully process them. The testbed itself is now correctly configured.
