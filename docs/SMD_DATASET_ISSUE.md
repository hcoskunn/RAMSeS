# SMD Dataset Memory Corruption Issue

## Problem
When running RAMSeS on SMD (ServerMachineDataset) datasets, the framework crashes with memory corruption:

```
malloc(): smallbin double linked list corrupted
app.py failed for machine-1-1.txt with exit code -6
```

## When Does It Occur?
- **Crash happens at**: ~7% progress (during Thompson Sampling module based on progress bars)
- **Dataset size**: SMD files are ~9MB (vs SKAB ~500KB)
- **Exit code**: -6 (SIGABRT - memory corruption detected)

## What Was Fixed (Testbed Infrastructure)

### 1. Entity Extraction (run_testbed_comprehensive.py)
**Fixed**: Line 230 - Entity extraction now handles both `.csv` and `.txt` extensions
```python
entity = re.sub(r'\.(csv|txt)$', '', dataset_file, flags=re.IGNORECASE)
```

### 2. Data Loader (Datasets/load.py)
**Fixed**: Line 125-148 - `load_any_dataset()` now checks for `.txt` files
```python
elif os.path.isfile(f'{data_path}.txt'):
    dataset = load_csv_file(f'{data_path}.txt', entity, group, normalize, verbose)
```

## What Still Needs Fixing (Core Framework)

### Memory Corruption Bug
**Location**: Likely in Thompson Sampling or data preprocessing modules
**Symptom**: `malloc(): smallbin double linked list corrupted`
**Trigger**: Large datasets (~9MB)

### Possible Root Causes
1. **Buffer overflow** in numpy array operations
2. **Memory leak** in Thompson Sampling iterations (100 iterations × large dataset)
3. **Improper memory deallocation** in C extensions (numpy/scipy)
4. **Stack overflow** from recursive operations
5. **Double-free** or **use-after-free** in data structures

### Debugging Steps
1. Run with memory profiler:
   ```bash
   python -m memory_profiler app.py --dataset servermachinedataset --entity machine-1-1
   ```

2. Run with address sanitizer (if using compiled extensions):
   ```bash
   ASAN_OPTIONS=detect_leaks=1 python app.py --dataset servermachinedataset --entity machine-1-1
   ```

3. Use gdb to catch the crash:
   ```bash
   gdb --args python app.py --dataset servermachinedataset --entity machine-1-1
   ```

4. Check numpy/scipy versions for known issues:
   ```bash
   pip list | grep -E 'numpy|scipy'
   ```

5. Test with smaller subset of SMD data to isolate size threshold

## Workaround
For now, use the testbed only with SKAB datasets until the memory corruption bug is fixed:
```bash
python run_testbed_comprehensive.py --dataset-list testbed/file_list/test_m_skab.csv
```

## Status
- ✅ Testbed infrastructure: **FIXED** - All path handling and file loading works correctly
- ❌ Core framework: **BUG EXISTS** - Memory corruption prevents SMD dataset processing
- 🔧 Next step: Debug the core RAMSeS framework to fix memory management issues
