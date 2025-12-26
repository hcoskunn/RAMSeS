# Quick Start: Updated Configuration System

## What Changed?

All hardcoded paths and dataset/entity values have been moved to configuration files. You now have a flexible, centralized configuration system!

## What You Need to Do

### 1. Update Your Config File

Edit `Configs/custom_config.yml` (or `Configs/config.yml`) with your paths:

```yaml
# Dataset and Entity Configuration
# NOTE: Dataset names are case-insensitive (SKAB, skab, Skab all work)
dataset: 'SKAB'
entity: '2'

# Update these paths to match your system
trained_model_path: "/home/maxoud/local-storage/projects/RAMSeS/Mononito/trained_models"
dataset_path: "/home/maxoud/local-storage/projects/RAMSeS/Mononito/datasets"
results_path: "/home/maxoud/local-storage/projects/RAMSeS/Mononito/results"
```

### 2. Validate Your Configuration

Run the validation script to check everything is set up correctly:

```bash
python validate_config.py -c Configs/custom_config.yml
```

### 3. Run Your Application

```bash
# Use default config
python app.py

# Or specify a custom config
python app.py -c Configs/custom_config.yml
```

## Key Benefits

✓ **No more hardcoded paths** - Everything in one config file  
✓ **Easy to switch environments** - Just change the config file  
✓ **Process multiple entities** - Set `entity: ['1', '2', '3']` in config  
✓ **Better error messages** - Know exactly which dataset/entity failed  

## Example Configurations

### Single Dataset/Entity
```yaml
dataset: 'SKAB'  # Case-insensitive
entity: '2'
```

### Multiple Entities
```yaml
dataset: 'SKAB'
entity: ['1', '2', '3']  # Will process all three
```

### Multiple Datasets (Advanced)
```yaml
dataset: ['SKAB', 'SMD']  # Case-insensitive
entity: ['1', '2']  # Will process all combinations
```

## Files You Can Now Customize

- `Configs/config.yml` - Default configuration
- `Configs/custom_config.yml` - Your local customization

## Detailed Documentation

- **CONFIG_GUIDE.md** - Complete configuration guide with examples
- **CONFIGURATION_REFACTORING.md** - Technical details of all changes
- **validate_config.py** - Script to validate your configuration

## Need Help?

1. Run `python validate_config.py -c Configs/custom_config.yml` to check your config
2. Check `CONFIG_GUIDE.md` for detailed examples
3. Look at error messages - they now include dataset/entity context

## What If I Get Errors?

### "FileNotFoundError: Config file not found"
→ Check your config file path, use `-c Configs/custom_config.yml`

### "Dataset file not found"
→ Verify `dataset_path` in your config and ensure data files exist  
→ Dataset names are case-insensitive (SKAB, skab, etc. all work)

### "Model directory not found"
→ Check `trained_model_path` in config and train models first

---

**Ready to go!** Just update your config file and run `python app.py -c Configs/custom_config.yml`
