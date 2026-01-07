# RAMSeS - Quick Run Guide

## Model Pool
**9 Base Models** (7 architectures with multiple instances):
- **Deep Learning**: LSTMVAE_1, DGHL_1, RNN_1, NN_1, NN_2
- **Classical**: LOF_1, LOF_2, CBLOF_1, MD_1

---

## Run Commands

### Individual Datasets
```bash
cd /home/maxoud/local-storage/projects/RAMSeS

# SKAB (3-6 hours)
./launch_skab_pipeline.sh

# SMD (7-12 hours)
./launch_smd_pipeline.sh

# UCR (3-6 hours)
./launch_ucr_pipeline.sh
```

### All Datasets
```bash
cd /home/maxoud/local-storage/projects/RAMSeS

# All 3 datasets sequentially (13-24 hours)
./launch_complete_pipeline.sh
```

### Recommended: Run in Screen
```bash
screen -S skab
cd /home/maxoud/local-storage/projects/RAMSeS
./launch_skab_pipeline.sh
# Ctrl+A, D to detach
# screen -r skab to reattach
```

---

## Configuration

### Force Retraining
Edit `Configs/config.yml` to control retraining behavior:
```yaml
overwrite: True   # Force retrain all models (ignores existing)
overwrite: False  # Skip already-trained models
```

### Training Parameters
All settings are in `Configs/config.yml`:
- `trained_model_path`: Where models are saved
- `dataset_path`: Where datasets are located
- `training_size`: Fraction of data to use (1.0 = 100%)
- `model_architectures`: Which models to train ('all' or specific list)

---

## Notes
- ✅ **Conda environment handled automatically** by launcher scripts
- ✅ All scripts activate `RAMS` environment before running
- ✅ Logs saved to `logs/train_*.log` and `logs/analysis_*.log`
- ✅ Results saved to `results_skab/`, `results_smd/`, `results_ucr/`
