#!/bin/bash
# train_all_ucr.sh - Train models for UCR entities before running online analysis

set -e  # Exit on error

echo "════════════════════════════════════════════════════════════════"
echo "  OFFLINE PHASE: Training Models for UCR Anomaly Archive"
echo "════════════════════════════════════════════════════════════════"
echo ""

# Activate conda environment
unset PYTHONPATH
unset PYTHONHOME
source /raid0_ssd1/anaconda3/etc/profile.d/conda.sh
conda activate RAMS

cd /home/maxoud/local-storage/projects/RAMSeS

# Read entities from CSV (skip header, remove quotes)
entities=$(tail -n +2 dataset_lists/ucr_sample.csv | cut -d',' -f2 | tr -d '"')

echo "Entities to train: $entities"
echo ""

for entity in $entities; do
    echo "════════════════════════════════════════════════════════════════"
    echo "  Training UCR Entity: ${entity}"
    echo "════════════════════════════════════════════════════════════════"
    echo ""
    
    # Check if models already exist
    model_dir="./Mononito/trained_models/anomaly_archive/${entity}"
    
    start_time=$(date +%s)
    
    # Note: All training parameters (paths, overwrite, model list, etc.) are read from Configs/config.yml
    # Only pass dataset and entity as command-line arguments
    python3 app.py \
        --dataset anomaly_archive \
        --entity ${entity} \
        2>&1 | tee "logs/train_ucr_${entity}.log"
    
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    
    echo ""
    echo "✅ Entity ${entity} training complete! (${duration}s = $((duration/60)) min)"
    echo "   Models saved to: ${model_dir}/"
    echo "   Models trained: $(ls $model_dir/*.pth 2>/dev/null | wc -l)"
    echo ""
    
done

echo "════════════════════════════════════════════════════════════════"
echo "  ✅ ALL TRAINING COMPLETE!"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "Trained models summary:"
for entity in $entities; do
    model_count=$(ls "./Mononito/trained_models/anomaly_archive/${entity}"/*.pth 2>/dev/null | wc -l || echo "0")
    if [ "$model_count" -ge 7 ]; then
        echo "  ✅ Entity ${entity}: ${model_count} models"
    else
        echo "  ⚠️  Entity ${entity}: ${model_count} models (expected 7+)"
    fi
done
echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  READY FOR ONLINE PHASE!"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "Run the online analysis with:"
echo "  ./shells/run_ucr_all_analyses.sh"
echo ""
echo "Expected online phase runtime: ~75 minutes"
echo "════════════════════════════════════════════════════════════════"
