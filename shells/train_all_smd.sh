#!/bin/bash
# train_all_smd.sh - Train models for SMD entities before running online analysis

set -e  # Exit on error

echo "════════════════════════════════════════════════════════════════"
echo "  OFFLINE PHASE: Training Models for SMD Dataset"
echo "════════════════════════════════════════════════════════════════"
echo ""

# Clean Python environment variables to prevent conflicts
unset PYTHONPATH
unset PYTHONHOME

# Activate conda environment
source /raid0_ssd1/anaconda3/etc/profile.d/conda.sh
conda activate RAMS

cd /home/maxoud/local-storage/projects/RAMSeS

# Read entities from CSV (skip header, remove quotes)
entities=$(tail -n +2 dataset_lists/smd_sample.csv | cut -d',' -f2 | tr -d '"')

echo "Entities to train: $entities"
echo ""

for entity in $entities; do
    echo "════════════════════════════════════════════════════════════════"
    echo "  Training SMD Entity: ${entity}"
    echo "════════════════════════════════════════════════════════════════"
    echo ""
    
    # Check if models already exist
    model_dir="./Mononito/trained_models/smd/${entity}"
    
    start_time=$(date +%s)
    
    # Note: All training parameters (paths, overwrite, model list, etc.) are read from Configs/config.yml
    # Only pass dataset and entity as command-line arguments
    python3 app.py \
        --dataset smd \
        --entity ${entity} \
        2>&1 | tee "logs/train_smd_${entity}.log"
    
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
    model_count=$(ls "./Mononito/trained_models/smd/${entity}"/*.pth 2>/dev/null | wc -l || echo "0")
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
echo "  ./shells/run_smd_all_analyses.sh"
echo ""
echo "Expected online phase runtime: ~3-4 hours (SMD is large!)"
echo "════════════════════════════════════════════════════════════════"
