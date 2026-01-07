#!/bin/bash
# check_training_status.sh - Check which datasets have trained models

echo "════════════════════════════════════════════════════════════════"
echo "  TRAINING STATUS CHECK"
echo "════════════════════════════════════════════════════════════════"
echo ""

cd /home/maxoud/local-storage/projects/RAMSeS

# Check SKAB
echo "SKAB Dataset:"
echo "─────────────────────────────────────────────────────────────────"
if [ -f "dataset_lists/skab_full.csv" ]; then
    entities=$(tail -n +2 dataset_lists/skab_full.csv | cut -d',' -f2 | tr -d '"' | head -5)
    for entity in $entities; do
        model_dir="./Mononito/trained_models/skab/${entity}"
        model_count=$(ls "${model_dir}"/*.pth 2>/dev/null | wc -l || echo "0")
        if [ "$model_count" -ge 7 ]; then
            echo "  ✅ Entity ${entity}: ${model_count} models"
        elif [ "$model_count" -gt 0 ]; then
            echo "  ⚠️  Entity ${entity}: ${model_count} models (need 7+)"
        else
            echo "  ❌ Entity ${entity}: Not trained"
        fi
    done
else
    echo "  ⚠️  No skab_full.csv found"
fi
echo ""

# Check SMD
echo "SMD Dataset:"
echo "─────────────────────────────────────────────────────────────────"
if [ -f "dataset_lists/smd_sample.csv" ]; then
    entities=$(tail -n +2 dataset_lists/smd_sample.csv | cut -d',' -f2 | tr -d '"' | head -5)
    for entity in $entities; do
        model_dir="./Mononito/trained_models/smd/${entity}"
        model_count=$(ls "${model_dir}"/*.pth 2>/dev/null | wc -l || echo "0")
        if [ "$model_count" -ge 7 ]; then
            echo "  ✅ Entity ${entity}: ${model_count} models"
        elif [ "$model_count" -gt 0 ]; then
            echo "  ⚠️  Entity ${entity}: ${model_count} models (need 7+)"
        else
            echo "  ❌ Entity ${entity}: Not trained"
        fi
    done
else
    echo "  ⚠️  No smd_sample.csv found"
fi
echo ""

# Check UCR
echo "UCR Anomaly Archive:"
echo "─────────────────────────────────────────────────────────────────"
if [ -f "dataset_lists/ucr_sample.csv" ]; then
    entities=$(tail -n +2 dataset_lists/ucr_sample.csv | cut -d',' -f2 | tr -d '"' | head -5)
    for entity in $entities; do
        model_dir="./Mononito/trained_models/anomaly_archive/${entity}"
        model_count=$(ls "${model_dir}"/*.pth 2>/dev/null | wc -l || echo "0")
        if [ "$model_count" -ge 7 ]; then
            echo "  ✅ Entity ${entity}: ${model_count} models"
        elif [ "$model_count" -gt 0 ]; then
            echo "  ⚠️  Entity ${entity}: ${model_count} models (need 7+)"
        else
            echo "  ❌ Entity ${entity}: Not trained"
        fi
    done
else
    echo "  ⚠️  No ucr_sample.csv found"
fi
echo ""

echo "════════════════════════════════════════════════════════════════"
echo "  NEXT STEPS"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "To train missing models:"
echo "  ./shells/train_all_skab.sh  # SKAB"
echo "  ./shells/train_all_smd.sh   # SMD"
echo "  ./shells/train_all_ucr.sh   # UCR"
echo ""
echo "After training, run online analysis:"
echo "  ./shells/run_skab_all_analyses.sh"
echo "  ./shells/run_smd_all_analyses.sh"
echo "  ./shells/run_ucr_all_analyses.sh"
echo ""
echo "════════════════════════════════════════════════════════════════"
