#!/bin/bash
# Quick scalability analysis test for RAMSeS
# Tests how RAMSeS performs with different-sized model pools

echo "=============================================="
echo "RAMSeS Scalability Analysis - Quick Test"
echo "=============================================="

# Activate conda environment
echo "Activating RAMS conda environment..."
source /raid0_ssd1/anaconda3/etc/profile.d/conda.sh
conda activate RAMS

# Configuration
DATASET_LIST="quick_test_skab_only.csv"
DATA_DIR="./Mononito/datasets"
TRAINED_MODEL_DIR="./Mononito/trained_models"
OUTPUT_DIR="./scalability_results_quick"
NUM_WINDOWS=50
POOL_SIZES="3,5,8"

echo ""
echo "Configuration:"
echo "  Dataset list: $DATASET_LIST"
echo "  Data directory: $DATA_DIR"
echo "  Models directory: $TRAINED_MODEL_DIR"
echo "  Output directory: $OUTPUT_DIR"
echo "  Windows: $NUM_WINDOWS"
echo "  Pool sizes to test: $POOL_SIZES"
echo "  Python: $(which python3)"
echo ""

echo "Starting scalability analysis..."
echo "This will test:"
echo "  1. Pool of 3 models → Does RAMSeS work with limited choices?"
echo "  2. Pool of 5 models → Moderate pool size"
echo "  3. Pool of 8 models → Full pool, maximum flexibility"
echo ""
echo "Question answered: Does RAMSeS need 8 models or achieve good F1 with 3-5?"
echo ""

python3 online_phase_analysis.py \
    --dataset-list "$DATASET_LIST" \
    --data-dir "$DATA_DIR" \
    --trained-model-dir "$TRAINED_MODEL_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --scalability-analysis \
    --num-windows "$NUM_WINDOWS" \
    --num-models-range "$POOL_SIZES"

if [ $? -eq 0 ]; then
    echo ""
    echo "=============================================="
    echo "✓ Scalability analysis completed successfully!"
    echo ""
    echo "Results saved in: $OUTPUT_DIR"
    echo ""
    echo "View results:"
    echo "  cat $OUTPUT_DIR/scalability_analysis/SKAB/0/scalability_summary.txt"
    echo ""
    echo "Key findings:"
    echo "  → Shows F1 vs pool size (3, 5, 8 models)"
    echo "  → Identifies diminishing returns"
    echo "  → Recommends optimal pool size"
    echo "=============================================="
else
    echo ""
    echo "=============================================="
    echo "✗ Scalability analysis failed with exit code $?"
    echo "=============================================="
fi
