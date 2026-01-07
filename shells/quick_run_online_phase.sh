#!/bin/bash
# Quick test run for online phase analysis
# Uses 3 entities per dataset, 20 windows per entity

echo "=============================================="
echo "RAMSeS Online Phase Analysis - Quick Test"
echo "=============================================="

# Activate conda environment
echo "Activating RAMS conda environment..."
source /raid0_ssd1/anaconda3/etc/profile.d/conda.sh
conda activate RAMS

# Check activation
if [ $? -ne 0 ]; then
    echo "WARNING: Failed to activate RAMS environment"
    echo "Continuing with base environment..."
fi

# Configuration
DATASET_LIST="quick_test_config.csv"
DATA_DIR="./Mononito"
TRAINED_MODEL_DIR="./Mononito/trained_models"
OUTPUT_DIR="./online_phase_results_quick"
NUM_WINDOWS=20

echo ""
echo "Configuration:"
echo "  Dataset list: $DATASET_LIST"
echo "  Data directory: $DATA_DIR"
echo "  Models directory: $TRAINED_MODEL_DIR"
echo "  Output directory: $OUTPUT_DIR"
echo "  Windows per entity: $NUM_WINDOWS"
echo "  Python: $(which python3)"
echo ""

# Check if dataset list exists
if [ ! -f "$DATASET_LIST" ]; then
    echo "ERROR: Dataset list not found: $DATASET_LIST"
    echo "Please create the dataset list file first."
    exit 1
fi

echo "Starting online phase analysis..."
echo ""

python3 online_phase_analysis.py \
    --dataset-list "$DATASET_LIST" \
    --data-dir "$DATA_DIR" \
    --trained-model-dir "$TRAINED_MODEL_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --num-windows $NUM_WINDOWS \
    --num-entities 3

EXIT_CODE=$?

echo ""
echo "=============================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Quick test completed successfully!"
    echo "Results saved in: $OUTPUT_DIR"
    echo ""
    echo "View aggregate results:"
    echo "  cat $OUTPUT_DIR/aggregate/aggregate_report.txt"
else
    echo "✗ Quick test failed with exit code $EXIT_CODE"
fi
echo "=============================================="

exit $EXIT_CODE
