#!/bin/bash
#
# Quickstart Script for Online Phase Analysis
# This script runs online phase analysis for RAMSeS framework
#

set -e  # Exit on error

echo "=========================================="
echo "RAMSeS Online Phase Analysis - Quickstart"
echo "=========================================="
echo ""

# Default configuration
DATA_DIR="${DATA_DIR:-./Mononito}"
TRAINED_MODEL_DIR="${TRAINED_MODEL_DIR:-./Mononito/trained_models}"
OUTPUT_DIR="${OUTPUT_DIR:-online_phase_results}"
NUM_WINDOWS="${NUM_WINDOWS:-50}"
NUM_ENTITIES="${NUM_ENTITIES:-3}"

# Check if dataset list is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <dataset_list_csv> [options]"
    echo ""
    echo "Examples:"
    echo "  $0 testbed/file_list/test_m_skab.csv"
    echo "  $0 testbed/file_list/test_m_smd.csv"
    echo "  $0 testbed/file_list/ucr_sample_10.csv"
    echo ""
    echo "Environment variables:"
    echo "  DATA_DIR=$DATA_DIR"
    echo "  TRAINED_MODEL_DIR=$TRAINED_MODEL_DIR"
    echo "  OUTPUT_DIR=$OUTPUT_DIR"
    echo "  NUM_WINDOWS=$NUM_WINDOWS"
    echo "  NUM_ENTITIES=$NUM_ENTITIES"
    echo ""
    exit 1
fi

DATASET_LIST="$1"

# Check if dataset list exists
if [ ! -f "$DATASET_LIST" ]; then
    echo "ERROR: Dataset list not found: $DATASET_LIST"
    exit 1
fi

echo "Configuration:"
echo "  Dataset List:       $DATASET_LIST"
echo "  Data Directory:     $DATA_DIR"
echo "  Trained Models:     $TRAINED_MODEL_DIR"
echo "  Output Directory:   $OUTPUT_DIR"
echo "  Windows per Entity: $NUM_WINDOWS"
echo "  Entities per Domain: $NUM_ENTITIES"
echo ""

# Check Python dependencies
echo "Checking dependencies..."
python3 -c "import psutil, matplotlib, numpy, pandas, torch" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "ERROR: Missing dependencies. Please install:"
    echo "  pip install psutil matplotlib numpy pandas torch"
    exit 1
fi
echo "✓ All dependencies found"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run analysis
echo "=========================================="
echo "Starting Online Phase Analysis"
echo "=========================================="
echo ""

python3 online_phase_analysis.py \
    --dataset-list "$DATASET_LIST" \
    --data-dir "$DATA_DIR" \
    --trained-model-dir "$TRAINED_MODEL_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --num-windows "$NUM_WINDOWS" \
    --num-entities "$NUM_ENTITIES"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓ Analysis Complete!"
    echo "=========================================="
    echo ""
    echo "Results saved to: $OUTPUT_DIR"
    echo ""
    echo "View results:"
    echo "  - Per-entity:  $OUTPUT_DIR/{dataset}/{entity}/online_phase_summary.txt"
    echo "  - Aggregate:   $OUTPUT_DIR/aggregate/aggregate_report.txt"
    echo "  - Plots:       $OUTPUT_DIR/{dataset}/{entity}/*.png"
    echo ""
else
    echo ""
    echo "ERROR: Analysis failed with exit code $EXIT_CODE"
    exit $EXIT_CODE
fi
