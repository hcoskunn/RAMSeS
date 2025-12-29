#!/bin/bash
# RAMSeS Testbed Runner with Enhanced Logging
# This script runs the testbed with flexible model loading (skips missing models)
# Usage: ./run_testbed.sh <dataset_list> [--parallel true|false]

cd /home/maxoud/local-storage/projects/RAMSeS

echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║                      RAMSeS TESTBED RUNNER                                   ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "→ Working directory: $(pwd)"
echo "→ Python version: $(python --version 2>&1)"
echo "→ Start time: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# Parse arguments
DATASET_LIST="$1"
shift  # Remove first argument

# Default to sequential mode
PARALLEL_MODE="false"

# Check for --parallel flag
while [[ $# -gt 0 ]]; do
    case $1 in
        --parallel)
            PARALLEL_MODE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            shift
            ;;
    esac
done

echo "→ Dataset list: $DATASET_LIST"
echo "→ Parallel mode: $PARALLEL_MODE"
echo ""

# Run the testbed
python run_testbed_comprehensive.py \
    --dataset-list "$DATASET_LIST" \
    --output-dir testbed_results \
    --timeout 360000 \
    --parallel "$PARALLEL_MODE" \
    2>&1 | tee testbed_run_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║                      TESTBED RUN COMPLETE                                    ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo "→ End time: $(date '+%Y-%m-%d %H:%M:%S')"
