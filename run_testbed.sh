#!/bin/bash
# RAMSeS Testbed Runner with Enhanced Logging
# This script runs the testbed with flexible model loading (skips missing models)

cd /home/maxoud/local-storage/projects/RAMSeS

echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║                      RAMSeS TESTBED RUNNER                                   ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "→ Working directory: $(pwd)"
echo "→ Python version: $(python --version 2>&1)"
echo "→ Start time: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# Run the testbed
python run_testbed_comprehensive.py \
    --dataset-list "$1" \
    --output-dir testbed_results \
    --timeout 360000 \
    2>&1 | tee testbed_run_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║                      TESTBED RUN COMPLETE                                    ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo "→ End time: $(date '+%Y-%m-%d %H:%M:%S')"
