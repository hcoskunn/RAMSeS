#!/bin/bash
######################echo "Workflow:"
echo "  → Train models on 80% of data (offline phase)"
echo "  → Run adaptive online analysis on 20% of data"
echo ""
echo "⚠️  ESTIMATED TIME: 10-15 HOURS"
echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""
echo "Starting pipeline automatically..."
echo ""

# Activate environment
echo ""
echo "Activating RAMS conda environment..."############################################
# run_complete_smd_pipeline.sh - Complete RAMSeS Pipeline (SMD Dataset Only)
#
# This script runs the COMPLETE workflow for SMD dataset:
#   SMD (Server Machine Dataset)
#   - 39D multivariate server metrics
#   - ~23,697 rows
#   - 8 entities
#
# Workflow:
#   STEP 1: Train models (offline phase on 80% data)
#   STEP 2: Check training status
#   STEP 3: Run online analysis (on 20% data)
#
# Total estimated time: 7-12 hours
#   - Training: 4-8 hours
#   - Analysis: 3-4 hours
#
# Run in screen/tmux recommended!
################################################################################

set -e  # Exit on error

echo "════════════════════════════════════════════════════════════════════════════════"
echo "  RAMSeS COMPLETE PIPELINE - SMD DATASET"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""
echo "Dataset: SMD (Server Machine Dataset)"
echo "  - Type: Highly multivariate server metrics"
echo "  - Dimensions: 39"
echo "  - Size: ~23,697 rows"
echo "  - Entities: 8"
echo ""
echo "Workflow:"
echo "  → Train models on 80% of data (offline phase)"
echo "  → Run adaptive online analysis on 20% of data"
echo ""
echo "⚠️  ESTIMATED TIME: 7-12 HOURS"
echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""

echo "Starting pipeline automatically..."
echo ""

# Clean Python environment variables to prevent conflicts
unset PYTHONPATH
unset PYTHONHOME

# Activate environment
echo ""
echo "Activating RAMS conda environment..."
source /raid0_ssd1/anaconda3/etc/profile.d/conda.sh
conda activate RAMS

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to activate RAMS environment"
    exit 1
fi

echo "✓ Environment activated"
echo ""

# Create logs directory
mkdir -p logs

# Track start time
start_time=$(date +%s)

################################################################################
# SMD DATASET PIPELINE
################################################################################

# STEP 1: Train SMD models
echo "─────────────────────────────────────────────────────────────────────────────────"
echo "STEP 1/3: Training SMD Models (Offline Phase)"
echo "─────────────────────────────────────────────────────────────────────────────────"
echo ""

./shells/train_all_smd.sh 2>&1 | tee logs/train_smd_$(date +%Y%m%d_%H%M%S).log

if [ $? -eq 0 ]; then
    echo "✅ SMD training completed"
else
    echo "❌ SMD training failed!"
    echo "Check logs/train_smd_*.log for details"
    exit 1
fi

# STEP 2: Check status
echo ""
echo "─────────────────────────────────────────────────────────────────────────────────"
echo "STEP 2/3: Checking SMD Training Status"
echo "─────────────────────────────────────────────────────────────────────────────────"
echo ""

./shells/check_training_status.sh | grep -A 10 "SMD Dataset"

# STEP 3: Run online analysis
echo ""
echo "─────────────────────────────────────────────────────────────────────────────────"
echo "STEP 3/3: Running SMD Online Analysis"
echo "─────────────────────────────────────────────────────────────────────────────────"
echo ""

./shells/run_smd_all_analyses.sh 2>&1 | tee logs/analysis_smd_$(date +%Y%m%d_%H%M%S).log

if [ $? -eq 0 ]; then
    echo "✅ SMD analysis completed"
else
    echo "❌ SMD analysis failed!"
    echo "Check logs/analysis_smd_*.log for details"
    exit 1
fi

################################################################################
# FINAL SUMMARY
################################################################################

end_time=$(date +%s)
duration=$((end_time - start_time))

echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
echo "  🎉 SMD PIPELINE COMPLETE!"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""
echo "Total Time: ${duration}s = $((duration/60)) min = $((duration/3600)) hr $((duration%3600/60)) min"
echo ""
echo "Results location:"
echo "  ./results_smd/"
echo ""
echo "Logs:"
echo "  Training: logs/train_smd_*.log"
echo "  Analysis: logs/analysis_smd_*.log"
echo ""
echo "Quick view results:"
echo "  find ./results_smd -name '*_summary.txt' -exec cat {} \\;"
echo "  find ./results_smd -name '*.png' | head -20"
echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
