#!/bin/bash
################################################################################
# run_complete_ucr_pipeline.sh - Complete RAMSeS Pipeline (UCR Dataset Only)
#
# This script runs the COMPLETE workflow for UCR dataset:
#   UCR Anomaly Archive
#   - 1D univariate time series
#   - Variable size
#   - 5 entities
#
# Workflow:
#   STEP 1: Train models (offline phase on 80% data)
#   STEP 2: Check training status
#   STEP 3: Run online analysis (on 20% data)
#
# Total estimated time: 3-6 hours
#   - Training: 2-5 hours
#   - Analysis: 75 minutes
#
# Run in screen/tmux recommended!
################################################################################

set -e  # Exit on error

echo "════════════════════════════════════════════════════════════════════════════════"
echo "  RAMSeS COMPLETE PIPELINE - UCR DATASET"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""
echo "Dataset: UCR Anomaly Archive"
echo "  - Type: Univariate diverse time series"
echo "  - Dimensions: 1"
echo "  - Size: Variable"
echo "  - Entities: 5"
echo ""
echo "Workflow:"
echo "  → Train models on 80% of data (offline phase)"
echo "  → Run adaptive online analysis on 20% of data"
echo ""
echo "⚠️  ESTIMATED TIME: 3-6 HOURS"
echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""

read -p "Press ENTER to start, or Ctrl+C to cancel..."

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
# UCR DATASET PIPELINE
################################################################################

# STEP 1: Train UCR models
echo "─────────────────────────────────────────────────────────────────────────────────"
echo "STEP 1/3: Training UCR Models (Offline Phase)"
echo "─────────────────────────────────────────────────────────────────────────────────"
echo ""

./shells/train_all_ucr.sh 2>&1 | tee logs/train_ucr_$(date +%Y%m%d_%H%M%S).log

if [ $? -eq 0 ]; then
    echo "✅ UCR training completed"
else
    echo "❌ UCR training failed!"
    echo "Check logs/train_ucr_*.log for details"
    exit 1
fi

# STEP 2: Check status
echo ""
echo "─────────────────────────────────────────────────────────────────────────────────"
echo "STEP 2/3: Checking UCR Training Status"
echo "─────────────────────────────────────────────────────────────────────────────────"
echo ""

./shells/check_training_status.sh | grep -A 10 "UCR Anomaly Archive"

# STEP 3: Run online analysis
echo ""
echo "─────────────────────────────────────────────────────────────────────────────────"
echo "STEP 3/3: Running UCR Online Analysis"
echo "─────────────────────────────────────────────────────────────────────────────────"
echo ""

./shells/run_ucr_all_analyses.sh 2>&1 | tee logs/analysis_ucr_$(date +%Y%m%d_%H%M%S).log

if [ $? -eq 0 ]; then
    echo "✅ UCR analysis completed"
else
    echo "❌ UCR analysis failed!"
    echo "Check logs/analysis_ucr_*.log for details"
    exit 1
fi

################################################################################
# FINAL SUMMARY
################################################################################

end_time=$(date +%s)
duration=$((end_time - start_time))

echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
echo "  🎉 UCR PIPELINE COMPLETE!"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""
echo "Total Time: ${duration}s = $((duration/60)) min = $((duration/3600)) hr $((duration%3600/60)) min"
echo ""
echo "Results location:"
echo "  ./results_ucr/"
echo ""
echo "Logs:"
echo "  Training: logs/train_ucr_*.log"
echo "  Analysis: logs/analysis_ucr_*.log"
echo ""
echo "Quick view results:"
echo "  find ./results_ucr -name '*_summary.txt' -exec cat {} \\;"
echo "  find ./results_ucr -name '*.png' | head -20"
echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
