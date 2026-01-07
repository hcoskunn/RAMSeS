#!/bin/bash
################################################################################
# run_complete_pipeline.sh - Complete RAMSeS Pipeline (All Datasets)
#
# This script runs the COMPLETE workflow for all 3 datasets:
#   1. SKAB   (11D multivariate, ~1,146 rows, 5 entities)
#   2. SMD    (39D multivariate, ~23,697 rows, 8 entities) 
#   3. UCR    (1D univariate, variable size, 5 entities)
#
# For each dataset:
#   STEP 1: Train models (offline phase on 80% data)
#   STEP 2: Check training status
#   STEP 3: Run online analysis (on 20% data)
#
# Total estimated time: 10-15 hours
#   - SKAB:  2-5 hours training + 70 min analysis
#   - SMD:   4-8 hours training + 3-4 hours analysis
#   - UCR:   2-5 hours training + 75 min analysis
#
# Run in screen/tmux recommended!
################################################################################

set -e  # Exit on error

echo "════════════════════════════════════════════════════════════════════════════════"
echo "  RAMSeS COMPLETE PIPELINE - ALL 3 DATASETS"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""
echo "This will run the complete workflow for:"
echo "  1. SKAB Dataset   (Multivariate industrial sensors)"
echo "  2. SMD Dataset    (Highly multivariate server metrics)"
echo "  3. UCR Dataset    (Univariate diverse time series)"
echo ""
echo "For EACH dataset:"
echo "  → Train models on 80% of data (offline phase)"
echo "  → Run adaptive online analysis on 20% of data"
echo ""
echo "⚠️  TOTAL ESTIMATED TIME: 10-15 HOURS"
echo ""
echo "Recommendation: Run in screen/tmux!"
echo "  screen -S ramses"
echo "  ./shells/run_complete_pipeline.sh"
echo "  Ctrl+A, D to detach"
echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""

echo "Starting pipeline automatically..."
echo ""

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

# Track overall start time
overall_start=$(date +%s)

################################################################################
# DATASET 1: SKAB
################################################################################

echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
echo "  DATASET 1/3: SKAB (Skoltech Anomaly Benchmark)"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""

skab_start=$(date +%s)

# STEP 1: Train SKAB models
echo "─────────────────────────────────────────────────────────────────────────────────"
echo "STEP 1/3: Training SKAB Models (Offline Phase)"
echo "─────────────────────────────────────────────────────────────────────────────────"
echo ""

./shells/train_all_skab.sh 2>&1 | tee logs/train_skab_$(date +%Y%m%d_%H%M%S).log

if [ $? -eq 0 ]; then
    echo "✅ SKAB training completed"
else
    echo "❌ SKAB training failed!"
    echo "Check logs/train_skab_*.log for details"
    exit 1
fi

# STEP 2: Check status
echo ""
echo "─────────────────────────────────────────────────────────────────────────────────"
echo "STEP 2/3: Checking SKAB Training Status"
echo "─────────────────────────────────────────────────────────────────────────────────"
echo ""

./shells/check_training_status.sh | grep -A 10 "SKAB Dataset"

# STEP 3: Run online analysis
echo ""
echo "─────────────────────────────────────────────────────────────────────────────────"
echo "STEP 3/3: Running SKAB Online Analysis"
echo "─────────────────────────────────────────────────────────────────────────────────"
echo ""

./shells/run_skab_all_analyses.sh 2>&1 | tee logs/analysis_skab_$(date +%Y%m%d_%H%M%S).log

if [ $? -eq 0 ]; then
    echo "✅ SKAB analysis completed"
else
    echo "❌ SKAB analysis failed!"
    echo "Check logs/analysis_skab_*.log for details"
    exit 1
fi

skab_end=$(date +%s)
skab_duration=$((skab_end - skab_start))

echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
echo "✅ SKAB DATASET COMPLETE!"
echo "   Time: ${skab_duration}s = $((skab_duration/60)) min = $((skab_duration/3600)) hr"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""

################################################################################
# DATASET 2: SMD
################################################################################

echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
echo "  DATASET 2/3: SMD (Server Machine Dataset)"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""

smd_start=$(date +%s)

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

smd_end=$(date +%s)
smd_duration=$((smd_end - smd_start))

echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
echo "✅ SMD DATASET COMPLETE!"
echo "   Time: ${smd_duration}s = $((smd_duration/60)) min = $((smd_duration/3600)) hr"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""

################################################################################
# DATASET 3: UCR
################################################################################

echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
echo "  DATASET 3/3: UCR Anomaly Archive"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""

ucr_start=$(date +%s)

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

ucr_end=$(date +%s)
ucr_duration=$((ucr_end - ucr_start))

echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
echo "✅ UCR DATASET COMPLETE!"
echo "   Time: ${ucr_duration}s = $((ucr_duration/60)) min = $((ucr_duration/3600)) hr"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""

################################################################################
# FINAL SUMMARY
################################################################################

overall_end=$(date +%s)
overall_duration=$((overall_end - overall_start))

echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
echo "  🎉 ALL DATASETS COMPLETE!"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""
echo "Timing Summary:"
echo "─────────────────────────────────────────────────────────────────────────────────"
echo "  SKAB:  ${skab_duration}s = $((skab_duration/60)) min = $((skab_duration/3600)) hr"
echo "  SMD:   ${smd_duration}s = $((smd_duration/60)) min = $((smd_duration/3600)) hr"
echo "  UCR:   ${ucr_duration}s = $((ucr_duration/60)) min = $((ucr_duration/3600)) hr"
echo "  ─────────────────────────────────────────────────────────────────────────────"
echo "  TOTAL: ${overall_duration}s = $((overall_duration/60)) min = $((overall_duration/3600)) hr"
echo ""
echo "Results locations:"
echo "─────────────────────────────────────────────────────────────────────────────────"
echo "  SKAB: ./results_skab/"
echo "  SMD:  ./results_smd/"
echo "  UCR:  ./results_ucr/"
echo ""
echo "Logs:"
echo "─────────────────────────────────────────────────────────────────────────────────"
echo "  Training: logs/train_*.log"
echo "  Analysis: logs/analysis_*.log"
echo ""
echo "Quick view results:"
echo "  find ./results_* -name '*_summary.txt' -exec cat {} \\;"
echo "  find ./results_* -name '*.png' | head -20"
echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
echo "  ✅ COMPLETE PIPELINE FINISHED SUCCESSFULLY!"
echo "════════════════════════════════════════════════════════════════════════════════"
