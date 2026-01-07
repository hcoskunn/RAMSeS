#!/bin/bash
################################################################################
# SKAB Dataset - All Three Analyses
# 
# Dataset: SKAB (Skoltech Anomaly Benchmark)
# Characteristics:
#   - Entities: 16 (using subset for speed)
#   - Dimensions: 11 columns (MULTIVARIATE)
#   - Rows per entity: ~1,146
#   - Total time estimate: ~60-90 minutes for all 3 analyses
#
# This script runs ALL THREE analyses optimized for SKAB:
#   1. Adaptive Analysis (update intervals)
#   2. Scalability Analysis (pool sizes)
#   3. Window Size Sensitivity Analysis
################################################################################

echo "================================================================================"
echo "SKAB Dataset - Complete Online Phase Analysis"
echo "================================================================================"
echo ""
echo "Dataset Characteristics:"
echo "  - Name: SKAB (Skoltech Anomaly Benchmark)"
echo "  - Entities: 16 (using first 3 for speed)"
echo "  - Dimensions: 11 columns (MULTIVARIATE - vibration sensors, etc.)"
echo "  - Size: ~1,146 rows per entity"
echo "  - Type: Industrial control system data"
echo ""
echo "Analyses to run:"
echo "  1. Adaptive Analysis     → ~20 minutes"
echo "  2. Scalability Analysis  → ~25 minutes"
echo "  3. Window Size Analysis  → ~25 minutes"
echo ""
echo "Total estimated time: 70 minutes"
echo "================================================================================"
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

# ============================================================================
# CHECK: Are models trained?
# ============================================================================
echo "Checking if models are trained..."
entities=$(tail -n +2 dataset_lists/skab_full.csv | cut -d',' -f2 | tr -d '"' | head -3)
all_trained=true

for entity in $entities; do
    model_dir="./Mononito/trained_models/skab/${entity}"
    model_count=$(ls "${model_dir}"/*.pth 2>/dev/null | wc -l)
    if [ "$model_count" -lt 7 ]; then
        echo "⚠️  Entity ${entity}: Only ${model_count} models found (need 7+)"
        all_trained=false
    else
        echo "✓ Entity ${entity}: ${model_count} models found"
    fi
done

if [ "$all_trained" = false ]; then
    echo ""
    echo "════════════════════════════════════════════════════════════════"
    echo "ERROR: Models not fully trained!"
    echo "════════════════════════════════════════════════════════════════"
    echo ""
    echo "You need to run OFFLINE training first:"
    echo "  ./shells/train_all_skab.sh"
    echo ""
    echo "Or continue anyway? (y/n)"
    read -r response
    if [ "$response" != "y" ]; then
        exit 1
    fi
fi
echo ""

# ============================================================================
# ANALYSIS 1: ADAPTIVE ANALYSIS
# ============================================================================
echo "================================================================================"
echo "ANALYSIS 1/3: ADAPTIVE ANALYSIS (Main Evidence for R1.O2, R1.O3)"
echo "================================================================================"
echo "Testing update intervals: 5, 10, 20, None"
echo "Num windows: 50 (optimal for SKAB size)"
echo "Injecting regime shifts: YES (at 25%, 50%, 75%)"
echo "Injecting synthetic anomalies: YES (test data has no real anomalies)"
echo ""

python3 online_phase_analysis.py \
    --dataset-list dataset_lists/skab_full.csv \
    --data-dir ./Mononito/datasets \
    --trained-model-dir ./Mononito/trained_models \
    --output-dir ./results_skab/adaptive \
    --adaptive-analysis \
    --inject-synthetic \
    --num-windows 50 \
    --num-entities 3 \
    --update-intervals "5,10,20,None"

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Adaptive analysis completed!"
    echo "Results: ./results_skab/adaptive/adaptive_analysis/"
    echo ""
else
    echo ""
    echo "✗ Adaptive analysis failed!"
    echo "Continue anyway? (y/n)"
    read -r response
    if [ "$response" != "y" ]; then
        exit 1
    fi
fi

# ============================================================================
# ANALYSIS 2: SCALABILITY ANALYSIS
# ============================================================================
echo ""
echo "================================================================================"
echo "ANALYSIS 2/3: SCALABILITY ANALYSIS (R1.O3 - Resource Efficiency)"
echo "================================================================================"
echo "Auto-determining pool sizes: start at 3, then increments of 5 up to total"
echo "Num windows: 60 (more windows for better scalability evidence)"
echo "Injecting synthetic anomalies: YES"
echo "Question: Does RAMSeS need all models or work well with fewer?"
echo ""

python3 online_phase_analysis.py \
    --dataset-list dataset_lists/skab_full.csv \
    --data-dir ./Mononito/datasets \
    --trained-model-dir ./Mononito/trained_models \
    --output-dir ./results_skab/scalability \
    --scalability-analysis \
    --inject-synthetic \
    --num-windows 60 \
    --num-entities 3

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Scalability analysis completed!"
    echo "Results: ./results_skab/scalability/scalability_analysis/"
    echo ""
else
    echo ""
    echo "✗ Scalability analysis failed!"
    echo "Continue anyway? (y/n)"
    read -r response
    if [ "$response" != "y" ]; then
        exit 1
    fi
fi

# ============================================================================
# ANALYSIS 3: WINDOW SIZE SENSITIVITY
# ============================================================================
echo ""
echo "================================================================================"
echo "ANALYSIS 3/3: WINDOW SIZE SENSITIVITY (R2.O5)"
echo "================================================================================"
echo "Auto-calculating smart window sizes: 2%, 5%, 10%, 20% of data length"
echo "Injecting synthetic anomalies: YES"
echo "Question: What's the optimal window size for SKAB?"
echo ""

python3 online_phase_analysis.py \
    --dataset-list dataset_lists/skab_full.csv \
    --data-dir ./Mononito/datasets \
    --trained-model-dir ./Mononito/trained_models \
    --output-dir ./results_skab/window_size \
    --window-size-analysis \
    --inject-synthetic \
    --num-entities 3

# Note: Removed --window-sizes and --num-windows flags
# The analysis will auto-calculate appropriate window sizes based on data length

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Window size analysis completed!"
    echo "Results: ./results_skab/window_size/window_size_analysis/"
    echo ""
else
    echo ""
    echo "✗ Window size analysis failed!"
fi

# ============================================================================
# SUMMARY
# ============================================================================
echo ""
echo "================================================================================"
echo "SKAB ANALYSIS COMPLETE!"
echo "================================================================================"
echo ""
echo "Results saved in ./results_skab/"
echo "  1. ./results_skab/adaptive/          ← Adaptive analysis (R1.O2, R1.O3)"
echo "  2. ./results_skab/scalability/       ← Scalability analysis (R1.O3)"
echo "  3. ./results_skab/window_size/       ← Window sensitivity (R2.O5)"
echo ""
echo "View key results:"
echo "  find ./results_skab -name '*_summary.txt' -exec cat {} \\;"
echo "  find ./results_skab -name '*.png'"
echo ""
echo "================================================================================"
echo "SKAB Characteristics:"
echo "  ✓ Multivariate (11 columns): Good for testing feature interactions"
echo "  ✓ Small size (~1,146 rows): Fast analysis, good for quick validation"
echo "  ✓ Industrial data: Real-world control system anomalies"
echo "================================================================================"
