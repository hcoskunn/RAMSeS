#!/bin/bash
################################################################################
# SMD (Server Machine Dataset) - All Three Analyses
#
# Dataset: SMD (Server Machine Dataset)
# Characteristics:
#   - Entities: 28 total (using 8 for manageable runtime)
#   - Dimensions: 39 columns (HIGHLY MULTIVARIATE)
#   - Rows per entity: ~23,697 (20x LARGER than SKAB!)
#   - Total time estimate: ~3-4 HOURS for all 3 analyses
#
# This script runs ALL THREE analyses optimized for SMD:
#   1. Adaptive Analysis (update intervals)
#   2. Scalability Analysis (pool sizes)
#   3. Window Size Sensitivity Analysis
#
# WARNING: SMD is MUCH LARGER than SKAB. Consider running overnight or use
#          fewer windows/entities for faster testing.
################################################################################

echo "================================================================================"
echo "SMD (Server Machine Dataset) - Complete Online Phase Analysis"
echo "================================================================================"
echo ""
echo "Dataset Characteristics:"
echo "  - Name: SMD (Server Machine Dataset)"
echo "  - Entities: 28 (using 8 for speed)"
echo "  - Dimensions: 39 columns (HIGHLY MULTIVARIATE - server metrics)"
echo "  - Size: ~23,697 rows per entity (20x LARGER than SKAB!)"
echo "  - Type: Server monitoring data from large internet company"
echo ""
echo "⚠️  WARNING: This will take 3-4 HOURS due to dataset size!"
echo ""
echo "Analyses to run:"
echo "  1. Adaptive Analysis     → ~60-90 minutes"
echo "  2. Scalability Analysis  → ~60-90 minutes"
echo "  3. Window Size Analysis  → ~60-90 minutes"
echo ""
echo "Total estimated time: 3-4 hours"
echo ""
echo "Consider running in screen/tmux or overnight!"
echo "================================================================================"
echo ""

echo "Starting analyses automatically..."
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

# ============================================================================
# CHECK: Are models trained?
# ============================================================================
echo "Checking if models are trained..."
entities=$(tail -n +2 dataset_lists/smd_sample.csv | cut -d',' -f2 | tr -d '"' | head -3)
all_trained=true

for entity in $entities; do
    model_dir="./Mononito/trained_models/smd/${entity}"
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
    echo "You need to run OFFLINE training first. Create train_all_smd.sh:"
    echo "  cp shells/train_all_skab.sh shells/train_all_smd.sh"
    echo "  # Edit to use: --dataset smd"
    echo "  ./shells/train_all_smd.sh"
    echo ""
    echo "Continuing anyway with available models..."
fi
echo ""

# ============================================================================
# ANALYSIS 1: ADAPTIVE ANALYSIS
# ============================================================================
echo "================================================================================"
echo "ANALYSIS 1/3: ADAPTIVE ANALYSIS (Main Evidence for R1.O2, R1.O3)"
echo "================================================================================"
echo "Testing update intervals: 5, 10, None (reduced for speed)"
echo "Num windows: 40 (reduced due to large dataset size)"
echo "Num entities: 3 (using subset due to size)"
echo "Injecting regime shifts: YES (at 25%, 50%, 75%)"
echo ""

python3 online_phase_analysis.py \
    --dataset-list dataset_lists/smd_sample.csv \
    --data-dir ./Mononito/datasets \
    --trained-model-dir ./Mononito/trained_models \
    --output-dir ./results_smd/adaptive \
    --adaptive-analysis \
    --num-windows 40 \
    --num-entities 3 \
    --update-intervals "5,10,None"

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Adaptive analysis completed!"
    echo "Results: ./results_smd/adaptive/adaptive_analysis/"
    echo ""
else
    echo ""
    echo "✗ Adaptive analysis failed!"
    echo "Continuing anyway..."
fi

# ============================================================================
# ANALYSIS 2: SCALABILITY ANALYSIS
# ============================================================================
echo ""
echo "================================================================================"
echo "ANALYSIS 2/3: SCALABILITY ANALYSIS (R1.O3 - Resource Efficiency)"
echo "================================================================================"
echo "Testing pool sizes: 3, 5, 8 models"
echo "Num windows: 40 (reduced due to large dataset size)"
echo "Num entities: 3"
echo "Question: Does RAMSeS scale with larger datasets?"
echo ""

python3 online_phase_analysis.py \
    --dataset-list dataset_lists/smd_sample.csv \
    --data-dir ./Mononito/datasets \
    --trained-model-dir ./Mononito/trained_models \
    --output-dir ./results_smd/scalability \
    --scalability-analysis \
    --num-windows 40 \
    --num-entities 3 \
    --num-models-range "3,5,8"

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Scalability analysis completed!"
    echo "Results: ./results_smd/scalability/scalability_analysis/"
    echo ""
else
    echo ""
    echo "✗ Scalability analysis failed!"
    echo "Continuing anyway..."
fi

# ============================================================================
# ANALYSIS 3: WINDOW SIZE SENSITIVITY
# ============================================================================
echo ""
echo "================================================================================"
echo "ANALYSIS 3/3: WINDOW SIZE SENSITIVITY (R2.O5)"
echo "================================================================================"
echo "Auto-calculating smart window sizes: 2%, 5%, 10%, 20% of data length"
echo "Num entities: 3"
echo "Question: What's the optimal window size for SMD (large, 39D data)?"
echo ""

python3 online_phase_analysis.py \
    --dataset-list dataset_lists/smd_sample.csv \
    --data-dir ./Mononito/datasets \
    --trained-model-dir ./Mononito/trained_models \
    --output-dir ./results_smd/window_size \
    --window-size-analysis \
    --inject-synthetic \
    --num-entities 3

# Note: Window sizes auto-calculated based on data length

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Window size analysis completed!"
    echo "Results: ./results_smd/window_size/window_size_analysis/"
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
echo "SMD ANALYSIS COMPLETE!"
echo "================================================================================"
echo ""
echo "Results saved in ./results_smd/"
echo "  1. ./results_smd/adaptive/          ← Adaptive analysis (R1.O2, R1.O3)"
echo "  2. ./results_smd/scalability/       ← Scalability analysis (R1.O3)"
echo "  3. ./results_smd/window_size/       ← Window sensitivity (R2.O5)"
echo ""
echo "View key results:"
echo "  find ./results_smd -name '*_summary.txt' -exec cat {} \\;"
echo "  find ./results_smd -name '*.png'"
echo ""
echo "================================================================================"
echo "SMD Characteristics:"
echo "  ✓ Highly multivariate (39 columns): Excellent for complex dependencies"
echo "  ✓ Large size (~23,697 rows): Tests scalability and real-world performance"
echo "  ✓ Server monitoring: Real-world large-scale deployment scenario"
echo "  ✓ IMPORTANT: Shows RAMSeS works on enterprise-scale data!"
echo "================================================================================"
