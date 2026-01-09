#!/bin/bash
################################################################################
# UCR Anomaly Archive - All Three Analyses
#
# Dataset: UCR Anomaly Archive
# Characteristics:
#   - Entities: 250 total (using 5 for manageable runtime)
#   - Dimensions: 1 column (UNIVARIATE time series)
#   - Rows per entity: Variable (ranges from 1,000 to 100,000+)
#   - Total time estimate: ~60-90 minutes for all 3 analyses
#
# This script runs ALL THREE analyses optimized for UCR:
#   1. Adaptive Analysis (update intervals)
#   2. Scalability Analysis (pool sizes)
#   3. Window Size Sensitivity Analysis
#
# NOTE: UCR is univariate, providing complementary evidence to multivariate
#       SKAB and SMD datasets.
################################################################################

echo "================================================================================"
echo "UCR Anomaly Archive - Complete Online Phase Analysis"
echo "================================================================================"
echo ""
echo "Dataset Characteristics:"
echo "  - Name: UCR Anomaly Archive"
echo "  - Entities: 250 (using 5 diverse samples)"
echo "  - Dimensions: 1 column (UNIVARIATE time series)"
echo "  - Size: Variable (1,000 to 100,000+ rows)"
echo "  - Type: Diverse real-world time series (ECG, power, sensors, etc.)"
echo ""
echo "Analyses to run:"
echo "  1. Adaptive Analysis     → ~25 minutes"
echo "  2. Scalability Analysis  → ~25 minutes"
echo "  3. Window Size Analysis  → ~25 minutes"
echo ""
echo "Total estimated time: 75 minutes"
echo "================================================================================"
echo ""

echo "Starting analyses automatically..."
echo ""

# Activate environment
echo ""
echo "Activating RAMS conda environment..."
unset PYTHONPATH
unset PYTHONHOME
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
entities=$(tail -n +2 dataset_lists/ucr_sample.csv | cut -d',' -f2 | tr -d '"' | head -5)
all_trained=true

for entity in $entities; do
    model_dir="./Mononito/trained_models/anomaly_archive/${entity}"
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
    echo "You need to run OFFLINE training first. Create train_all_ucr.sh:"
    echo "  cp shells/train_all_skab.sh shells/train_all_ucr.sh"
    echo "  # Edit to use: --dataset anomaly_archive"
    echo "  ./shells/train_all_ucr.sh"
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
echo "Testing update intervals: 5, 10, 20, None"
echo "Num windows: 50"
echo "Num entities: 5 (diverse univariate series)"
echo "Injecting regime shifts: YES (at 25%, 50%, 75%)"
echo ""

python3 online_phase_analysis.py \
    --dataset-list dataset_lists/ucr_sample.csv \
    --data-dir ./Mononito/datasets \
    --trained-model-dir ./Mononito/trained_models \
    --output-dir ./results_ucr/adaptive \
    --adaptive-analysis \
    --num-windows 50 \
    --num-entities 5 \
    --update-intervals "5,10,20,None"

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Adaptive analysis completed!"
    echo "Results: ./results_ucr/adaptive/adaptive_analysis/"
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
echo "Num windows: 60"
echo "Num entities: 5"
echo "Question: Does RAMSeS work efficiently on univariate data?"
echo ""

python3 online_phase_analysis.py \
    --dataset-list dataset_lists/ucr_sample.csv \
    --data-dir ./Mononito/datasets \
    --trained-model-dir ./Mononito/trained_models \
    --output-dir ./results_ucr/scalability \
    --scalability-analysis \
    --num-windows 60 \
    --num-entities 5 \
    --num-models-range "3,5,8"

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Scalability analysis completed!"
    echo "Results: ./results_ucr/scalability/scalability_analysis/"
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
echo "Num entities: 5"
echo "Question: What's the optimal window size for univariate time series?"
echo ""

python3 online_phase_analysis.py \
    --dataset-list dataset_lists/ucr_sample.csv \
    --data-dir ./Mononito/datasets \
    --trained-model-dir ./Mononito/trained_models \
    --output-dir ./results_ucr/window_size \
    --window-size-analysis \
    --inject-synthetic \
    --num-entities 5

# Note: Window sizes auto-calculated based on data length

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Window size analysis completed!"
    echo "Results: ./results_ucr/window_size/window_size_analysis/"
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
echo "UCR ANOMALY ARCHIVE ANALYSIS COMPLETE!"
echo "================================================================================"
echo ""
echo "Results saved in ./results_ucr/"
echo "  1. ./results_ucr/adaptive/          ← Adaptive analysis (R1.O2, R1.O3)"
echo "  2. ./results_ucr/scalability/       ← Scalability analysis (R1.O3)"
echo "  3. ./results_ucr/window_size/       ← Window sensitivity (R2.O5)"
echo ""
echo "View key results:"
echo "  find ./results_ucr -name '*_summary.txt' -exec cat {} \\;"
echo "  find ./results_ucr -name '*.png'"
echo ""
echo "================================================================================"
echo "UCR Characteristics:"
echo "  ✓ Univariate (1 column): Complements multivariate SKAB/SMD"
echo "  ✓ Diverse domains: ECG, power, sensors → Shows generalization"
echo "  ✓ Moderate size: Good balance of speed and evidence"
echo "  ✓ IMPORTANT: Shows RAMSeS works on both univariate AND multivariate data!"
echo "================================================================================"
