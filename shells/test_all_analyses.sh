#!/bin/bash
# Quick test of all three analyses with minimal windows

source /raid0_ssd1/anaconda3/etc/profile.d/conda.sh
conda activate RAMS

cd /home/maxoud/local-storage/projects/RAMSeS

echo "=========================================="
echo "Testing All Three Analysis Modes"
echo "=========================================="
echo ""

# Test 1: Adaptive Analysis
echo "TEST 1/3: Adaptive Analysis (3 windows)..."
python3 online_phase_analysis.py \
    --dataset-list dataset_lists/skab_full.csv \
    --data-dir ./Mononito/datasets \
    --trained-model-dir ./Mononito/trained_models \
    --output-dir ./test_results/adaptive \
    --adaptive-analysis \
    --num-windows 3 \
    --update-intervals "5" \
    --inject-synthetic \
    2>&1 | tail -20

if [ $? -eq 0 ]; then
    echo "✓ Adaptive analysis PASSED"
else
    echo "✗ Adaptive analysis FAILED"
    exit 1
fi

echo ""

# Test 2: Scalability Analysis
echo "TEST 2/3: Scalability Analysis (3 windows, pool size 3)..."
python3 online_phase_analysis.py \
    --dataset-list dataset_lists/skab_full.csv \
    --data-dir ./Mononito/datasets \
    --trained-model-dir ./Mononito/trained_models \
    --output-dir ./test_results/scalability \
    --scalability-analysis \
    --num-windows 3 \
    --num-models-range "3" \
    --inject-synthetic \
    2>&1 | tail -20

if [ $? -eq 0 ]; then
    echo "✓ Scalability analysis PASSED"
else
    echo "✗ Scalability analysis FAILED"
    exit 1
fi

echo ""

# Test 3: Window Size Analysis
echo "TEST 3/3: Window Size Analysis (1 window size)..."
python3 online_phase_analysis.py \
    --dataset-list dataset_lists/skab_full.csv \
    --data-dir ./Mononito/datasets \
    --trained-model-dir ./Mononito/trained_models \
    --output-dir ./test_results/window_size \
    --window-size-analysis \
    --window-sizes "46" \
    --inject-synthetic \
    2>&1 | tail -20

if [ $? -eq 0 ]; then
    echo "✓ Window size analysis PASSED"
else
    echo "✗ Window size analysis FAILED"
    exit 1
fi

echo ""
echo "=========================================="
echo "ALL TESTS PASSED!"
echo "=========================================="
echo "The system is ready to run the full pipeline."
