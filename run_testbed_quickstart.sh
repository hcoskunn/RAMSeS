#!/bin/bash
#
# Quick Start Script for RAMSeS Testbed
# 
# This script helps you quickly run the testbed system
#

set -e  # Exit on error

echo "================================================================================"
echo "RAMSeS Comprehensive Testbed - Quick Start"
echo "================================================================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored messages
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Python is available
if ! command -v python &> /dev/null; then
    print_error "Python is not installed or not in PATH"
    exit 1
fi

print_info "Python found: $(python --version)"

# Check required packages
print_info "Checking required packages..."
python -c "import pandas, numpy, matplotlib, seaborn, psutil, yaml" 2>/dev/null || {
    print_warning "Some required packages are missing"
    print_info "Installing required packages..."
    pip install pandas numpy matplotlib seaborn psutil pyyaml
}

echo ""
echo "================================================================================"
echo "Setup Options"
echo "================================================================================"
echo ""
echo "1. Run testbed on SKAB dataset (from existing CSV)"
echo "2. Generate new dataset lists and run testbed"
echo "3. Run testbed on specific domain"
echo "4. Generate visualizations from existing results"
echo "5. Full pipeline (generate lists + run testbed + visualize)"
echo ""

read -p "Select option (1-5): " option

case $option in
    1)
        print_info "Running testbed on SKAB dataset..."
        DATASET_LIST="/home/maxoud/local-storage/projects/TSB-AutoAD/testbed/file_list/test_m_skab.csv"
        
        if [ ! -f "$DATASET_LIST" ]; then
            print_error "Dataset list not found: $DATASET_LIST"
            print_info "Please update the path or use option 2 to generate lists"
            exit 1
        fi
        
        OUTPUT_DIR="testbed_results_$(date +%Y%m%d_%H%M%S)"
        
        print_info "Dataset list: $DATASET_LIST"
        print_info "Output directory: $OUTPUT_DIR"
        
        python run_testbed_comprehensive.py \
            --dataset-list "$DATASET_LIST" \
            --output-dir "$OUTPUT_DIR"
        
        print_info "Testbed complete! Results in: $OUTPUT_DIR"
        print_info "Generating visualizations..."
        
        python visualize_testbed_comprehensive.py \
            --results-dir "$OUTPUT_DIR" \
            --plot all
        
        print_info "Visualizations saved in: $OUTPUT_DIR/plots/"
        ;;
        
    2)
        print_info "Generating dataset lists..."
        python generate_dataset_lists.py --type all
        
        print_info "Dataset lists generated in: testbed/file_list/"
        print_info "Available lists:"
        ls -lh testbed/file_list/*.csv
        
        echo ""
        read -p "Enter dataset list to use (e.g., testbed/file_list/test_skab_all.csv): " DATASET_LIST
        
        if [ ! -f "$DATASET_LIST" ]; then
            print_error "File not found: $DATASET_LIST"
            exit 1
        fi
        
        OUTPUT_DIR="testbed_results_$(date +%Y%m%d_%H%M%S)"
        
        print_info "Running testbed..."
        python run_testbed_comprehensive.py \
            --dataset-list "$DATASET_LIST" \
            --output-dir "$OUTPUT_DIR"
        
        print_info "Generating visualizations..."
        python visualize_testbed_comprehensive.py \
            --results-dir "$OUTPUT_DIR" \
            --plot all
        
        print_info "Complete! Results in: $OUTPUT_DIR"
        ;;
        
    3)
        read -p "Enter dataset list CSV path: " DATASET_LIST
        
        if [ ! -f "$DATASET_LIST" ]; then
            print_error "File not found: $DATASET_LIST"
            exit 1
        fi
        
        read -p "Enter domain name (e.g., SKAB): " DOMAIN
        OUTPUT_DIR="testbed_results_${DOMAIN}_$(date +%Y%m%d_%H%M%S)"
        
        print_info "Running testbed for domain: $DOMAIN"
        python run_testbed_comprehensive.py \
            --dataset-list "$DATASET_LIST" \
            --output-dir "$OUTPUT_DIR" \
            --domain "$DOMAIN"
        
        print_info "Generating visualizations..."
        python visualize_testbed_comprehensive.py \
            --results-dir "$OUTPUT_DIR" \
            --plot all
        
        print_info "Complete! Results in: $OUTPUT_DIR"
        ;;
        
    4)
        read -p "Enter results directory path: " RESULTS_DIR
        
        if [ ! -d "$RESULTS_DIR" ]; then
            print_error "Directory not found: $RESULTS_DIR"
            exit 1
        fi
        
        print_info "Generating visualizations..."
        python visualize_testbed_comprehensive.py \
            --results-dir "$RESULTS_DIR" \
            --plot all
        
        print_info "Visualizations saved in: $RESULTS_DIR/plots/"
        ;;
        
    5)
        print_info "Running full pipeline..."
        
        # Generate lists
        print_info "Step 1/3: Generating dataset lists..."
        python generate_dataset_lists.py --type all
        
        # Run testbed
        OUTPUT_DIR="testbed_results_full_$(date +%Y%m%d_%H%M%S)"
        print_info "Step 2/3: Running comprehensive testbed..."
        print_warning "This may take several hours depending on dataset size"
        
        python run_testbed_comprehensive.py \
            --dataset-list testbed/file_list/test_all_datasets.csv \
            --output-dir "$OUTPUT_DIR"
        
        # Generate visualizations
        print_info "Step 3/3: Generating visualizations..."
        python visualize_testbed_comprehensive.py \
            --results-dir "$OUTPUT_DIR" \
            --plot all
        
        print_info "Full pipeline complete!"
        print_info "Results directory: $OUTPUT_DIR"
        print_info "Summary report: $OUTPUT_DIR/overall_summary.txt"
        print_info "Plots directory: $OUTPUT_DIR/plots/"
        ;;
        
    *)
        print_error "Invalid option: $option"
        exit 1
        ;;
esac

echo ""
echo "================================================================================"
echo "Done!"
echo "================================================================================"
echo ""
print_info "Quick links:"
echo "  - Summary report: $OUTPUT_DIR/overall_summary.txt"
echo "  - Detailed JSON: $OUTPUT_DIR/overall_summary.json"
echo "  - Plots: $OUTPUT_DIR/plots/"
echo ""
print_info "To view results:"
echo "  cat $OUTPUT_DIR/overall_summary.txt"
echo "  ls $OUTPUT_DIR/plots/"
echo ""
