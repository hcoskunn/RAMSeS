#!/bin/bash
################################################################################
# launch_complete_pipeline.sh - RAMSeS Complete Pipeline Launcher
#
# This script:
#   1. Changes to RAMSeS directory
#   2. Activates RAMS conda environment
#   3. Runs the complete pipeline for all 3 datasets
#
# Usage:
#   ./launch_complete_pipeline.sh
#
# Or from anywhere:
#   /home/maxoud/local-storage/projects/RAMSeS/launch_complete_pipeline.sh
#
# Recommended: Run in screen/tmux for long execution
#   screen -S ramses
#   ./launch_complete_pipeline.sh
#   Ctrl+A, D to detach
################################################################################

# Navigate to RAMSeS directory
cd /home/maxoud/local-storage/projects/RAMSeS || {
    echo "ERROR: Cannot find RAMSeS directory!"
    exit 1
}

echo "✓ Changed to RAMSeS directory: $(pwd)"

# Activate conda environment
source /raid0_ssd1/anaconda3/etc/profile.d/conda.sh
conda activate RAMS

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to activate RAMS environment"
    exit 1
fi

echo "✓ Activated RAMS conda environment"
echo ""

# Run complete pipeline
./shells/run_complete_pipeline.sh

# Capture exit code
exit_code=$?

if [ $exit_code -eq 0 ]; then
    echo ""
    echo "════════════════════════════════════════════════════════════════════════════════"
    echo "  ✅ PIPELINE COMPLETED SUCCESSFULLY!"
    echo "════════════════════════════════════════════════════════════════════════════════"
else
    echo ""
    echo "════════════════════════════════════════════════════════════════════════════════"
    echo "  ❌ PIPELINE FAILED (exit code: $exit_code)"
    echo "  Check logs in ./logs/ directory"
    echo "════════════════════════════════════════════════════════════════════════════════"
fi

exit $exit_code
