#!/bin/bash
################################################################################
# launch_ucr_pipeline.sh - RAMSeS UCR Pipeline Launcher
#
# This script:
#   1. Changes to RAMSeS directory
#   2. Activates RAMS conda environment
#   3. Runs the complete pipeline for UCR dataset only
#
# Usage:
#   ./launch_ucr_pipeline.sh
#
# Or from anywhere:
#   /home/maxoud/local-storage/projects/RAMSeS/launch_ucr_pipeline.sh
#
# Recommended: Run in screen/tmux
#   screen -S ucr
#   ./launch_ucr_pipeline.sh
#   Ctrl+A, D to detach
################################################################################

# Navigate to RAMSeS directory
cd /home/maxoud/local-storage/projects/RAMSeS || {
    echo "ERROR: Cannot find RAMSeS directory!"
    exit 1
}

echo "✓ Changed to RAMSeS directory: $(pwd)"

# Activate conda environment
unset PYTHONPATH
unset PYTHONHOME
source /raid0_ssd1/anaconda3/etc/profile.d/conda.sh
conda activate RAMS

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to activate RAMS environment"
    exit 1
fi

echo "✓ Activated RAMS conda environment"
echo ""

# Run UCR pipeline
./shells/run_complete_ucr_pipeline.sh

# Capture exit code
exit_code=$?

if [ $exit_code -eq 0 ]; then
    echo ""
    echo "════════════════════════════════════════════════════════════════════════════════"
    echo "  ✅ UCR PIPELINE COMPLETED SUCCESSFULLY!"
    echo "════════════════════════════════════════════════════════════════════════════════"
else
    echo ""
    echo "════════════════════════════════════════════════════════════════════════════════"
    echo "  ❌ UCR PIPELINE FAILED (exit code: $exit_code)"
    echo "  Check logs in ./logs/ directory"
    echo "════════════════════════════════════════════════════════════════════════════════"
fi

exit $exit_code
