#!/bin/bash
################################################################################
# launch_skab_pipeline.sh - RAMSeS SKAB Pipeline Launcher
#
# This script:
#   1. Changes to RAMSeS directory
#   2. Activates RAMS conda environment
#   3. Runs the complete pipeline for SKAB dataset only
#
# Usage:
#   ./launch_skab_pipeline.sh
#
# Or from anywhere:
#   /home/maxoud/local-storage/projects/RAMSeS/launch_skab_pipeline.sh
#
# Recommended: Run in screen/tmux
#   screen -S skab
#   ./launch_skab_pipeline.sh
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

# Run SKAB pipeline
./shells/run_complete_skab_pipeline.sh

# Capture exit code
exit_code=$?

if [ $exit_code -eq 0 ]; then
    echo ""
    echo "════════════════════════════════════════════════════════════════════════════════"
    echo "  ✅ SKAB PIPELINE COMPLETED SUCCESSFULLY!"
    echo "════════════════════════════════════════════════════════════════════════════════"
else
    echo ""
    echo "════════════════════════════════════════════════════════════════════════════════"
    echo "  ❌ SKAB PIPELINE FAILED (exit code: $exit_code)"
    echo "  Check logs in ./logs/ directory"
    echo "════════════════════════════════════════════════════════════════════════════════"
fi

exit $exit_code
