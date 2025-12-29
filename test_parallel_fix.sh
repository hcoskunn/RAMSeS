#!/bin/bash
# Quick test script to verify the parallel fix is working

echo "Testing parallel execution fix..."
echo ""

cd /home/maxoud/local-storage/projects/RAMSeS

# Clear Python cache
echo "→ Clearing Python cache..."
find . -type f -name "*.pyc" -delete 2>/dev/null
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
echo "✓ Cache cleared"
echo ""

# Run a quick test
echo "→ Running quick test with machine-1-1..."
echo "→ Looking for 'Created independent data copies' message..."
echo ""

timeout 60 python app.py --dataset smd --entity machine-1-1 --parallel true 2>&1 | grep -E "(Created independent|Launching:|dimension|features)" | head -20

echo ""
echo "─────────────────────────────────────────────"
echo "If you see 'Created independent data copies' BEFORE 'Launching:', the fix is active!"
echo "If you see dimension errors, the old code is still running."
echo "─────────────────────────────────────────────"
