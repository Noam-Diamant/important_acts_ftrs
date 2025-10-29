#!/bin/bash
# Convenience script to run SAE gradient analysis

# Set script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo "Running SAE Gradient Analysis for Qwen2.5-0.5B"
echo "=============================================="
echo ""

# Run with default settings
python important_feaures/sae_gradient_analysis.py \
    --num_samples 500 \
    --batch_size 4 \
    --max_length 128 \
    --layers "0-23" \
    --output_dir important_feaures \
    --device cuda

echo ""
echo "=============================================="
echo "Analysis complete! Check important_feaures/ for results."

