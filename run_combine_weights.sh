#!/bin/bash

################################################################################
# Weight Combination Script
################################################################################
# 
# This script combines layer_7_ones_weights.npy with any chosen weight file
# using the formula: combined = lambda * ones + (1 - lambda) * weights
#
# USAGE:
#   1. Edit the WEIGHTS_PATH variable below to point to your weights file
#   2. Optionally edit LAMBDA_VALUE (default: 0)
#   3. Run: bash run_combine_weights.sh
#
# OUTPUT:
#   Combined weights will be saved in the same directory as the input weights
#   with "_lambda_<VALUE>" appended to the filename
#
################################################################################

# ============================================================================
# CONFIGURATION - EDIT THESE VARIABLES
# ============================================================================

# Path to the weights file to combine (REQUIRED - change this for each run)
WEIGHTS_PATH="layer_activations_model_HuggingFaceH4_zephyr-7b-beta_num_samples_10000_batch_size_8_layers_7_agg_values_power_0.5_20251116_003556/layer_7_mean_abs_norm_sum_abs_gradients.npy"

# Lambda mixing coefficient (optional, default: 0)
# Combined = lambda * ones + (1 - lambda) * weights
LAMBDA_VALUE=0

# ============================================================================
# SCRIPT EXECUTION - DO NOT EDIT BELOW THIS LINE
# ============================================================================

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Run the Python script
echo "Running weight combination..."
echo "Weights path: $WEIGHTS_PATH"
echo "Lambda value: $LAMBDA_VALUE"
echo ""

python "$SCRIPT_DIR/combine_weights.py" \
    --weights_path "$WEIGHTS_PATH" \
    --lambda_value "$LAMBDA_VALUE"

# Check if the script succeeded
if [ $? -eq 0 ]; then
    echo ""
    echo "Weight combination completed successfully!"
else
    echo ""
    echo "Error: Weight combination failed!"
    exit 1
fi

