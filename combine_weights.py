#!/usr/bin/env python3
"""
Combine weights using lambda mixing.

This script combines layer_7_ones_weights.npy with any chosen weight file using:
    combined = lambda * ones + (1 - lambda) * weights

The combined weights are saved in the same directory as the input weights
with "_lambda_<VALUE>" appended to the filename.
"""

import argparse
import numpy as np
from pathlib import Path
import sys


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Combine weights using lambda mixing: lambda * ones + (1-lambda) * weights'
    )
    parser.add_argument(
        '--weights_path',
        type=str,
        required=True,
        help='Path to the .npy weights file to combine with ones'
    )
    parser.add_argument(
        '--lambda_value',
        type=float,
        default=0.0,
        help='Mixing coefficient (default: 0.0). Combined = lambda*ones + (1-lambda)*weights'
    )
    return parser.parse_args()


def main():
    """Main function to combine weights."""
    args = parse_args()
    
    # Get script directory and construct path to ones weights
    script_dir = Path(__file__).parent
    ones_path = script_dir / "layer_7_ones_weights.npy"
    
    # Validate inputs
    weights_path = Path(args.weights_path)
    if not weights_path.exists():
        print(f"Error: Weights file not found: {weights_path}")
        sys.exit(1)
    
    if not ones_path.exists():
        print(f"Error: Ones weights file not found: {ones_path}")
        sys.exit(1)
    
    lambda_value = args.lambda_value
    if not (0 <= lambda_value <= 1):
        print(f"Warning: Lambda value {lambda_value} is outside [0, 1] range")
    
    print("="*70)
    print("Weight Combination Script")
    print("="*70)
    print(f"Ones weights path:   {ones_path}")
    print(f"Input weights path:  {weights_path}")
    print(f"Lambda value:        {lambda_value}")
    print(f"Formula:             combined = {lambda_value} * ones + {1-lambda_value} * weights")
    print()
    
    # Load weights
    print("Loading weights...")
    ones_weights = np.load(ones_path)
    input_weights = np.load(weights_path)
    
    print(f"  Ones shape:  {ones_weights.shape}")
    print(f"  Input shape: {input_weights.shape}")
    
    # Validate shapes match
    if ones_weights.shape != input_weights.shape:
        print(f"\nError: Shape mismatch!")
        print(f"  Ones weights shape:  {ones_weights.shape}")
        print(f"  Input weights shape: {input_weights.shape}")
        sys.exit(1)
    
    # Combine weights
    print("\nCombining weights...")
    combined_weights = lambda_value * ones_weights + (1 - lambda_value) * input_weights
    
    # Build output filename
    # Remove .npy extension, add lambda suffix, then add .npy back
    output_name = weights_path.stem + f"_lambda_{lambda_value}" + ".npy"
    output_path = weights_path.parent / output_name
    
    # Save combined weights
    print(f"Saving combined weights to: {output_path}")
    np.save(output_path, combined_weights)
    
    # Print statistics
    print("\nStatistics:")
    print(f"  Ones weights     - Min: {ones_weights.min():.6f}, Max: {ones_weights.max():.6f}, Mean: {ones_weights.mean():.6f}")
    print(f"  Input weights    - Min: {input_weights.min():.6f}, Max: {input_weights.max():.6f}, Mean: {input_weights.mean():.6f}")
    print(f"  Combined weights - Min: {combined_weights.min():.6f}, Max: {combined_weights.max():.6f}, Mean: {combined_weights.mean():.6f}")
    
    print("\n" + "="*70)
    print("SUCCESS: Combined weights saved!")
    print("="*70)
    print(f"Output file: {output_path}")
    print(f"File size:   {output_path.stat().st_size / 1024:.2f} KB")


if __name__ == "__main__":
    main()

