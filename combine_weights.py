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
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend


def create_histogram(weights: np.ndarray, title: str, output_path: Path):
    """Create and save histogram plot for weight values.
    
    Args:
        weights: Array of weight values
        title: Title for the plot
        output_path: Path to save the plot
    """
    plt.figure(figsize=(12, 6))
    
    # Create histogram
    plt.hist(weights, bins=100, alpha=0.7, color='blue', edgecolor='black')
    
    # Add statistics
    mean_val = weights.mean()
    max_val = np.abs(weights).max()
    std_val = weights.std()
    
    plt.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.6f}')
    
    plt.xlabel('Weight Value', fontsize=12)
    plt.ylabel('Number of Dimensions', fontsize=12)
    plt.title(f'{title}\nMax: {max_val:.6f}, Mean: {mean_val:.6f}, Std: {std_val:.6f}',
              fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save plot
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved histogram to {output_path}")


def create_scatter_plot(weights: np.ndarray, title: str, output_path: Path):
    """Create and save scatter plot showing weight per dimension.
    
    Args:
        weights: Array of weight values (one per dimension)
        title: Title for the plot
        output_path: Path to save the plot
    """
    plt.figure(figsize=(14, 6))
    
    num_dims = len(weights)
    dim_indices = np.arange(num_dims)
    
    # Plot weight per dimension (scatter only, no line)
    plt.scatter(dim_indices, weights, s=3, alpha=0.6, color='blue')
    
    # Add statistics
    mean_val = weights.mean()
    max_val = np.abs(weights).max()
    std_val = weights.std()
    
    # Highlight top dimensions
    top_k = min(20, num_dims)
    top_indices = np.argsort(np.abs(weights))[-top_k:]
    plt.scatter(top_indices, weights[top_indices], 
                s=80, color='red', alpha=0.8, 
                label=f'Top {top_k} dimensions', zorder=5, marker='*')
    
    plt.axhline(mean_val, color='green', linestyle='--', linewidth=1.5, 
                label=f'Mean: {mean_val:.6f}', alpha=0.7)
    
    plt.xlabel('Dimension Index', fontsize=12)
    plt.ylabel('Weight Value', fontsize=12)
    plt.title(f'{title}\nTotal Dimensions: {num_dims}, Max: {max_val:.6f}, Mean: {mean_val:.6f}, Std: {std_val:.6f}',
              fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save plot
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved scatter plot to {output_path}")


def create_comparison_histogram(ones_weights: np.ndarray, input_weights: np.ndarray, 
                               combined_weights: np.ndarray, lambda_value: float, 
                               output_dir: Path, base_name: str):
    """Create comparison histogram showing all three weight distributions.
    
    Args:
        ones_weights: Ones weight array
        input_weights: Input weight array
        combined_weights: Combined weight array
        lambda_value: Lambda mixing coefficient
        output_dir: Directory to save plots
        base_name: Base name for output files
    """
    # Histogram comparison
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 6))
    
    weights_list = [
        (ones_weights, 'Ones Weights', ax1),
        (input_weights, 'Input Weights', ax2),
        (combined_weights, f'Combined (λ={lambda_value})', ax3)
    ]
    
    for weights, title, ax in weights_list:
        ax.hist(weights, bins=100, alpha=0.7, color='blue', edgecolor='black')
        mean_val = weights.mean()
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.6f}')
        ax.set_xlabel('Weight Value', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title(f'{title}\nMean: {mean_val:.6f}, Max: {np.abs(weights).max():.6f}', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    fig.suptitle(f'Weight Distribution Comparison (λ={lambda_value})', fontsize=16)
    plt.tight_layout()
    
    output_path = output_dir / f'{base_name}_lambda_{lambda_value}_comparison_hist.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved comparison histogram to {output_path}")


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
    try:
        np.save(output_path, combined_weights)
        # Verify file was created
        if output_path.exists():
            print(f"✓ Combined weights file created successfully")
        else:
            print(f"✗ Warning: File was not created at {output_path}")
    except Exception as e:
        print(f"✗ Error saving combined weights: {e}")
        sys.exit(1)
    
    # Print statistics
    print("\nStatistics:")
    print(f"  Ones weights     - Min: {ones_weights.min():.6f}, Max: {ones_weights.max():.6f}, Mean: {ones_weights.mean():.6f}")
    print(f"  Input weights    - Min: {input_weights.min():.6f}, Max: {input_weights.max():.6f}, Mean: {input_weights.mean():.6f}")
    print(f"  Combined weights - Min: {combined_weights.min():.6f}, Max: {combined_weights.max():.6f}, Mean: {combined_weights.mean():.6f}")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    output_dir = weights_path.parent
    base_name = weights_path.stem
    
    # Comparison histogram (3-panel)
    print("  Creating comparison histogram...")
    create_comparison_histogram(ones_weights, input_weights, combined_weights, 
                                lambda_value, output_dir, base_name)
    
    # Separate scatter plots
    print("  Creating individual scatter plots...")
    create_scatter_plot(ones_weights, 'Ones Weights Scatter', 
                       output_dir / f'{base_name}_lambda_{lambda_value}_ones_scatter.png')
    create_scatter_plot(input_weights, 'Input Weights Scatter', 
                       output_dir / f'{base_name}_lambda_{lambda_value}_input_scatter.png')
    create_scatter_plot(combined_weights, f'Combined Weights Scatter (λ={lambda_value})', 
                       output_dir / f'{base_name}_lambda_{lambda_value}_combined_scatter.png')
    
    print("\n" + "="*70)
    print("SUCCESS: Combined weights saved!")
    print("="*70)
    print(f"Output file: {output_path}")
    print(f"File size:   {output_path.stat().st_size / 1024:.2f} KB")
    print(f"\nVisualization plots saved to: {output_dir}")


if __name__ == "__main__":
    main()

