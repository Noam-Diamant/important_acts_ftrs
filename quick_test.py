"""
Quick test script for manual verification of gradient analysis tools.

This script runs a single quick test with minimal data to verify basic functionality.

Usage:
    # Test layer gradient analysis with values
    python quick_test.py --script layer --aggregate_by values
    
    # Test layer gradient analysis with gradients and power
    python quick_test.py --script layer --aggregate_by gradients --power 2.0
    
    # Test SAE gradient analysis
    python quick_test.py --script sae --aggregate_by values --power 0.5
"""

import argparse
import subprocess
import sys
import json
import numpy as np
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description='Quick test for gradient analysis')
    parser.add_argument('--script', type=str, choices=['layer', 'sae'], default='layer',
                        help='Which script to test: layer or sae')
    parser.add_argument('--aggregate_by', type=str, choices=['values', 'gradients'], 
                        default='values',
                        help='Aggregation type')
    parser.add_argument('--aggregation_mode', type=str, choices=['mean', 'mean_abs', 'both'],
                        default='mean_abs',
                        help='Aggregation mode')
    parser.add_argument('--power', type=float, default=None,
                        help='Power transformation')
    parser.add_argument('--normalize', type=str, default='none',
                        choices=['none', 'sum_abs', 'sum', 'both'],
                        help='Normalization mode')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of samples (default: 5 for quick test)')
    parser.add_argument('--layers', type=str, default='0',
                        help='Layers to test (default: just layer 0)')
    return parser.parse_args()


def run_test(args):
    """Run the test with given configuration."""
    
    # Determine script name
    if args.script == 'layer':
        script_name = 'layer_gradient_analysis.py'
    else:
        script_name = 'sae_gradient_analysis.py'
    
    # Build command
    cmd = [
        'python',
        script_name,
        '--model_name', 'Qwen/Qwen2.5-0.5B',
        '--num_samples', str(args.num_samples),
        '--batch_size', '2',
        '--max_length', '64',
        '--layers', args.layers,
        '--aggregation_mode', args.aggregation_mode,
        '--normalize_gradients', args.normalize,
        '--aggregate_by', args.aggregate_by,
        '--device', 'cuda',
        '--save_outputs',
    ]
    
    if args.power is not None:
        cmd.extend(['--power', str(args.power)])
    
    print("="*80)
    print("QUICK TEST - Gradient Analysis")
    print("="*80)
    print(f"\nScript: {script_name}")
    print(f"Configuration:")
    print(f"  - Aggregate by: {args.aggregate_by}")
    print(f"  - Aggregation mode: {args.aggregation_mode}")
    print(f"  - Power: {args.power}")
    print(f"  - Normalization: {args.normalize}")
    print(f"  - Samples: {args.num_samples}")
    print(f"  - Layers: {args.layers}")
    print(f"\nCommand:\n{' '.join(cmd)}")
    print("\n" + "="*80)
    print("Running test...\n")
    
    # Run the script
    try:
        result = subprocess.run(
            cmd,
            cwd=str(Path(__file__).parent),
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode != 0:
            print("\n❌ TEST FAILED with return code", result.returncode)
            return False
        
        print("\n" + "="*80)
        print("TEST COMPLETED SUCCESSFULLY")
        print("="*80)
        
        # Find the output directory
        import os
        output_dirs = [d for d in os.listdir(Path(__file__).parent) 
                      if os.path.isdir(Path(__file__).parent / d) and 
                      (d.startswith('layer_activations_') or d.startswith('sae_features_'))]
        
        if output_dirs:
            latest_dir = sorted(output_dirs)[-1]
            full_path = Path(__file__).parent / latest_dir
            
            print(f"\n📁 Output directory: {latest_dir}")
            
            # Load and inspect results
            print("\n🔍 Inspecting results:")
            
            # Check metadata
            metadata_file = full_path / 'metadata.json'
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                config = metadata['analysis_config']
                print(f"\n  Metadata verification:")
                print(f"    ✓ aggregate_by: {config['aggregate_by']}")
                print(f"    ✓ power: {config['power']}")
                print(f"    ✓ aggregation_mode: {config['aggregation_mode']}")
                print(f"    ✓ normalization_mode: {config['normalization_mode']}")
                
                # Check dimension info
                if 'dimension_info' in metadata:
                    print(f"\n  Dimension sizes:")
                    for layer, size in metadata['dimension_info'].items():
                        print(f"    - {layer}: {size} dimensions")
            
            # Load and inspect gradient arrays
            npy_files = list(full_path.glob('*.npy'))
            print(f"\n  Generated files: {len(npy_files)} .npy files")
            
            if npy_files:
                print(f"\n  Sample data from first file:")
                first_file = sorted(npy_files)[0]
                data = np.load(first_file)
                print(f"    File: {first_file.name}")
                print(f"    Shape: {data.shape}")
                print(f"    Mean: {data.mean():.6f}")
                print(f"    Std: {data.std():.6f}")
                print(f"    Min: {data.min():.6f}")
                print(f"    Max: {data.max():.6f}")
                
                # Show first few values
                print(f"    First 10 values: {data[:10]}")
                
                # Verify power transformation effect
                if args.power is not None:
                    print(f"\n  Power transformation verification:")
                    print(f"    Applied power: {args.power}")
                    if args.power > 1.0:
                        print(f"    ✓ Large values should be emphasized")
                    elif args.power < 1.0:
                        print(f"    ✓ Large values should be compressed")
                
                # Verify normalization
                if args.normalize == 'sum_abs':
                    sum_abs = np.sum(np.abs(data))
                    print(f"\n  Normalization verification (sum_abs):")
                    print(f"    Sum of absolute values: {sum_abs:.1f}")
                    print(f"    Target (dimension size): {len(data)}")
                    if abs(sum_abs - len(data)) < 0.1:
                        print(f"    ✓ Normalization correct!")
                    else:
                        print(f"    ⚠️  Normalization may be off")
                elif args.normalize == 'sum':
                    sum_val = np.sum(data)
                    print(f"\n  Normalization verification (sum):")
                    print(f"    Sum: {sum_val:.1f}")
                    print(f"    Target (dimension size): {len(data)}")
                    if abs(sum_val - len(data)) < 0.1:
                        print(f"    ✓ Normalization correct!")
                    else:
                        print(f"    ⚠️  Normalization may be off")
                elif args.normalize == 'none':
                    print(f"\n  No normalization applied (as expected)")
            
            # List all files
            print(f"\n  All generated files:")
            for f in sorted(npy_files):
                size_kb = f.stat().st_size / 1024
                print(f"    - {f.name} ({size_kb:.1f} KB)")
            
            # Check for plots
            png_files = list(full_path.glob('*.png'))
            if png_files:
                print(f"\n  Generated {len(png_files)} plots:")
                for f in sorted(png_files)[:5]:  # Show first 5
                    print(f"    - {f.name}")
        
        print("\n✅ Test completed successfully!")
        print("="*80)
        return True
        
    except subprocess.TimeoutExpired:
        print("\n❌ TEST FAILED: Timeout")
        return False
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    args = parse_args()
    success = run_test(args)
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())

