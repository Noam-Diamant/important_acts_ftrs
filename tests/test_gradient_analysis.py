"""
Test script for gradient analysis tools.

This script tests both layer_gradient_analysis.py and sae_gradient_analysis.py
with various parameter combinations to verify functionality.

Usage:
    python test_gradient_analysis.py
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Configuration
TEST_MODEL = "Qwen/Qwen2.5-0.5B"
TEST_SAMPLES = 10  # Small number for fast testing
TEST_BATCH_SIZE = 2
TEST_LAYERS = "0,5"  # Just two layers for quick testing
TEST_MAX_LENGTH = 64

# Test configurations
TEST_CONFIGS = [
    {
        "name": "Test 1: Values aggregation, mean_abs, no power",
        "args": {
            "aggregate_by": "values",
            "aggregation_mode": "mean_abs",
            "normalize_gradients": "none",
            "power": None,
        }
    },
    {
        "name": "Test 2: Values aggregation, mean_abs, power=2.0",
        "args": {
            "aggregate_by": "values",
            "aggregation_mode": "mean_abs",
            "normalize_gradients": "sum_abs",
            "power": 2.0,
        }
    },
    {
        "name": "Test 3: Gradients aggregation, mean, no power",
        "args": {
            "aggregate_by": "gradients",
            "aggregation_mode": "mean",
            "normalize_gradients": "sum_abs",
            "power": None,
        }
    },
    {
        "name": "Test 4: Values aggregation, both modes, power=0.5",
        "args": {
            "aggregate_by": "values",
            "aggregation_mode": "both",
            "normalize_gradients": "none",
            "power": 0.5,
        }
    },
    {
        "name": "Test 5: Gradients aggregation, mean_abs, both normalizations",
        "args": {
            "aggregate_by": "gradients",
            "aggregation_mode": "mean_abs",
            "normalize_gradients": "both",
            "power": None,
        }
    },
]


def build_command(script_name, config, output_dir):
    """Build command line for running the script."""
    args = config["args"]
    
    cmd = [
        "python",
        script_name,
        "--model_name", TEST_MODEL,
        "--num_samples", str(TEST_SAMPLES),
        "--batch_size", str(TEST_BATCH_SIZE),
        "--layers", TEST_LAYERS,
        "--max_length", str(TEST_MAX_LENGTH),
        "--aggregation_mode", args["aggregation_mode"],
        "--normalize_gradients", args["normalize_gradients"],
        "--aggregate_by", args["aggregate_by"],
        "--device", "cuda",
        "--save_outputs",
    ]
    
    # Add power if specified
    if args["power"] is not None:
        cmd.extend(["--power", str(args["power"])])
    
    return cmd


def run_test(script_name, config, test_num, total_tests):
    """Run a single test configuration."""
    print("\n" + "="*80)
    print(f"Running {config['name']}")
    print(f"Script: {script_name}")
    print(f"Progress: Test {test_num}/{total_tests}")
    print("="*80)
    
    # Build command
    output_dir = f"test_output_{test_num}"
    cmd = build_command(script_name, config, output_dir)
    
    print("\nCommand:")
    print(" ".join(cmd))
    print()
    
    # Run the command
    try:
        result = subprocess.run(
            cmd,
            cwd=str(Path(__file__).parent),
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )
        
        # Print key output lines
        print("\n--- Output Summary ---")
        output_lines = result.stdout.split('\n')
        
        # Print configuration detection
        for line in output_lines:
            if any(keyword in line for keyword in [
                "aggregate_by=",
                "power=",
                "Processing layer",
                "before power/normalization",
                "after power=",
                "after normalization",
                "Completed analysis",
                "Results saved to",
                "Model type detected",
            ]):
                print(line)
        
        # Check for errors
        if result.returncode != 0:
            print("\n❌ TEST FAILED")
            print("Error output:")
            print(result.stderr[:500])  # Print first 500 chars of error
            return False
        
        print("\n✅ TEST PASSED")
        
        # Print output directory info
        output_dirs = [d for d in os.listdir(Path(__file__).parent) 
                      if os.path.isdir(Path(__file__).parent / d) and 
                      (d.startswith('layer_activations_') or d.startswith('sae_features_'))]
        if output_dirs:
            latest_dir = sorted(output_dirs)[-1]
            full_path = Path(__file__).parent / latest_dir
            print(f"\nOutput directory: {latest_dir}")
            
            # List files in directory
            files = list(full_path.glob('*.npy'))
            print(f"Generated {len(files)} .npy files")
            if files:
                print("Sample files:")
                for f in sorted(files)[:5]:
                    size_kb = f.stat().st_size / 1024
                    print(f"  - {f.name} ({size_kb:.1f} KB)")
            
            # Check metadata
            metadata_file = full_path / 'metadata.json'
            if metadata_file.exists():
                import json
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                print("\nMetadata:")
                print(f"  - aggregate_by: {metadata['analysis_config']['aggregate_by']}")
                print(f"  - power: {metadata['analysis_config']['power']}")
                print(f"  - aggregation_mode: {metadata['analysis_config']['aggregation_mode']}")
                print(f"  - normalization_mode: {metadata['analysis_config']['normalization_mode']}")
        
        return True
        
    except subprocess.TimeoutExpired:
        print("\n❌ TEST FAILED: Timeout (>10 minutes)")
        return False
    except Exception as e:
        print(f"\n❌ TEST FAILED: {str(e)}")
        return False


def cleanup_old_outputs():
    """Remove old test output directories."""
    print("\nCleaning up old test outputs...")
    parent = Path(__file__).parent
    
    for d in os.listdir(parent):
        if os.path.isdir(parent / d) and (
            d.startswith('layer_activations_model_') or 
            d.startswith('sae_features_model_')
        ):
            try:
                shutil.rmtree(parent / d)
                print(f"  Removed: {d}")
            except Exception as e:
                print(f"  Warning: Could not remove {d}: {e}")


def main():
    """Run all tests."""
    print("="*80)
    print("GRADIENT ANALYSIS TESTING SUITE")
    print("="*80)
    print(f"\nTest Configuration:")
    print(f"  Model: {TEST_MODEL}")
    print(f"  Samples: {TEST_SAMPLES}")
    print(f"  Batch size: {TEST_BATCH_SIZE}")
    print(f"  Layers: {TEST_LAYERS}")
    print(f"  Max length: {TEST_MAX_LENGTH}")
    print(f"  Total tests: {len(TEST_CONFIGS)} × 2 scripts = {len(TEST_CONFIGS) * 2}")
    
    # Ask for cleanup
    response = input("\nClean up old test outputs? [y/N]: ").strip().lower()
    if response == 'y':
        cleanup_old_outputs()
    
    input("\nPress Enter to start tests...")
    
    results = {
        "layer_gradient_analysis.py": [],
        "sae_gradient_analysis.py": []
    }
    
    # Test layer_gradient_analysis.py
    print("\n" + "="*80)
    print("TESTING: layer_gradient_analysis.py")
    print("="*80)
    
    for i, config in enumerate(TEST_CONFIGS, 1):
        success = run_test(
            "layer_gradient_analysis.py",
            config,
            i,
            len(TEST_CONFIGS)
        )
        results["layer_gradient_analysis.py"].append((config["name"], success))
    
    # Test sae_gradient_analysis.py
    print("\n" + "="*80)
    print("TESTING: sae_gradient_analysis.py")
    print("="*80)
    print("\nNote: SAE tests require SAE models to be available.")
    print("If SAE models are not found, tests will be skipped.\n")
    
    response = input("Run SAE tests? [y/N]: ").strip().lower()
    if response == 'y':
        for i, config in enumerate(TEST_CONFIGS, 1):
            success = run_test(
                "sae_gradient_analysis.py",
                config,
                i + len(TEST_CONFIGS),
                len(TEST_CONFIGS) * 2
            )
            results["sae_gradient_analysis.py"].append((config["name"], success))
    else:
        print("Skipping SAE tests.")
    
    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    for script_name, test_results in results.items():
        if not test_results:
            continue
        
        print(f"\n{script_name}:")
        passed = sum(1 for _, success in test_results if success)
        total = len(test_results)
        print(f"  Passed: {passed}/{total}")
        
        for test_name, success in test_results:
            status = "✅" if success else "❌"
            print(f"  {status} {test_name}")
    
    # Overall result
    all_results = [success for test_results in results.values() 
                   for _, success in test_results]
    if all_results and all(all_results):
        print("\n🎉 ALL TESTS PASSED!")
        return 0
    elif any(all_results):
        print("\n⚠️  SOME TESTS FAILED")
        return 1
    else:
        print("\n❌ NO TESTS RUN OR ALL FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())

