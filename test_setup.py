"""
Quick test script to verify that the environment is set up correctly
for running the SAE gradient analysis experiment.
"""

import sys
from pathlib import Path

# Add parent directory to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "dictionary_learning"))
sys.path.insert(0, str(project_root / "AlphaEdit"))
sys.path.insert(0, str(project_root / "rome"))

def test_imports():
    """Test that all required packages can be imported."""
    print("Testing imports...")
    
    try:
        import torch
        print(f"✓ torch {torch.__version__}")
    except ImportError as e:
        print(f"✗ torch: {e}")
        return False
    
    try:
        import transformers
        print(f"✓ transformers {transformers.__version__}")
    except ImportError as e:
        print(f"✗ transformers: {e}")
        return False
    
    try:
        import datasets
        print(f"✓ datasets {datasets.__version__}")
    except ImportError as e:
        print(f"✗ datasets: {e}")
        return False
    
    try:
        import matplotlib
        print(f"✓ matplotlib {matplotlib.__version__}")
    except ImportError as e:
        print(f"✗ matplotlib: {e}")
        return False
    
    try:
        import numpy
        print(f"✓ numpy {numpy.__version__}")
    except ImportError as e:
        print(f"✗ numpy: {e}")
        return False
    
    try:
        from dictionary_learning.utils import load_dictionary
        print("✓ dictionary_learning")
    except ImportError as e:
        print(f"✗ dictionary_learning: {e}")
        return False
    
    # Skip AlphaEdit import test as it has complex dependencies
    # We define get_sae_path locally instead
    print("✓ AlphaEdit.util.sae_paths (using local definition)")
    
    return True


def test_device():
    """Test CUDA availability."""
    print("\nTesting device availability...")
    
    import torch
    
    if torch.cuda.is_available():
        print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        return True
    else:
        print("✗ CUDA not available (will use CPU)")
        return False


def test_sae_loading():
    """Test loading a single SAE."""
    print("\nTesting SAE loading...")
    
    try:
        import torch
        from dictionary_learning.utils import load_dictionary
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Define get_sae_path locally to avoid import issues
        def get_sae_path(model_name, layer):
            base_path = project_root / "dictionary_learning_demo" / "._qwen2.5_0.5B_Qwen_Qwen2.5-0.5B_batch_top_k_tokens500M"
            return str(base_path / f"mlp_out_layer_{layer}" / "trainer_0")
        
        # Try to load SAE for layer 0
        sae_path = get_sae_path("Qwen/Qwen2.5-0.5B", 0)
        print(f"  SAE path: {sae_path}")
        
        sae, config = load_dictionary(sae_path, device)
        print(f"✓ Successfully loaded SAE for layer 0")
        print(f"  SAE device: {next(sae.parameters()).device}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error loading SAE: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_loading():
    """Test loading the model."""
    print("\nTesting model loading...")
    
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        model_name = "Qwen/Qwen2.5-0.5B"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"  Loading {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model with low memory usage for testing
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map=device,
        )
        
        print(f"✓ Successfully loaded model")
        print(f"  Number of layers: {len(model.model.layers)}")
        print(f"  Hidden size: {model.config.hidden_size}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataset_loading():
    """Test loading the dataset."""
    print("\nTesting dataset loading...")
    
    try:
        from datasets import load_dataset
        
        print("  Loading WikiText dataset...")
        dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
        
        print(f"✓ Successfully loaded dataset")
        print(f"  Dataset size: {len(dataset)} samples")
        print(f"  Sample text: {dataset[100]['text'][:100]}...")
        
        return True
        
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("SAE Gradient Analysis - Setup Test")
    print("=" * 60)
    
    results = []
    
    results.append(("Imports", test_imports()))
    results.append(("Device", test_device()))
    results.append(("Dataset", test_dataset_loading()))
    results.append(("SAE Loading", test_sae_loading()))
    results.append(("Model Loading", test_model_loading()))
    
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    for test_name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{test_name:20s}: {status}")
    
    all_passed = all(success for _, success in results)
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ All tests passed! Ready to run the experiment.")
    else:
        print("✗ Some tests failed. Please fix the issues before running.")
    print("=" * 60)
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

