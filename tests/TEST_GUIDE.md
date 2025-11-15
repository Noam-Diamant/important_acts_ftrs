# Testing Guide for Gradient Analysis Tools

This guide explains how to use the test scripts to verify the functionality of `layer_gradient_analysis.py` and `sae_gradient_analysis.py`.

## Test Files

1. **`quick_test.py`** - Fast single test for manual verification
2. **`test_gradient_analysis.py`** - Comprehensive test suite with multiple configurations

## Quick Test (Recommended for Manual Verification)

### Basic Usage

Test with minimal data (5 samples, 1 layer) for quick verification:

```bash
# Test layer analysis with values (fastest)
python quick_test.py --script layer --aggregate_by values

# Test layer analysis with gradients
python quick_test.py --script layer --aggregate_by gradients

# Test with power transformation
python quick_test.py --script layer --aggregate_by values --power 2.0

# Test SAE analysis (requires SAE models)
python quick_test.py --script sae --aggregate_by values
```

### Advanced Options

```bash
# Test with more samples and layers
python quick_test.py \
    --script layer \
    --aggregate_by values \
    --power 2.0 \
    --aggregation_mode both \
    --normalize sum_abs \
    --num_samples 20 \
    --layers "0,5"

# Test different power values
python quick_test.py --script layer --power 0.5  # Compress large values
python quick_test.py --script layer --power 2.0  # Emphasize large values
python quick_test.py --script layer --power 3.0  # Strong emphasis
```

### What to Look For

The quick test will show:
- ✅ Configuration verification (aggregate_by, power, normalization)
- ✅ Dimension sizes
- ✅ Generated files
- ✅ Sample statistics (mean, std, min, max)
- ✅ First 10 values
- ✅ Power transformation verification
- ✅ Normalization verification

### Example Output

```
==============================================================================
QUICK TEST - Gradient Analysis
==============================================================================

Script: layer_gradient_analysis.py
Configuration:
  - Aggregate by: values
  - Aggregation mode: mean_abs
  - Power: 2.0
  - Normalization: sum_abs
  - Samples: 5
  - Layers: 0

...

📁 Output directory: layer_activations_model_Qwen_Qwen2.5-0.5B_...

🔍 Inspecting results:

  Metadata verification:
    ✓ aggregate_by: values
    ✓ power: 2.0
    ✓ aggregation_mode: mean_abs
    ✓ normalization_mode: sum_abs

  Dimension sizes:
    - layer_0: 896 dimensions

  Generated files: 1 .npy files

  Sample data from first file:
    File: layer_0_mean_abs_gradients.npy
    Shape: (896,)
    Mean: 1.234567
    Std: 2.345678
    Min: 0.001234
    Max: 12.345678
    First 10 values: [1.23 2.34 3.45 ...]

  Power transformation verification:
    Applied power: 2.0
    ✓ Large values should be emphasized

  Normalization verification (sum_abs):
    Sum of absolute values: 896.0
    Target (dimension size): 896
    ✓ Normalization correct!

✅ Test completed successfully!
```

## Comprehensive Test Suite

### Running All Tests

```bash
# Run full test suite
python test_gradient_analysis.py
```

This will:
1. Run 5 different test configurations on `layer_gradient_analysis.py`
2. Optionally run the same tests on `sae_gradient_analysis.py`
3. Generate a comprehensive test report

### Test Configurations

The suite tests:
1. **Values aggregation, mean_abs, no power**
2. **Values aggregation, mean_abs, power=2.0**
3. **Gradients aggregation, mean, no power**
4. **Values aggregation, both modes, power=0.5**
5. **Gradients aggregation, mean_abs, both normalizations**

### What It Tests

- ✅ Basic functionality (values vs gradients)
- ✅ Power transformations (None, 0.5, 2.0)
- ✅ Aggregation modes (mean, mean_abs, both)
- ✅ Normalization modes (none, sum_abs, both)
- ✅ File generation
- ✅ Metadata storage
- ✅ Output verification

## Testing Specific Features

### Test Value Aggregation (New Feature)

```bash
# Fast value aggregation (no gradients computed)
python quick_test.py --script layer --aggregate_by values --num_samples 5

# Compare with gradient aggregation (slower)
python quick_test.py --script layer --aggregate_by gradients --num_samples 5
```

**Expected behavior:**
- `values`: Much faster, no loss computation
- `gradients`: Slower, includes backpropagation

### Test Power Transformation (New Feature)

```bash
# No power (baseline)
python quick_test.py --script layer --power none --num_samples 5

# Square values (emphasize large)
python quick_test.py --script layer --power 2.0 --num_samples 5

# Square root (compress large)
python quick_test.py --script layer --power 0.5 --num_samples 5
```

**Expected behavior:**
- `power=2.0`: Max value should be much larger relative to mean
- `power=0.5`: Max value should be closer to mean
- Distribution should change accordingly

### Test Normalization

```bash
# No normalization
python quick_test.py --normalize none

# Sum of absolute values = dimension size
python quick_test.py --normalize sum_abs

# Sum = dimension size
python quick_test.py --normalize sum

# Both normalizations
python quick_test.py --normalize both
```

**Expected behavior:**
- `sum_abs`: `np.sum(np.abs(data))` should equal dimension size
- `sum`: `np.sum(data)` should equal dimension size
- `both`: Creates 2 files per layer

## Manual Verification Checklist

After running tests, verify:

### 1. Configuration Storage
- [ ] Check `metadata.json` contains `aggregate_by` field
- [ ] Check `metadata.json` contains `power` field
- [ ] Values match command line arguments

### 2. Output Files
- [ ] Correct number of `.npy` files generated
- [ ] Files have expected naming pattern
- [ ] File sizes are reasonable

### 3. Data Validation
- [ ] Array shapes are correct (e.g., 896 for layer activations)
- [ ] Values are in reasonable ranges
- [ ] No NaN or Inf values
- [ ] Statistics (mean, std) make sense

### 4. Power Transformation
- [ ] When `power > 1`: Large values emphasized
- [ ] When `power < 1`: Large values compressed
- [ ] Signs preserved for signed modes

### 5. Normalization
- [ ] `sum_abs`: Sum of absolute values ≈ dimension size
- [ ] `sum`: Sum of values ≈ dimension size
- [ ] `none`: No normalization applied

### 6. Performance
- [ ] `aggregate_by=values` faster than `gradients`
- [ ] Execution completes within timeout
- [ ] No memory errors

## Debugging Tips

### If tests fail:

1. **Check CUDA availability**
   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```

2. **Check model loading**
   ```bash
   python -c "from transformers import AutoModel; AutoModel.from_pretrained('Qwen/Qwen2.5-0.5B')"
   ```

3. **Run with verbose output**
   ```bash
   # Add print statements or run directly:
   python layer_gradient_analysis.py --num_samples 1 --layers "0" --save_outputs
   ```

4. **Check individual components**
   ```python
   # Test in Python directly
   import numpy as np
   
   # Test power transformation
   data = np.array([1, 2, 3, -4])
   power = 2.0
   result = np.power(np.abs(data), power) * np.sign(data)
   print(result)  # Should be [1, 4, 9, -16]
   ```

5. **Verify file permissions**
   ```bash
   ls -la layer_gradient_analysis.py
   ls -la sae_gradient_analysis.py
   ```

## Common Issues

### Issue: "No SAE models found"
- **Solution**: SAE tests require trained SAE models. Skip SAE tests if not available.

### Issue: "CUDA out of memory"
- **Solution**: Reduce `--num_samples` or `--batch_size`

### Issue: "Timeout"
- **Solution**: Reduce number of samples or layers, or increase timeout in test script

### Issue: "Import errors"
- **Solution**: Ensure you're in the correct directory and all dependencies are installed

## Example Test Session

```bash
# 1. Quick smoke test
python quick_test.py --script layer --num_samples 5

# 2. Test new features
python quick_test.py --script layer --aggregate_by values --power 2.0

# 3. Compare values vs gradients
python quick_test.py --script layer --aggregate_by values --num_samples 10
python quick_test.py --script layer --aggregate_by gradients --num_samples 10

# 4. Run full test suite if all quick tests pass
python test_gradient_analysis.py
```

## Success Criteria

Tests are successful if:
- ✅ All test configurations run without errors
- ✅ Output files are generated correctly
- ✅ Metadata contains all expected fields
- ✅ Data statistics are reasonable
- ✅ Normalization is applied correctly
- ✅ Power transformation works as expected
- ✅ `aggregate_by=values` is faster than `gradients`

## Additional Resources

- Main README: `README.md`
- Script documentation: See docstrings in scripts
- Example usage: See examples in `README.md`

