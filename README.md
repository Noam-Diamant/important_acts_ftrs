# Gradient Analysis Tools

This directory contains two complementary tools for analyzing how different representations affect language model loss through gradient analysis on Qwen2.5-0.5B.

## Overview

### 1. SAE Feature Gradient Analysis (`sae_gradient_analysis.py`)

Analyzes how Sparse Autoencoder (SAE) features affect the language model loss:
1. Loads Qwen2.5-0.5B model and corresponding SAEs for each layer
2. Processes text samples from WikiText dataset
3. For each layer:
   - Captures MLP output activations
   - Encodes activations through the SAE (65536 features)
   - Decodes back to activation space
   - Computes cross-entropy loss on next token prediction
   - Computes gradients of loss w.r.t. SAE feature dimensions
   - Aggregates gradients across samples
4. Generates visualization plots showing gradient distributions

### 2. Layer Activation Gradient Analysis (`layer_gradient_analysis.py`)

Analyzes how raw MLP layer activations affect the language model loss:
1. Loads Qwen2.5-0.5B model (no SAEs needed)
2. Processes text samples from WikiText dataset
3. For each layer:
   - Computes gradients of loss w.r.t. raw MLP output activations (896 dimensions)
   - Aggregates gradients across samples
4. Generates visualization plots showing gradient distributions

**Key Difference**: This tool analyzes the 896-dimensional raw MLP activations directly, without SAE encoding/decoding, making it faster and useful for understanding the raw layer representations.

## Aggregation Modes

Both tools support three aggregation modes for gradient analysis:

- **`mean_abs` (default)**: Mean of absolute gradient values
  - Shows overall gradient magnitude regardless of direction
  - All values are positive
  - Indicates which dimensions have the strongest effect on loss (in either direction)

- **`mean`**: Signed mean of gradients
  - Positive and negative gradients can cancel out
  - Shows net directional effect on loss
  - Useful for understanding whether increasing a dimension increases or decreases loss

- **`both`**: Computes both aggregation modes
  - Creates separate plots for each mode with `_mean` and `_mean_abs` suffixes
  - Also creates combined subplot figures showing both modes side-by-side
  - Allows direct comparison of magnitude vs. directional effects

## Usage

### SAE Feature Analysis

#### Basic Usage

Run with default settings (500 samples, all 24 layers, mean_abs aggregation):

```bash
cd /dsi/fetaya-lab/noam_diamant/projects/Unlearning_with_SAE
python important_feaures/sae_gradient_analysis.py --save_outputs
```

#### Custom Configuration

```bash
python important_feaures/sae_gradient_analysis.py \
    --num_samples 1000 \
    --batch_size 8 \
    --max_length 128 \
    --layers "0-23" \
    --aggregation_mode both \
    --save_outputs \
    --device cuda
```

#### Analyze Specific Layers with Different Aggregation Modes

```bash
# Analyze specific layers with signed mean
python important_feaures/sae_gradient_analysis.py \
    --layers "0,5,10,15,20,23" \
    --aggregation_mode mean \
    --save_outputs

# Analyze a range with both aggregation modes
python important_feaures/sae_gradient_analysis.py \
    --layers "10-15" \
    --aggregation_mode both \
    --save_outputs
```

### Layer Activation Analysis

#### Basic Usage

Run with default settings (500 samples, all 24 layers, mean_abs aggregation):

```bash
python important_feaures/layer_gradient_analysis.py --save_outputs
```

#### Custom Configuration

```bash
python important_feaures/layer_gradient_analysis.py \
    --num_samples 1000 \
    --batch_size 8 \
    --max_length 128 \
    --layers "0-23" \
    --aggregation_mode both \
    --save_outputs \
    --device cuda
```

## Command Line Arguments

Both scripts accept the following arguments:

- `--model_name`: Model to load (default: `Qwen/Qwen2.5-0.5B`)
- `--num_samples`: Number of text samples to process (default: 500)
- `--batch_size`: Batch size for processing (default: 4)
- `--max_length`: Maximum sequence length (default: 128)
- `--save_outputs`: Save outputs to automatically named directory (default: False)
- `--layers`: Layer range to analyze (default: `0-23`)
  - Format: `"0-23"` for range or `"0,5,10"` for specific layers
- `--device`: Device to use (default: `cuda`)
- `--aggregation_mode`: Gradient aggregation mode (default: `mean_abs`)
  - `mean`: Signed mean (shows directional effect)
  - `mean_abs`: Mean absolute value (shows magnitude)
  - `both`: Compute and plot both modes

## Output Files

### SAE Feature Analysis Output

When using single aggregation mode (`mean` or `mean_abs`):
- `layer_{i}_{mode}_gradient_hist.png`: Histogram of gradient distribution
- `layer_{i}_{mode}_feature_scatter.png`: Scatter plot of gradients per feature
- `layer_{i}_{mode}_gradients.npy`: Raw gradient data (NumPy array)
  - Shape: `[65536]` (one value per SAE feature)
- `all_layers_{mode}_gradient_histograms.png`: Combined histograms for all layers
- `all_layers_{mode}_feature_scatter.png`: Combined scatter plots for all layers

When using `both` aggregation mode:
- All of the above for both `_mean` and `_mean_abs`
- `layer_{i}_gradient_hist_combined.png`: Side-by-side comparison of both modes
- `layer_{i}_gradient_scatter_combined.png`: Side-by-side scatter comparison

### Layer Activation Analysis Output

When using single aggregation mode:
- `layer_{i}_{mode}_gradient_hist.png`: Histogram of gradient distribution
- `layer_{i}_{mode}_activation_scatter.png`: Scatter plot of gradients per dimension
- `layer_{i}_{mode}_gradients.npy`: Raw gradient data (NumPy array)
  - Shape: `[896]` (one value per activation dimension)
- `all_layers_{mode}_gradient_histograms.png`: Combined histograms for all layers
- `all_layers_{mode}_activation_scatter.png`: Combined scatter plots for all layers

When using `both` aggregation mode:
- All of the above for both `_mean` and `_mean_abs`
- `layer_{i}_gradient_hist_combined.png`: Side-by-side comparison
- `layer_{i}_gradient_scatter_combined.png`: Side-by-side scatter comparison

## Interpreting Results

### Gradient Magnitude Interpretation

The plots show which dimensions (SAE features or raw activations) are most "important" for the language model loss:

- **High absolute gradient**: Dimensions that strongly affect loss when perturbed
- **Low absolute gradient**: Dimensions with minimal impact on loss
- **Distribution shape**: Indicates concentration of important dimensions
  - Heavy tail suggests few dimensions are critical
  - Uniform distribution suggests many dimensions contribute equally

### Aggregation Mode Comparison

When using `both` mode, compare the two aggregations:

- **Mean Absolute (`mean_abs`)**:
  - Shows which dimensions have the strongest effect (regardless of direction)
  - All values are positive
  - Useful for identifying dimensions that matter most to the loss

- **Signed Mean (`mean`)**:
  - Shows net directional effect (can be positive or negative)
  - Positive: Increasing this dimension tends to increase loss
  - Negative: Increasing this dimension tends to decrease loss
  - Near zero: Either no effect OR positive and negative effects cancel out

- **Comparison Insights**:
  - High `mean_abs` but low `mean`: Dimension has strong but inconsistent effects
  - High in both: Dimension consistently affects loss in one direction
  - If `mean` ≈ 0 but `mean_abs` is high: The dimension's effect depends heavily on context

### SAE vs Layer Activation Analysis

- **SAE Analysis** (65536 dimensions): 
  - Shows importance of learned sparse features
  - Higher sparsity expected (few active features)
  - Useful for interpretability research

- **Layer Activation Analysis** (896 dimensions):
  - Shows importance of raw neural representations
  - More distributed importance expected
  - Faster to compute, no SAE dependency
  - Useful for understanding raw model representations

## Dependencies

Required packages:
- torch
- transformers
- datasets
- matplotlib
- numpy
- tqdm
- dictionary_learning (custom package in repo)

## Technical Details

### Gradient Computation

For each text sample and layer:
1. Forward pass captures activations at layer `i` MLP output
2. Encode: `features = SAE.encode(activations)` with `requires_grad=True`
3. Decode: `reconstructed = SAE.decode(features)`
4. Replace activations with reconstructed version
5. Compute cross-entropy loss on next token prediction
6. Compute: `∂loss/∂features` using `torch.autograd.grad()`
7. Store absolute gradient values

Aggregation:
- Average gradients across all sequence positions
- Average across all samples
- Result: one gradient magnitude per SAE feature

### Memory Optimization

To handle large models and datasets:
- Uses `torch.float32` for gradient computation
- Processes positions with stride (every 4th position)
- Batch processing with configurable batch size
- Releases hooks after each batch

## Example Output

### SAE Feature Analysis

```
Loading model: Qwen/Qwen2.5-0.5B
Loading SAEs for layers: [0, 1, 2, ..., 23]
Successfully loaded 24 SAEs
Loading WikiText dataset
Prepared 500 samples in 125 batches
Aggregation mode: both

Processing layer 0
Layer 0:
  Mean gradient (signed) = -0.000023, Max = 0.045678, Std = 0.001234
  Mean gradient (abs) = 0.000156, Max = 0.045678, Std = 0.001134
Saved histogram to layer_0_mean_gradient_hist.png
Saved histogram to layer_0_mean_abs_gradient_hist.png
Saved combined histogram plot to layer_0_gradient_hist_combined.png
...

Completed analysis for 24 layers
Results saved to model_Qwen_Qwen2.5-0.5B_num_samples_500_batch_size_4_layers_0to23_20241029_143022
```

### Layer Activation Analysis

```
Loading model: Qwen/Qwen2.5-0.5B
Loading WikiText dataset
Prepared 500 samples in 125 batches
Aggregation mode: mean_abs

Processing layer 0
Layer 0: Mean abs gradient = 0.002345, Max gradient = 0.123456, Std gradient = 0.003456
Saved histogram to layer_0_mean_abs_gradient_hist.png
Saved activation scatter plot to layer_0_mean_abs_activation_scatter.png
...

Completed analysis for 24 layers
Results saved to layer_activations_model_Qwen_Qwen2.5-0.5B_num_samples_500_batch_size_4_layers_0to23_20241029_143522
```

