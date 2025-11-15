# Gradient Analysis Tools

Two tools for analyzing how model representations affect language model loss across multiple architectures.

## Overview

**`sae_gradient_analysis.py`**: Analyzes gradients w.r.t. SAE features (e.g., 65,536 dimensions)
- Requires SAE models, slower but shows learned sparse features

**`layer_gradient_analysis.py`**: Analyzes gradients w.r.t. raw MLP activations (e.g., 896 dimensions)
- No SAE needed, faster, shows raw neural representations

## Supported Models

Both tools automatically detect and support:
- **Qwen** (Qwen2.5-0.5B, etc.)
- **Llama** (Llama-3.1-8B, etc.)
- **Mistral** (Mistral-7B-v0.1, etc.)
- **Zephyr** (Zephyr-7B-beta, etc.)
- **Mixtral** (Mixtral-8x7B-Instruct-v0.1, etc.)
- **Yi** (Yi-34B-Chat, etc.)
- **Gemma** (Gemma-2-2B, etc.)

The scripts automatically configure layer paths based on model architecture.

## Analysis Types

### Aggregate By: Values vs Gradients

- **`values`** (default): Aggregates raw activation/feature values - faster, simpler, no loss computation
- **`gradients`**: Aggregates loss gradients w.r.t. activations/features - shows impact on model loss

### Aggregation Modes

- **`mean_abs`** (default): Mean absolute values - shows magnitude of effect
- **`mean`**: Signed mean values - shows directional effect (positive/negative)
- **`both`**: Computes both modes, creates separate + combined comparison plots

### Power Transformation (Optional)

- Apply power transformation after aggregation, before normalization
- Can specify single value (e.g., `--power 2.0`) or comma-separated list (e.g., `--power 0.5,1.0,2.0`)
- When multiple values provided, only the first value is used (infrastructure for future multi-power analysis)
- Example: `--power 2.0` squares the aggregated values to emphasize large values
- Preserves sign for signed aggregation modes
- Power value is included in output directory name when specified

## Usage

### SAE Feature Analysis

#### Basic Usage

Run with default settings (500 samples, all 24 layers, mean_abs aggregation):

```bash
cd /dsi/fetaya-lab/noam_diamant/projects/Unlearning_with_SAE
python important_acts_ftrs/sae_gradient_analysis.py --save_outputs
```

#### Custom Configuration

```bash
python important_acts_ftrs/sae_gradient_analysis.py \
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
python important_acts_ftrs/sae_gradient_analysis.py \
    --layers "0,5,10,15,20,23" \
    --aggregation_mode mean \
    --save_outputs

# Analyze a range with both aggregation modes
python important_acts_ftrs/sae_gradient_analysis.py \
    --layers "10-15" \
    --aggregation_mode both \
    --save_outputs
```

### Layer Activation Analysis

#### Basic Usage

Run with default settings (500 samples, all 24 layers, mean_abs aggregation):

```bash
python important_acts_ftrs/layer_gradient_analysis.py --save_outputs
```

#### Custom Configuration

```bash
python important_acts_ftrs/layer_gradient_analysis.py \
    --num_samples 1000 \
    --batch_size 8 \
    --max_length 128 \
    --layers "0-23" \
    --aggregation_mode both \
    --save_outputs \
    --device cuda
```

#### Using Different Models

```bash
# Llama model
python important_acts_ftrs/layer_gradient_analysis.py \
    --model_name meta-llama/Llama-3.1-8B \
    --layers "0,31" \
    --save_outputs

# Mistral model
python important_acts_ftrs/layer_gradient_analysis.py \
    --model_name mistralai/Mistral-7B-v0.1 \
    --layers "0-31" \
    --save_outputs

# Gemma model
python important_acts_ftrs/layer_gradient_analysis.py \
    --model_name google/gemma-2-2b \
    --layers "0-25" \
    --save_outputs
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
- `--aggregate_by`: What to aggregate (default: `values`)
  - `values`: Raw activation/feature values (faster, no loss computation)
  - `gradients`: Loss gradients w.r.t. activations/features (shows impact on loss)
- `--aggregation_mode`: How to aggregate (default: `mean_abs`)
  - `mean`: Signed mean (shows directional effect)
  - `mean_abs`: Mean absolute value (shows magnitude)
  - `both`: Compute and plot both modes
- `--power`: Power transformation to apply after aggregation (optional, default: None)
  - Example: `2.0` squares values to emphasize larger ones
  - Preserves sign for signed modes
- `--normalize_gradients`: Normalization mode (default: `sum_abs`)
  - `none`: No normalization (raw values)
  - `sum_abs`: Sum of absolute values equals dimension size
  - `sum`: Sum of values equals dimension size
  - `both`: Apply both `sum` and `sum_abs` normalizations, save separate results
  - For SAE: 65,536 features, for layer activations: 896 dimensions
- `--save_consolidated`: Path to save consolidated .npz file (optional)
  - If provided, saves all results in a single compressed file
  - If not provided but `--save_outputs` is used, auto-saves to output directory

## Output Files

Results are saved to automatically-named directories:
- **SAE analysis**: `sae_features_model_{model}_num_samples_{N}_batch_size_{B}_layers_{L}_{timestamp}/`
- **Layer analysis**: `layer_activations_model_{model}_num_samples_{N}_batch_size_{B}_layers_{L}_{timestamp}/`

Each directory contains:
- **Per-layer plots**: Histograms and scatter plots showing gradient distributions
- **Combined plots**: All layers visualized together
- **Individual gradient files**: `.npy` arrays with gradient values (65,536 for SAE, 896 for activations)
- **Consolidated results**: `consolidated_results.npz` - all layers in single compressed file
- **Enhanced metadata**: `metadata.json` - comprehensive experiment information including:
  - Experiment configuration (model, samples, layers, normalization status)
  - Per-layer statistics (mean, std, max, sum of absolute values)
  - Dimension sizes for each layer
  - Timing information
  - Loading instructions
- **Timing files**: `timing_stats.json` and `timing_summary.txt`

### File Naming Conventions

Files use mode-specific suffixes:

**Aggregation mode only:**
- `layer_0_mean_gradients.npy` (signed mean)
- `layer_0_mean_abs_gradients.npy` (absolute mean)

**Normalization mode = `both`:**
- `layer_0_mean_abs_norm_sum_gradients.npy` (mean_abs with sum normalization)
- `layer_0_mean_abs_norm_sum_abs_gradients.npy` (mean_abs with sum_abs normalization)

**Both aggregation AND normalization = `both` (4 files per layer):**
- `layer_0_mean_norm_sum_gradients.npy`
- `layer_0_mean_norm_sum_abs_gradients.npy`
- `layer_0_mean_abs_norm_sum_gradients.npy`
- `layer_0_mean_abs_norm_sum_abs_gradients.npy`

With aggregation `both` mode, also creates `_combined` subplot figures comparing mean vs mean_abs.

### Understanding Mode Combinations

The two "both" options work independently and multiply the number of output variants:

| Aggregation Mode | Normalization Mode | Output Files per Layer |
|-----------------|-------------------|------------------------|
| `mean` | `sum_abs` | 1 file: `layer_X_mean_gradients.npy` |
| `mean` | `both` | 2 files: `layer_X_mean_norm_sum.npy`, `layer_X_mean_norm_sum_abs.npy` |
| `both` | `sum_abs` | 2 files: `layer_X_mean.npy`, `layer_X_mean_abs.npy` |
| `both` | `both` | **4 files**: all combinations of (mean, mean_abs) × (norm_sum, norm_sum_abs) |

**Example: 10 layers with `--aggregation_mode both --normalize_gradients both`:**
- 10 layers × 2 aggregations × 2 normalizations = **40 gradient files** total
- Plus plots, metadata, and consolidated .npz file

## Interpreting Results

### Value-Based Analysis (`--aggregate_by values`)

**High values**: Dimension is highly activated across samples  
**Low values**: Dimension is rarely activated

**`mean_abs`**: Shows which dimensions are most active (always positive)  
**`mean`**: Shows average activation (can be positive or negative)  
**High `mean_abs` but low `mean`**: Activation is strong but bidirectional (both positive and negative)

### Gradient-Based Analysis (`--aggregate_by gradients`)

**High gradient magnitude**: Dimension strongly affects loss  
**Low gradient magnitude**: Minimal impact on loss

**`mean_abs`**: Shows which dimensions matter most for loss (always positive)  
**`mean`**: Shows direction of effect (positive = increases loss, negative = decreases loss)  
**High `mean_abs` but low `mean`**: Effect is strong but context-dependent

## Gradient Normalization

The `--normalize_gradients` option provides four normalization modes to scale gradients for different analysis needs.

### Normalization Modes

**1. `none` - No Normalization**
- Returns raw gradient values as computed
- Preserves absolute scale of gradients
- Best for understanding actual gradient magnitudes

**2. `sum_abs` - Sum of Absolute Values (Default)**
- Scales gradients so `sum(abs(gradients)) = dimension_size`
- Preserves relative importance and signs
- Best for comparing importance across layers/models with different dimensions

**3. `sum` - Total Sum**
- Scales gradients so `sum(gradients) = dimension_size`
- Can amplify or diminish values based on overall gradient direction
- Useful when gradient direction balance is important

**4. `both` - Both Normalizations**
- Applies both `sum` and `sum_abs` normalizations
- Saves separate files for each normalization
- Creates separate plots for each variant
- Best when you want to compare different normalization effects side-by-side

### When to Use Each Mode

**Use `none` (no normalization) when:**
- Analyzing absolute gradient magnitudes across the entire model
- Understanding the actual scale of gradient effects
- Debugging or analyzing specific raw gradient values
- Comparing layers with the same dimension size

**Use `sum_abs` (default) when:**
- Comparing gradient importance across different layers or models
- Working with different dimension sizes (e.g., comparing SAE features vs raw activations)
- You want gradients to represent relative importance within each layer
- Building downstream models that need consistent input scales
- You want to preserve the sign of individual gradients

**Use `sum` when:**
- You need the total sum to equal dimension size (not absolute sum)
- Analyzing directional balance of gradients
- Working with algorithms that expect this specific normalization

**Use `both` when:**
- You're exploring which normalization works best for your analysis
- You want to compare the effects of different normalizations
- Building models and need to evaluate both normalization approaches
- Creating comprehensive analysis with all normalization variants

### Example Usage

```bash
# Default: sum_abs normalization
python important_acts_ftrs/sae_gradient_analysis.py \
    --normalize_gradients sum_abs \
    --save_outputs

# No normalization - raw gradient values
python important_acts_ftrs/sae_gradient_analysis.py \
    --normalize_gradients none \
    --save_outputs

# Sum normalization
python important_acts_ftrs/sae_gradient_analysis.py \
    --normalize_gradients sum \
    --save_outputs

# Both normalizations - saves 2 variants
python important_acts_ftrs/sae_gradient_analysis.py \
    --normalize_gradients both \
    --save_outputs

# Both normalizations + both aggregations = 4 variants total
python important_acts_ftrs/sae_gradient_analysis.py \
    --normalize_gradients both \
    --aggregation_mode both \
    --save_outputs
```

### Normalization Results

After `sum_abs` normalization:
- **SAE features**: `sum(abs(gradients)) = 65,536` (number of features)
- **Layer activations**: `sum(abs(gradients)) = 896` (activation dimension)

After `sum` normalization:
- **SAE features**: `sum(gradients) = 65,536`
- **Layer activations**: `sum(gradients) = 896`

### Example with Numbers

```python
# Original gradients (3 dimensions)
gradients = [0.5, -0.3, 0.2]
dimension_size = 3

# Mode: none
# Result: [0.5, -0.3, 0.2]

# Mode: sum_abs
# sum(abs([0.5, -0.3, 0.2])) = 1.0
# scale = 3 / 1.0 = 3.0
# Result: [1.5, -0.9, 0.6]  → sum(abs) = 3.0 ✓

# Mode: sum
# sum([0.5, -0.3, 0.2]) = 0.4
# scale = 3 / 0.4 = 7.5
# Result: [3.75, -2.25, 1.5]  → sum = 3.0 ✓
```

## Values vs Gradients Analysis

### When to Use `--aggregate_by values` (Default)

**Use value-based analysis when:**
- You want to understand which dimensions are naturally active
- You're doing exploratory analysis of activation patterns
- You want faster execution (no loss computation or backpropagation)
- You're identifying "hot" features/dimensions that fire frequently
- You're building activation-based representations or embeddings
- You don't need causal information about loss impact

**Example:**
```bash
# Fast analysis of which SAE features are most active
python important_acts_ftrs/sae_gradient_analysis.py \
    --aggregate_by values \
    --aggregation_mode mean_abs \
    --save_outputs
```

### When to Use `--aggregate_by gradients`

**Use gradient-based analysis when:**
- You want to understand which dimensions affect model loss
- You're doing mechanistic interpretability (causal analysis)
- You want to identify dimensions that matter for model predictions
- You're building loss-informed pruning or compression strategies
- You need to know the direction of effect (increases/decreases loss)
- You're studying how interventions affect model behavior

**Example:**
```bash
# Analyze which features causally affect loss
python important_acts_ftrs/sae_gradient_analysis.py \
    --aggregate_by gradients \
    --aggregation_mode both \
    --save_outputs
```

### Comparison: Values vs Gradients

| Aspect | Values | Gradients |
|--------|--------|-----------|
| **Speed** | Fast (no backprop) | Slower (requires backprop) |
| **Memory** | Low | Higher |
| **Interpretation** | Activation frequency/magnitude | Causal effect on loss |
| **Use Case** | Exploratory, activation patterns | Mechanistic, causal analysis |
| **Best for** | Understanding what fires | Understanding what matters |

### Example: Different Results

A dimension can have:
- **High values, low gradients**: Fires frequently but doesn't affect loss much
- **Low values, high gradients**: Rarely fires but has strong effect when it does
- **High both**: Important and frequently activated
- **Low both**: Not important and rarely activated

## Power Transformation

The `--power` option applies an element-wise power transformation **after aggregation** but **before normalization**.

### When to Use Power Transformation

**Use power transformation when:**
- You want to emphasize large values and de-emphasize small ones
- You're identifying the most important dimensions (outlier detection)
- You want to apply non-linear scaling to your analysis
- You're building weighted representations where large values matter more

### Common Power Values

- **`--power 2.0`**: Square values (quadratic emphasis)
  - Good for emphasizing outliers
  - Common in many ML applications
- **`--power 0.5`**: Square root (compression)
  - Reduces the impact of very large values
  - Makes distribution more uniform
- **`--power 3.0`**: Cubic (strong emphasis)
  - Very strong emphasis on largest values
  - Useful for identifying clear winners

### Examples

```bash
# Emphasize important features with squaring
python important_acts_ftrs/sae_gradient_analysis.py \
    --aggregate_by values \
    --power 2.0 \
    --save_outputs

# Strong emphasis on outliers
python important_acts_ftrs/layer_gradient_analysis.py \
    --aggregate_by gradients \
    --power 3.0 \
    --aggregation_mode mean_abs \
    --save_outputs

# Compress large values
python important_acts_ftrs/sae_gradient_analysis.py \
    --aggregate_by values \
    --power 0.5 \
    --save_outputs

# Specify multiple power values (uses first value)
python important_acts_ftrs/layer_gradient_analysis.py \
    --aggregate_by values \
    --power "0.5,1.0,2.0" \
    --save_outputs
# Output: "Multiple power values provided: [0.5, 1.0, 2.0]"
# Output: "Using first power value: 0.5"
```

### Power Transformation with Signed Values

For signed aggregation modes (`aggregation_mode=mean`), power is applied while preserving sign:
```python
# For signed mean mode
result = sign(value) * |value|^power
```

For absolute aggregation modes (`aggregation_mode=mean_abs`), power is applied directly:
```python
# For mean_abs mode (values already positive)
result = value^power
```

### Example with Numbers

```python
# Original aggregated values
values = [10.0, 5.0, 1.0, -8.0]

# Power = 2.0, signed mode
# Result: [100.0, 25.0, 1.0, -64.0]
# Large values emphasized, signs preserved

# Power = 0.5, signed mode  
# Result: [3.16, 2.24, 1.0, -2.83]
# Large values compressed, signs preserved

# Power = 2.0, mean_abs mode (all positive)
values_abs = [10.0, 5.0, 1.0, 8.0]
# Result: [100.0, 25.0, 1.0, 64.0]
```

### Complete Example: All Features Together

```bash
# Comprehensive analysis:
# - Aggregate raw values (faster)
# - Square them to emphasize important ones
# - Use both signed and absolute aggregation
# - Apply both normalizations for comparison
python important_acts_ftrs/sae_gradient_analysis.py \
    --aggregate_by values \
    --power 2.0 \
    --aggregation_mode both \
    --normalize_gradients both \
    --num_samples 1000 \
    --save_outputs

# This creates 4 variants per layer:
# - mean_norm_sum, mean_norm_sum_abs
# - mean_abs_norm_sum, mean_abs_norm_sum_abs
# All with power=2.0 applied before normalization
```

## Loading Saved Results

Results can be loaded in multiple ways depending on your needs:

### Option 1: Load Consolidated .npz File (Recommended)

Load all layers at once from the consolidated file:

```python
import numpy as np

# Load the consolidated results file
data = np.load('path/to/consolidated_results.npz', allow_pickle=True)

# Access metadata
metadata = data['metadata'].item()
print(f"Model: {metadata['model_name']}")
print(f"Aggregate by: {metadata['aggregate_by']}")
print(f"Power: {metadata['power']}")
print(f"Normalization mode: {metadata['normalization_mode']}")
print(f"Dimension sizes: {metadata['dimension_sizes']}")

# Access gradient data for specific layers and modes
layer_0_mean_abs = data['layer_0_mean_abs']
layer_5_mean = data['layer_5_mean']

# With normalize_gradients='both', access specific normalizations
layer_0_mean_norm_sum = data['layer_0_mean_norm_sum']
layer_0_mean_norm_sum_abs = data['layer_0_mean_norm_sum_abs']

# With both aggregation AND normalization = 'both'
layer_0_mean_norm_sum = data['layer_0_mean_norm_sum']
layer_0_mean_norm_sum_abs = data['layer_0_mean_norm_sum_abs']
layer_0_mean_abs_norm_sum = data['layer_0_mean_abs_norm_sum']
layer_0_mean_abs_norm_sum_abs = data['layer_0_mean_abs_norm_sum_abs']

# See all available layers and modes
available_keys = [key for key in data.keys() if key != 'metadata']
print(f"Available data: {available_keys}")
```

### Option 2: Use the Helper Function

Both scripts include a `load_results()` helper function:

```python
import sys
sys.path.append('path/to/important_acts_ftrs')
from sae_gradient_analysis import load_results  # or layer_gradient_analysis

# Load consolidated file
results = load_results('path/to/consolidated_results.npz')

# Access data
gradients = results['data']
metadata = results['metadata']

# Work with specific layer
layer_0 = gradients['layer_0_mean_abs']

# Or load from directory with individual .npy files
results = load_results('path/to/output_directory')
```

### Option 3: Load Individual .npy Files

Load specific layers individually (useful for large datasets):

```python
import numpy as np

# Load specific layer with single aggregation/normalization
layer_0_gradients = np.load('path/to/output_dir/layer_0_mean_abs_gradients.npy')
layer_5_gradients = np.load('path/to/output_dir/layer_5_mean_gradients.npy')

# Load specific normalization variant (when using normalize_gradients='both')
layer_0_norm_sum = np.load('path/to/output_dir/layer_0_mean_abs_norm_sum_gradients.npy')
layer_0_norm_sum_abs = np.load('path/to/output_dir/layer_0_mean_abs_norm_sum_abs_gradients.npy')

# Load all 4 variants (when using both aggregation AND normalization = 'both')
layer_0_mean_norm_sum = np.load('path/to/output_dir/layer_0_mean_norm_sum_gradients.npy')
layer_0_mean_norm_sum_abs = np.load('path/to/output_dir/layer_0_mean_norm_sum_abs_gradients.npy')
layer_0_mean_abs_norm_sum = np.load('path/to/output_dir/layer_0_mean_abs_norm_sum_gradients.npy')
layer_0_mean_abs_norm_sum_abs = np.load('path/to/output_dir/layer_0_mean_abs_norm_sum_abs_gradients.npy')

# Load metadata separately
import json
with open('path/to/output_dir/metadata.json', 'r') as f:
    metadata = json.load(f)
    
# Check analysis configuration
config = metadata['analysis_config']
print(f"Aggregate by: {config['aggregate_by']}")
print(f"Power: {config['power']}")
print(f"Normalization mode: {config['normalization_mode']}")
```

### Option 4: Custom Path with --save_consolidated

Save to a specific location:

```bash
# Save consolidated results to custom path
python important_acts_ftrs/sae_gradient_analysis.py \
    --save_outputs \
    --save_consolidated /path/to/my_results.npz

# Load later
import numpy as np
data = np.load('/path/to/my_results.npz', allow_pickle=True)
```

### Choosing the Right Loading Method

- **Consolidated .npz** (Option 1 or 2): Best for loading multiple layers, smaller file footprint, faster I/O
- **Individual .npy** (Option 3): Best when you need only specific layers, more flexible for selective loading
- **Custom path** (Option 4): Best for organizing results in your own directory structure

