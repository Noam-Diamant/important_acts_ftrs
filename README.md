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

## Aggregation Modes

- **`mean_abs`** (default): Mean absolute gradient - shows magnitude of effect
- **`mean`**: Signed mean gradient - shows directional effect (positive/negative)
- **`both`**: Computes both modes, creates separate + combined comparison plots

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
- `--aggregation_mode`: Gradient aggregation mode (default: `mean_abs`)
  - `mean`: Signed mean (shows directional effect)
  - `mean_abs`: Mean absolute value (shows magnitude)
  - `both`: Compute and plot both modes
- `--normalize_gradients`: Normalize gradients by dimension size (default: False)
  - When enabled, sum of absolute gradient values equals the dimension size
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

Files use `_{mode}` suffix (`_mean` or `_mean_abs`). With `both` mode, also creates `_combined` subplot figures.

## Interpreting Results

**High gradient magnitude**: Dimension strongly affects loss  
**Low gradient magnitude**: Minimal impact on loss

**`mean_abs`**: Shows which dimensions matter most (always positive)  
**`mean`**: Shows direction of effect (positive = increases loss, negative = decreases loss)  
**High `mean_abs` but low `mean`**: Effect is strong but context-dependent

## Gradient Normalization

The `--normalize_gradients` flag scales gradients so that the sum of absolute values equals the dimension size. This is useful for:

### When to Use Normalization

**Use normalized gradients when:**
- Comparing gradient importance across different layers or models
- Working with different dimension sizes (e.g., comparing SAE features vs raw activations)
- You want gradients to represent relative importance within each layer
- Building downstream models that need consistent input scales

**Use raw gradients when:**
- Analyzing absolute gradient magnitudes across the entire model
- Understanding the actual scale of gradient effects
- Debugging or analyzing specific gradient values

### Example Usage

```bash
# With normalization - sum of abs(gradients) will equal 65,536 for SAE
python important_acts_ftrs/sae_gradient_analysis.py \
    --normalize_gradients \
    --save_outputs

# Without normalization - raw gradient values
python important_acts_ftrs/sae_gradient_analysis.py \
    --save_outputs
```

After normalization:
- **SAE features**: `sum(abs(gradients)) = 65,536` (number of features)
- **Layer activations**: `sum(abs(gradients)) = 896` (activation dimension)

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
print(f"Normalized: {metadata['normalized']}")
print(f"Dimension sizes: {metadata['dimension_sizes']}")

# Access gradient data for specific layers
layer_0_mean_abs = data['layer_0_mean_abs']
layer_5_mean = data['layer_5_mean']

# See all available layers
available_layers = [key for key in data.keys() if key != 'metadata']
print(f"Available layers: {available_layers}")
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

# Load specific layer
layer_0_gradients = np.load('path/to/output_dir/layer_0_mean_abs_gradients.npy')
layer_5_gradients = np.load('path/to/output_dir/layer_5_mean_gradients.npy')

# Load metadata separately
import json
with open('path/to/output_dir/metadata.json', 'r') as f:
    metadata = json.load(f)
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

