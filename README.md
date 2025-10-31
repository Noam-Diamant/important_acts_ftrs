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
    --layers "0-31" \
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

## Output Files

Results are saved to automatically-named directories:
- **SAE analysis**: `sae_features_model_{model}_num_samples_{N}_batch_size_{B}_layers_{L}_{timestamp}/`
- **Layer analysis**: `layer_activations_model_{model}_num_samples_{N}_batch_size_{B}_layers_{L}_{timestamp}/`

Each directory contains:
- **Per-layer plots**: Histograms and scatter plots showing gradient distributions
- **Combined plots**: All layers visualized together
- **Data files**: `.npy` arrays with gradient values (65,536 for SAE, 896 for activations)

Files use `_{mode}` suffix (`_mean` or `_mean_abs`). With `both` mode, also creates `_combined` subplot figures.

## Interpreting Results

**High gradient magnitude**: Dimension strongly affects loss  
**Low gradient magnitude**: Minimal impact on loss

**`mean_abs`**: Shows which dimensions matter most (always positive)  
**`mean`**: Shows direction of effect (positive = increases loss, negative = decreases loss)  
**High `mean_abs` but low `mean`**: Effect is strong but context-dependent

