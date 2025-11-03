"""
Layer Activation Gradient Analysis Experiment

This script analyzes how layer MLP output activations affect the language model loss by computing
gradients and visualizing their distributions across all 24 layers of Qwen2.5-0.5B.

Unlike the SAE feature analysis, this script computes gradients directly w.r.t. the raw
896-dimensional MLP output activations, without any SAE encoding/decoding.
"""

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Add parent directory to path to import from other modules
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def get_model_config(model_name: str) -> Dict[str, str]:
    """Get model-specific configuration.
    
    Args:
        model_name: Name of the model
    
    Returns:
        Dictionary with model configuration
    """
    model_lower = model_name.lower()
    
    # Determine model architecture
    if 'qwen' in model_lower:
        return {
            'layer_path': 'model.layers.{}.mlp',
            'model_type': 'qwen'
        }
    elif 'llama' in model_lower:
        return {
            'layer_path': 'model.layers.{}.mlp',
            'model_type': 'llama'
        }
    elif 'mistral' in model_lower or 'zephyr' in model_lower:
        return {
            'layer_path': 'model.layers.{}.mlp',
            'model_type': 'mistral'
        }
    elif 'mixtral' in model_lower:
        return {
            'layer_path': 'model.layers.{}.mlp',
            'model_type': 'mixtral'
        }
    elif 'yi' in model_lower:
        return {
            'layer_path': 'model.layers.{}.mlp',
            'model_type': 'yi'
        }
    elif 'gemma' in model_lower:
        return {
            'layer_path': 'model.layers.{}.mlp',
            'model_type': 'gemma'
        }
    else:
        # Default to standard transformer architecture
        return {
            'layer_path': 'model.layers.{}.mlp',
            'model_type': 'unknown'
        }


def get_layer_module(model, layer_idx: int, model_config: Dict[str, str]):
    """Get the MLP module for a specific layer.
    
    Args:
        model: The language model
        layer_idx: Layer index
        model_config: Model configuration dictionary
    
    Returns:
        The MLP module for the specified layer
    """
    layer_path = model_config['layer_path'].format(layer_idx)
    parts = layer_path.split('.')
    
    module = model
    for part in parts:
        module = getattr(module, part)
    
    return module


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Analyze layer activation gradients across layers')
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen2.5-0.5B',
                        help='Model name to load')
    parser.add_argument('--num_samples', type=int, default=500,
                        help='Number of text samples to process (default: 500)')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for processing (default: 4)')
    parser.add_argument('--max_length', type=int, default=128,
                        help='Maximum sequence length (default: 128)')
    parser.add_argument('--save_outputs', action='store_true',
                        help='Save outputs to automatically named directory (default: False)')
    parser.add_argument('--layers', type=str, default='0-23',
                        help='Layer range to analyze (e.g., "0-23" or "0,5,10")')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (default: cuda)')
    parser.add_argument('--aggregation_mode', type=str, default='mean_abs',
                        choices=['mean', 'mean_abs', 'both'],
                        help='Aggregation mode for gradients: mean (signed), mean_abs (absolute), or both (default: mean_abs)')
    parser.add_argument('--normalize_gradients', type=str, default='sum_abs',
                        choices=['none', 'sum_abs', 'sum', 'both'],
                        help='Normalization mode: none (no normalization), sum_abs (sum of absolute values = dim_size), sum (sum = dim_size), both (apply both sum and sum_abs) (default: sum_abs)')
    parser.add_argument('--save_consolidated', type=str, default=None,
                        help='Path to save consolidated .npz file with all results (optional)')
    return parser.parse_args()


def parse_layer_range(layer_spec: str) -> List[int]:
    """Parse layer specification into list of layer indices.
    
    Args:
        layer_spec: String like "0-23" or "0,5,10,15"
    
    Returns:
        List of layer indices
    """
    if '-' in layer_spec:
        start, end = map(int, layer_spec.split('-'))
        return list(range(start, end + 1))
    else:
        return [int(x.strip()) for x in layer_spec.split(',')]


def normalize_gradients(gradients: np.ndarray, dimension_size: int, mode: str = 'sum_abs') -> np.ndarray:
    """Normalize gradients based on the specified mode.
    
    Args:
        gradients: Array of gradient values
        dimension_size: Target dimension size for normalization
        mode: Normalization mode ('none', 'sum_abs', or 'sum')
            - 'none': No normalization, returns original gradients
            - 'sum_abs': Normalize so sum(abs(gradients)) == dimension_size
            - 'sum': Normalize so sum(gradients) == dimension_size
    
    Returns:
        Normalized gradient array based on the specified mode
    """
    if mode == 'none':
        return gradients
    
    elif mode == 'sum_abs':
        abs_sum = np.sum(np.abs(gradients))
        if abs_sum == 0:
            print("Warning: Sum of absolute gradients is zero, returning original gradients")
            return gradients
        normalized = gradients * (dimension_size / abs_sum)
        return normalized
    
    elif mode == 'sum':
        total_sum = np.sum(gradients)
        if total_sum == 0:
            print("Warning: Sum of gradients is zero, returning original gradients")
            return gradients
        normalized = gradients * (dimension_size / total_sum)
        return normalized
    
    else:
        raise ValueError(f"Unknown normalization mode: {mode}. Use 'none', 'sum_abs', or 'sum'")


def load_model_and_tokenizer(model_name: str, device: str):
    """Load the language model and tokenizer.
    
    Args:
        model_name: Name of the model to load
        device: Device to load model on
    
    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Set padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,  # Use float32 for gradient computation
        device_map=device,
    )
    model.eval()
    
    return model, tokenizer


def prepare_dataset(tokenizer, num_samples: int, max_length: int, batch_size: int):
    """Prepare WikiText dataset for processing.
    
    Args:
        tokenizer: Tokenizer to use
        num_samples: Number of samples to process
        max_length: Maximum sequence length
        batch_size: Batch size
    
    Returns:
        DataLoader with tokenized samples
    """
    print("Loading WikiText dataset")
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    
    # Filter out empty or very short texts
    dataset = dataset.filter(lambda x: len(x['text'].strip()) > 50)
    
    # Take only the number of samples we need
    if len(dataset) > num_samples:
        dataset = dataset.select(range(num_samples))
    
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )
    
    # Tokenize dataset
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    
    # Create dataloader
    dataloader = DataLoader(tokenized_dataset, batch_size=batch_size, shuffle=True)
    
    print(f"Prepared {len(tokenized_dataset)} samples in {len(dataloader)} batches")
    return dataloader


def compute_gradients_for_layer(
    model,
    tokenizer,
    dataloader,
    layer_idx: int,
    device: str,
    aggregation_mode: str,
    model_config: Dict[str, str],
    normalize: str = 'none'
) -> Union[np.ndarray, Dict[str, np.ndarray]]:
    """Compute gradients of loss w.r.t. raw MLP output activations for a specific layer.
    
    Args:
        model: Language model
        tokenizer: Tokenizer
        dataloader: DataLoader with text samples
        layer_idx: Index of the layer to analyze
        device: Device to use
        aggregation_mode: 'mean', 'mean_abs', or 'both'
        model_config: Model configuration dictionary
        normalize: Normalization mode ('none', 'sum_abs', or 'sum')
    
    Returns:
        Array of aggregated gradients for each activation dimension, or dict with both if mode is 'both'
    """
    print(f"\nProcessing layer {layer_idx}")
    
    # Storage for gradients across all samples
    all_gradients_signed = []
    all_gradients_abs = []
    
    for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Layer {layer_idx}")):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        # Skip if batch is too small
        if input_ids.size(0) == 0:
            continue
        
        # Storage for this batch
        batch_activations = []
        
        # Hook to capture MLP output activations
        def capture_activation_hook(module, input, output):
            # output shape: [batch, seq_len, hidden_dim]
            batch_activations.append(output)
            return output
        
        # Register hook
        target_module = get_layer_module(model, layer_idx, model_config)
        hook_handle = target_module.register_forward_hook(capture_activation_hook)
        
        try:
            # Forward pass to capture activations (with gradients)
            activations_with_grad = None
            
            # We need to do a custom forward pass where we can replace activations
            def replace_activation_hook(module, input, output):
                nonlocal activations_with_grad
                # Enable gradients on activations
                activations_with_grad = output.detach().clone()
                activations_with_grad.requires_grad_(True)
                return activations_with_grad
            
            # Remove the capture hook and add replace hook
            hook_handle.remove()
            hook_handle = target_module.register_forward_hook(replace_activation_hook)
            
            # Forward pass with replaced activations
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss
            
            # Remove hook
            hook_handle.remove()
            
            if activations_with_grad is None:
                continue
            
            # Compute gradients
            if loss.requires_grad and activations_with_grad.requires_grad:
                grads = torch.autograd.grad(
                    loss,
                    activations_with_grad,
                    retain_graph=False,
                    create_graph=False
                )
                
                # grads[0] shape: [batch, seq_len, hidden_dim]
                # Get gradients for valid (non-padded) positions
                batch_size, seq_len, hidden_dim = grads[0].shape
                mask_expanded = attention_mask.unsqueeze(-1).expand_as(grads[0])
                
                # Apply mask and compute mean across valid positions
                valid_grads = grads[0] * mask_expanded
                num_valid = attention_mask.sum()
                
                if num_valid > 0:
                    # Compute mean across batch and sequence dimensions
                    if aggregation_mode in ['mean', 'both']:
                        grad_mean_signed = valid_grads.sum(dim=(0, 1)) / num_valid  # [hidden_dim]
                        all_gradients_signed.append(grad_mean_signed.detach().cpu().numpy())
                    
                    if aggregation_mode in ['mean_abs', 'both']:
                        grad_mean_abs = torch.abs(valid_grads).sum(dim=(0, 1)) / num_valid  # [hidden_dim]
                        all_gradients_abs.append(grad_mean_abs.detach().cpu().numpy())
        
        except Exception as e:
            print(f"Error processing batch {batch_idx} for layer {layer_idx}: {e}")
            if 'hook_handle' in locals():
                hook_handle.remove()
            continue
    
    # Compute mean across all batches
    if aggregation_mode == 'mean':
        if len(all_gradients_signed) == 0:
            print(f"Warning: No gradients computed for layer {layer_idx}")
            return None
        mean_gradients = np.mean(all_gradients_signed, axis=0)
        dimension_size = len(mean_gradients)
        
        print(f"Layer {layer_idx} (before normalization): Mean gradient = {mean_gradients.mean():.6f}, "
              f"Max gradient = {np.abs(mean_gradients).max():.6f}, "
              f"Std gradient = {mean_gradients.std():.6f}")
        
        if normalize == 'none':
            return mean_gradients
        elif normalize == 'both':
            # Apply both normalizations
            norm_sum = normalize_gradients(mean_gradients, dimension_size, mode='sum')
            norm_sum_abs = normalize_gradients(mean_gradients, dimension_size, mode='sum_abs')
            print(f"Layer {layer_idx} (after normalization):")
            print(f"  norm_sum: Sum = {np.sum(norm_sum):.1f} (target: {dimension_size})")
            print(f"  norm_sum_abs: Sum of abs = {np.sum(np.abs(norm_sum_abs)):.1f} (target: {dimension_size})")
            return {'norm_sum': norm_sum, 'norm_sum_abs': norm_sum_abs}
        else:
            mean_gradients = normalize_gradients(mean_gradients, dimension_size, mode=normalize)
            if normalize == 'sum_abs':
                print(f"Layer {layer_idx} (after {normalize}): Sum of abs values = {np.sum(np.abs(mean_gradients)):.1f} (target: {dimension_size})")
            elif normalize == 'sum':
                print(f"Layer {layer_idx} (after {normalize}): Sum = {np.sum(mean_gradients):.1f} (target: {dimension_size})")
            return mean_gradients
    
    elif aggregation_mode == 'mean_abs':
        if len(all_gradients_abs) == 0:
            print(f"Warning: No gradients computed for layer {layer_idx}")
            return None
        mean_gradients = np.mean(all_gradients_abs, axis=0)
        dimension_size = len(mean_gradients)
        
        print(f"Layer {layer_idx} (before normalization): Mean abs gradient = {mean_gradients.mean():.6f}, "
              f"Max gradient = {mean_gradients.max():.6f}, "
              f"Std gradient = {mean_gradients.std():.6f}")
        
        if normalize == 'none':
            return mean_gradients
        elif normalize == 'both':
            # Apply both normalizations
            norm_sum = normalize_gradients(mean_gradients, dimension_size, mode='sum')
            norm_sum_abs = normalize_gradients(mean_gradients, dimension_size, mode='sum_abs')
            print(f"Layer {layer_idx} (after normalization):")
            print(f"  norm_sum: Sum = {np.sum(norm_sum):.1f} (target: {dimension_size})")
            print(f"  norm_sum_abs: Sum of abs = {np.sum(np.abs(norm_sum_abs)):.1f} (target: {dimension_size})")
            return {'norm_sum': norm_sum, 'norm_sum_abs': norm_sum_abs}
        else:
            mean_gradients = normalize_gradients(mean_gradients, dimension_size, mode=normalize)
            if normalize == 'sum_abs':
                print(f"Layer {layer_idx} (after {normalize}): Sum of abs values = {np.sum(np.abs(mean_gradients)):.1f} (target: {dimension_size})")
            elif normalize == 'sum':
                print(f"Layer {layer_idx} (after {normalize}): Sum = {np.sum(mean_gradients):.1f} (target: {dimension_size})")
            return mean_gradients
    
    else:  # both aggregation modes
        if len(all_gradients_signed) == 0 or len(all_gradients_abs) == 0:
            print(f"Warning: No gradients computed for layer {layer_idx}")
            return None
        mean_gradients_signed = np.mean(all_gradients_signed, axis=0)
        mean_gradients_abs = np.mean(all_gradients_abs, axis=0)
        dimension_size = len(mean_gradients_signed)
        
        print(f"Layer {layer_idx} (before normalization):")
        print(f"  Mean gradient (signed) = {mean_gradients_signed.mean():.6f}, "
              f"Max = {np.abs(mean_gradients_signed).max():.6f}, Std = {mean_gradients_signed.std():.6f}")
        print(f"  Mean gradient (abs) = {mean_gradients_abs.mean():.6f}, "
              f"Max = {mean_gradients_abs.max():.6f}, Std = {mean_gradients_abs.std():.6f}")
        
        if normalize == 'none':
            return {'mean': mean_gradients_signed, 'mean_abs': mean_gradients_abs}
        elif normalize == 'both':
            # Apply both normalizations to both aggregation modes
            result = {
                'mean': {
                    'norm_sum': normalize_gradients(mean_gradients_signed, dimension_size, mode='sum'),
                    'norm_sum_abs': normalize_gradients(mean_gradients_signed, dimension_size, mode='sum_abs')
                },
                'mean_abs': {
                    'norm_sum': normalize_gradients(mean_gradients_abs, dimension_size, mode='sum'),
                    'norm_sum_abs': normalize_gradients(mean_gradients_abs, dimension_size, mode='sum_abs')
                }
            }
            print(f"Layer {layer_idx} (after both normalizations):")
            print(f"  Mean (norm_sum): Sum = {np.sum(result['mean']['norm_sum']):.1f}")
            print(f"  Mean (norm_sum_abs): Sum of abs = {np.sum(np.abs(result['mean']['norm_sum_abs'])):.1f}")
            print(f"  Mean_abs (norm_sum): Sum = {np.sum(result['mean_abs']['norm_sum']):.1f}")
            print(f"  Mean_abs (norm_sum_abs): Sum of abs = {np.sum(np.abs(result['mean_abs']['norm_sum_abs'])):.1f}")
            return result
        else:
            mean_gradients_signed = normalize_gradients(mean_gradients_signed, dimension_size, mode=normalize)
            mean_gradients_abs = normalize_gradients(mean_gradients_abs, dimension_size, mode=normalize)
            print(f"Layer {layer_idx} (after {normalize}):")
            if normalize == 'sum_abs':
                print(f"  Mean gradient (signed): Sum of abs = {np.sum(np.abs(mean_gradients_signed)):.1f} (target: {dimension_size})")
                print(f"  Mean gradient (abs): Sum of abs = {np.sum(np.abs(mean_gradients_abs)):.1f} (target: {dimension_size})")
            elif normalize == 'sum':
                print(f"  Mean gradient (signed): Sum = {np.sum(mean_gradients_signed):.1f} (target: {dimension_size})")
                print(f"  Mean gradient (abs): Sum = {np.sum(mean_gradients_abs):.1f} (target: {dimension_size})")
            return {'mean': mean_gradients_signed, 'mean_abs': mean_gradients_abs}


def create_histogram(gradients: np.ndarray, layer_idx: int, output_dir: str, mode_suffix: str = ""):
    """Create and save histogram plot for gradient values.
    
    Args:
        gradients: Array of aggregated gradients
        layer_idx: Layer index
        output_dir: Directory to save plot
        mode_suffix: Suffix for filename (e.g., "_mean" or "_mean_abs")
    """
    plt.figure(figsize=(12, 6))
    
    # Create histogram
    plt.hist(gradients, bins=100, alpha=0.7, color='blue', edgecolor='black')
    
    # Add statistics
    mean_grad = gradients.mean()
    max_grad = np.abs(gradients).max()
    std_grad = gradients.std()
    
    plt.axvline(mean_grad, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_grad:.6f}')
    
    mode_name = "Signed Mean" if "mean" in mode_suffix and "abs" not in mode_suffix else "Mean Absolute"
    plt.xlabel(f'Gradient Magnitude ({mode_name})', fontsize=12)
    plt.ylabel('Number of Activation Dimensions', fontsize=12)
    plt.title(f'Layer {layer_idx}: Activation Gradient Distribution ({mode_name})\n'
              f'Max: {max_grad:.6f}, Mean: {mean_grad:.6f}, Std: {std_grad:.6f}',
              fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save plot
    output_path = os.path.join(output_dir, f'layer_{layer_idx}{mode_suffix}_gradient_hist.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved histogram to {output_path}")


def create_feature_scatter_plot(gradients: np.ndarray, layer_idx: int, output_dir: str, mode_suffix: str = ""):
    """Create and save scatter plot showing gradient per activation dimension.
    
    Args:
        gradients: Array of aggregated gradients (one per activation dimension)
        layer_idx: Layer index
        output_dir: Directory to save plot
        mode_suffix: Suffix for filename (e.g., "_mean" or "_mean_abs")
    """
    plt.figure(figsize=(14, 6))
    
    num_dims = len(gradients)
    dim_indices = np.arange(num_dims)
    
    # Plot gradient per dimension (scatter only, no line)
    plt.scatter(dim_indices, gradients, s=3, alpha=0.6, color='blue')
    
    # Add statistics
    mean_grad = gradients.mean()
    max_grad = np.abs(gradients).max()
    std_grad = gradients.std()
    
    # Highlight top dimensions
    top_k = min(20, num_dims)
    top_indices = np.argsort(np.abs(gradients))[-top_k:]
    plt.scatter(top_indices, gradients[top_indices], 
                s=80, color='red', alpha=0.8, 
                label=f'Top {top_k} dimensions', zorder=5, marker='*')
    
    plt.axhline(mean_grad, color='green', linestyle='--', linewidth=1.5, 
                label=f'Mean: {mean_grad:.6f}', alpha=0.7)
    
    mode_name = "Signed Mean" if "mean" in mode_suffix and "abs" not in mode_suffix else "Mean Absolute"
    plt.xlabel('Activation Dimension Index', fontsize=12)
    plt.ylabel(f'Gradient Magnitude ({mode_name})', fontsize=12)
    plt.title(f'Layer {layer_idx}: Activation Gradient Scatter Plot ({mode_name})\n'
              f'Total Dimensions: {num_dims}, Max: {max_grad:.6f}, Mean: {mean_grad:.6f}, Std: {std_grad:.6f}',
              fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save plot
    output_path = os.path.join(output_dir, f'layer_{layer_idx}{mode_suffix}_activation_scatter.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved activation scatter plot to {output_path}")


def create_combined_single_layer_plot(gradients_dict: Dict[str, np.ndarray], layer_idx: int, 
                                     output_dir: str, plot_type: str = 'histogram'):
    """Create combined plot showing both mean and mean_abs side by side for a single layer.
    
    Args:
        gradients_dict: Dict with 'mean' and 'mean_abs' gradient arrays
        layer_idx: Layer index
        output_dir: Directory to save plot
        plot_type: 'histogram' or 'scatter'
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))
    
    for idx, (mode, gradients) in enumerate([('mean', gradients_dict['mean']), 
                                              ('mean_abs', gradients_dict['mean_abs'])]):
        ax = ax1 if idx == 0 else ax2
        mode_name = "Signed Mean" if mode == 'mean' else "Mean Absolute"
        
        if plot_type == 'histogram':
            # Create histogram
            ax.hist(gradients, bins=100, alpha=0.7, color='blue', edgecolor='black')
            
            # Add statistics
            mean_grad = gradients.mean()
            ax.axvline(mean_grad, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_grad:.6f}')
            
            ax.set_xlabel(f'Gradient Magnitude ({mode_name})', fontsize=12)
            ax.set_ylabel('Number of Activation Dimensions', fontsize=12)
            ax.set_title(f'{mode_name}\nMean: {mean_grad:.6f}, Max: {np.abs(gradients).max():.6f}', fontsize=12)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        else:  # scatter
            num_dims = len(gradients)
            dim_indices = np.arange(num_dims)
            
            # Scatter plot
            ax.scatter(dim_indices, gradients, s=3, alpha=0.6, color='blue')
            
            # Add statistics
            mean_grad = gradients.mean()
            
            # Highlight top dimensions
            top_k = min(20, num_dims)
            top_indices = np.argsort(np.abs(gradients))[-top_k:]
            ax.scatter(top_indices, gradients[top_indices], 
                      s=80, color='red', alpha=0.8, 
                      label=f'Top {top_k}', zorder=5, marker='*')
            
            ax.axhline(mean_grad, color='green', linestyle='--', linewidth=1.5, 
                      label=f'Mean: {mean_grad:.6f}', alpha=0.7)
            
            ax.set_xlabel('Activation Dimension Index', fontsize=12)
            ax.set_ylabel(f'Gradient Magnitude ({mode_name})', fontsize=12)
            ax.set_title(f'{mode_name}\nMean: {mean_grad:.6f}, Max: {np.abs(gradients).max():.6f}', fontsize=12)
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    fig.suptitle(f'Layer {layer_idx}: Activation Gradient Comparison', fontsize=16)
    plt.tight_layout()
    
    # Save plot
    suffix = 'hist' if plot_type == 'histogram' else 'scatter'
    output_path = os.path.join(output_dir, f'layer_{layer_idx}_gradient_{suffix}_combined.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved combined {plot_type} plot to {output_path}")


def create_combined_histograms(all_gradients: Dict[int, np.ndarray], output_dir: str, mode_suffix: str = ""):
    """Create combined plot showing histograms for all layers.
    
    Args:
        all_gradients: Dictionary mapping layer_idx to gradient arrays
        output_dir: Directory to save plot
        mode_suffix: Suffix for filename (e.g., "_mean" or "_mean_abs")
    """
    num_layers = len(all_gradients)
    layers = sorted(all_gradients.keys())
    
    # Create grid layout (4 rows x 6 columns for 24 layers)
    n_cols = 6
    n_rows = (num_layers + n_cols - 1) // n_cols  # Ceiling division
    
    mode_name = "Signed Mean" if mode_suffix == "_mean" else "Mean Absolute"
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(24, n_rows * 3))
    fig.suptitle(f'Activation Gradient Distribution - All Layers ({mode_name})', fontsize=16, y=0.995)
    
    # Flatten axes for easier iteration
    if num_layers == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx, layer_idx in enumerate(layers):
        ax = axes[idx]
        gradients = all_gradients[layer_idx]
        
        # Create histogram
        ax.hist(gradients, bins=50, alpha=0.7, color='blue', edgecolor='black')
        
        # Add statistics
        mean_grad = gradients.mean()
        ax.axvline(mean_grad, color='red', linestyle='--', linewidth=1, 
                   label=f'Mean: {mean_grad:.2e}')
        
        ax.set_xlabel('Gradient Magnitude', fontsize=8)
        ax.set_ylabel('Count', fontsize=8)
        ax.set_title(f'Layer {layer_idx}', fontsize=10, fontweight='bold')
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)
    
    # Hide unused subplots
    for idx in range(num_layers, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    # Save plot
    output_path = os.path.join(output_dir, f'all_layers{mode_suffix}_gradient_histograms.png')
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"\nSaved combined histogram plot to {output_path}")


def create_combined_scatter_plots(all_gradients: Dict[int, np.ndarray], output_dir: str, mode_suffix: str = ""):
    """Create combined plot showing scatter plots for all layers.
    
    Args:
        all_gradients: Dictionary mapping layer_idx to gradient arrays
        output_dir: Directory to save plot
        mode_suffix: Suffix for filename (e.g., "_mean" or "_mean_abs")
    """
    num_layers = len(all_gradients)
    layers = sorted(all_gradients.keys())
    
    # Create grid layout (4 rows x 6 columns for 24 layers)
    n_cols = 6
    n_rows = (num_layers + n_cols - 1) // n_cols  # Ceiling division
    
    mode_name = "Signed Mean" if mode_suffix == "_mean" else "Mean Absolute"
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(24, n_rows * 3))
    fig.suptitle(f'Activation Gradient Scatter - All Layers ({mode_name})', fontsize=16, y=0.995)
    
    # Flatten axes for easier iteration
    if num_layers == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx, layer_idx in enumerate(layers):
        ax = axes[idx]
        gradients = all_gradients[layer_idx]
        
        num_dims = len(gradients)
        dim_indices = np.arange(num_dims)
        
        # Scatter plot with smaller points
        ax.scatter(dim_indices, gradients, s=0.5, alpha=0.4, color='blue')
        
        # Add mean line
        mean_grad = gradients.mean()
        ax.axhline(mean_grad, color='green', linestyle='--', linewidth=1, alpha=0.7)
        
        # Highlight top dimensions
        top_k = min(20, num_dims)
        top_indices = np.argsort(np.abs(gradients))[-top_k:]
        ax.scatter(top_indices, gradients[top_indices], 
                  s=15, color='red', alpha=0.8, marker='*', zorder=5)
        
        ax.set_xlabel('Dimension Index', fontsize=8)
        ax.set_ylabel('Gradient Magnitude', fontsize=8)
        ax.set_title(f'Layer {layer_idx}', fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)
        
        # Use scientific notation for y-axis
        ax.ticklabel_format(axis='y', style='scientific', scilimits=(0,0))
    
    # Hide unused subplots
    for idx in range(num_layers, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    # Save plot
    output_path = os.path.join(output_dir, f'all_layers{mode_suffix}_activation_scatter.png')
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"Saved combined scatter plot to {output_path}")


def save_consolidated_results(
    all_results: Union[Dict[int, np.ndarray], Dict[str, Dict[int, np.ndarray]]],
    args,
    layers: List[int],
    output_path: str,
    aggregation_mode: str
):
    """Save all results in a consolidated .npz file.
    
    Args:
        all_results: Dictionary with gradient results per layer
        args: Command line arguments
        layers: List of layer indices
        output_path: Path to save .npz file
        aggregation_mode: Aggregation mode used
    """
    # Prepare data dictionary for npz
    save_dict = {}
    
    if aggregation_mode == 'both' and args.normalize_gradients == 'both':
        # Both aggregation AND normalization
        for agg_mode in ['mean', 'mean_abs']:
            for norm_mode in ['norm_sum', 'norm_sum_abs']:
                for layer_idx in sorted(all_results[agg_mode][norm_mode].keys()):
                    key = f'layer_{layer_idx}_{agg_mode}_{norm_mode}'
                    save_dict[key] = all_results[agg_mode][norm_mode][layer_idx]
    elif aggregation_mode == 'both':
        # Both aggregation modes, single normalization
        for mode in ['mean', 'mean_abs']:
            for layer_idx in sorted(all_results[mode].keys()):
                key = f'layer_{layer_idx}_{mode}'
                save_dict[key] = all_results[mode][layer_idx]
    elif args.normalize_gradients == 'both':
        # Single aggregation, both normalizations
        for norm_mode in ['norm_sum', 'norm_sum_abs']:
            for layer_idx in sorted(all_results[norm_mode].keys()):
                key = f'layer_{layer_idx}_{aggregation_mode}_{norm_mode}'
                save_dict[key] = all_results[norm_mode][layer_idx]
    else:
        # Single aggregation, single normalization
        for layer_idx in sorted(all_results.keys()):
            key = f'layer_{layer_idx}_{aggregation_mode}'
            save_dict[key] = all_results[layer_idx]
    
    # Add metadata
    metadata = {
        'model_name': args.model_name,
        'num_samples': args.num_samples,
        'batch_size': args.batch_size,
        'max_length': args.max_length,
        'layers': layers,
        'aggregation_mode': aggregation_mode,
        'normalization_mode': args.normalize_gradients,
        'device': args.device,
        'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
    }
    
    # Add dimension info per layer
    if aggregation_mode == 'both' and args.normalize_gradients == 'both':
        dimension_info = {layer_idx: len(all_results['mean']['norm_sum'][layer_idx]) 
                         for layer_idx in all_results['mean']['norm_sum'].keys()}
    elif aggregation_mode == 'both':
        dimension_info = {layer_idx: len(all_results['mean'][layer_idx]) 
                         for layer_idx in all_results['mean'].keys()}
    elif args.normalize_gradients == 'both':
        dimension_info = {layer_idx: len(all_results['norm_sum'][layer_idx]) 
                         for layer_idx in all_results['norm_sum'].keys()}
    else:
        dimension_info = {layer_idx: len(all_results[layer_idx]) 
                         for layer_idx in all_results.keys()}
    metadata['dimension_sizes'] = dimension_info
    
    # Save as npz with metadata
    save_dict['metadata'] = np.array([metadata], dtype=object)
    
    np.savez_compressed(output_path, **save_dict)
    print(f"\nSaved consolidated results to: {output_path}")
    print(f"  Layers included: {sorted([k for k in save_dict.keys() if k != 'metadata'])}")
    print(f"  File size: {os.path.getsize(output_path) / (1024**2):.2f} MB")


def create_enhanced_metadata(
    all_results: Union[Dict[int, np.ndarray], Dict[str, Dict[int, np.ndarray]]],
    args,
    layers: List[int],
    timing_stats: Dict,
    layer_timings: Dict,
    aggregation_mode: str
) -> Dict:
    """Create enhanced metadata dictionary with comprehensive information.
    
    Args:
        all_results: Dictionary with gradient results per layer
        args: Command line arguments
        layers: List of layer indices
        timing_stats: Overall timing statistics
        layer_timings: Per-layer timing information
        aggregation_mode: Aggregation mode used
    
    Returns:
        Enhanced metadata dictionary
    """
    metadata = {
        'experiment_info': {
            'model_name': args.model_name,
            'num_samples': args.num_samples,
            'batch_size': args.batch_size,
            'max_length': args.max_length,
            'device': args.device,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
        'analysis_config': {
            'layers': layers,
            'aggregation_mode': aggregation_mode,
            'normalization_mode': args.normalize_gradients,
        },
        'timing_stats': timing_stats,
        'per_layer_timing': layer_timings,
        'layer_statistics': {},
        'dimension_info': {}
    }
    
    # Add per-layer statistics and dimension info
    if aggregation_mode == 'both' and args.normalize_gradients == 'both':
        for layer_idx in all_results['mean']['norm_sum'].keys():
            dimension_size = len(all_results['mean']['norm_sum'][layer_idx])
            metadata['dimension_info'][f'layer_{layer_idx}'] = dimension_size
            metadata['layer_statistics'][f'layer_{layer_idx}'] = {}
            for agg_mode in ['mean', 'mean_abs']:
                metadata['layer_statistics'][f'layer_{layer_idx}'][agg_mode] = {}
                for norm_mode in ['norm_sum', 'norm_sum_abs']:
                    data = all_results[agg_mode][norm_mode][layer_idx]
                    metadata['layer_statistics'][f'layer_{layer_idx}'][agg_mode][norm_mode] = {
                        'mean': float(data.mean()),
                        'std': float(data.std()),
                        'max': float(np.abs(data).max()),
                        'sum_abs': float(np.sum(np.abs(data)))
                    }
    elif aggregation_mode == 'both':
        for layer_idx in all_results['mean'].keys():
            dimension_size = len(all_results['mean'][layer_idx])
            metadata['dimension_info'][f'layer_{layer_idx}'] = dimension_size
            metadata['layer_statistics'][f'layer_{layer_idx}'] = {
                'mean': {
                    'mean': float(all_results['mean'][layer_idx].mean()),
                    'std': float(all_results['mean'][layer_idx].std()),
                    'max': float(np.abs(all_results['mean'][layer_idx]).max()),
                    'sum_abs': float(np.sum(np.abs(all_results['mean'][layer_idx])))
                },
                'mean_abs': {
                    'mean': float(all_results['mean_abs'][layer_idx].mean()),
                    'std': float(all_results['mean_abs'][layer_idx].std()),
                    'max': float(all_results['mean_abs'][layer_idx].max()),
                    'sum_abs': float(np.sum(np.abs(all_results['mean_abs'][layer_idx])))
                }
            }
    elif args.normalize_gradients == 'both':
        for layer_idx in all_results['norm_sum'].keys():
            dimension_size = len(all_results['norm_sum'][layer_idx])
            metadata['dimension_info'][f'layer_{layer_idx}'] = dimension_size
            metadata['layer_statistics'][f'layer_{layer_idx}'] = {}
            for norm_mode in ['norm_sum', 'norm_sum_abs']:
                data = all_results[norm_mode][layer_idx]
                metadata['layer_statistics'][f'layer_{layer_idx}'][norm_mode] = {
                    'mean': float(data.mean()),
                    'std': float(data.std()),
                    'max': float(np.abs(data).max()),
                    'sum_abs': float(np.sum(np.abs(data)))
                }
    else:
        for layer_idx in all_results.keys():
            dimension_size = len(all_results[layer_idx])
            metadata['dimension_info'][f'layer_{layer_idx}'] = dimension_size
            metadata['layer_statistics'][f'layer_{layer_idx}'] = {
                'mean': float(all_results[layer_idx].mean()),
                'std': float(all_results[layer_idx].std()),
                'max': float(np.abs(all_results[layer_idx]).max()),
                'sum_abs': float(np.sum(np.abs(all_results[layer_idx])))
            }
    
    # Add loading instructions
    metadata['loading_instructions'] = {
        'npz_file': 'Use np.load(filename, allow_pickle=True) to load consolidated results',
        'individual_npy': 'Use np.load(layer_X_mode_gradients.npy) for individual layers',
        'example_code': [
            "# Load consolidated .npz file:",
            "data = np.load('results.npz', allow_pickle=True)",
            "metadata = data['metadata'].item()",
            "layer_0_gradients = data['layer_0_mean_abs']",
            "",
            "# Or use the helper function:",
            "results = load_results('results.npz')"
        ]
    }
    
    return metadata


def load_results(path: str) -> Dict:
    """Helper function to load saved gradient analysis results.
    
    Args:
        path: Path to .npz file or directory containing results
    
    Returns:
        Dictionary with 'data' (gradients per layer) and 'metadata' keys
    """
    if path.endswith('.npz'):
        # Load consolidated npz file
        data = np.load(path, allow_pickle=True)
        
        # Extract metadata
        metadata = data['metadata'].item() if 'metadata' in data else {}
        
        # Extract gradient data
        gradients = {}
        for key in data.keys():
            if key != 'metadata':
                gradients[key] = data[key]
        
        return {
            'data': gradients,
            'metadata': metadata
        }
    else:
        # Load from directory with individual .npy files
        import json
        
        # Load metadata
        metadata_path = os.path.join(path, 'metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = {}
        
        # Find all .npy files
        gradients = {}
        for file in os.listdir(path):
            if file.endswith('_gradients.npy'):
                key = file.replace('_gradients.npy', '')
                gradients[key] = np.load(os.path.join(path, file))
        
        return {
            'data': gradients,
            'metadata': metadata
        }


def main():
    """Main function to run the experiment."""
    start_time_total = time.time()
    timing_stats = {}
    
    args = parse_args()
    
    # Get model configuration
    model_config = get_model_config(args.model_name)
    print(f"Model type detected: {model_config['model_type']}")
    
    # Parse layer range
    layers = parse_layer_range(args.layers)
    print(f"Analyzing layers: {layers}")
    print(f"Aggregation mode: {args.aggregation_mode}")
    
    # Determine output directory
    output_dir: Optional[str] = None
    if args.save_outputs:
        # Generate directory name from parameters
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_safe = args.model_name.replace("/", "_")
        layers_safe = args.layers.replace("-", "to").replace(",", "_")
        output_dir = f"layer_activations_model_{model_safe}_num_samples_{args.num_samples}_batch_size_{args.batch_size}_layers_{layers_safe}_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        print(f"\nOutputs will be saved to: {output_dir}")
    else:
        print("\nOutputs will NOT be saved (use --save_outputs to enable)")
        output_dir = None
    
    # Load model and tokenizer
    print("\n" + "="*60)
    print("TIMING: Loading Model and Tokenizer")
    print("="*60)
    start_time = time.time()
    model, tokenizer = load_model_and_tokenizer(args.model_name, args.device)
    timing_stats['model_loading'] = time.time() - start_time
    print(f"Model loading took: {timing_stats['model_loading']:.2f} seconds")
    
    # Prepare dataset
    print("\n" + "="*60)
    print("TIMING: Preparing Dataset")
    print("="*60)
    start_time = time.time()
    dataloader = prepare_dataset(tokenizer, args.num_samples, args.max_length, args.batch_size)
    timing_stats['dataset_preparation'] = time.time() - start_time
    print(f"Dataset preparation took: {timing_stats['dataset_preparation']:.2f} seconds")
    
    # Process each layer - initialize results structure based on modes
    if args.aggregation_mode == 'both' and args.normalize_gradients == 'both':
        all_results = {
            'mean': {'norm_sum': {}, 'norm_sum_abs': {}},
            'mean_abs': {'norm_sum': {}, 'norm_sum_abs': {}}
        }
    elif args.aggregation_mode == 'both':
        all_results = {'mean': {}, 'mean_abs': {}}
    elif args.normalize_gradients == 'both':
        all_results = {'norm_sum': {}, 'norm_sum_abs': {}}
    else:
        all_results = {}
    layer_timings = {}
    
    print("\n" + "="*60)
    print("TIMING: Processing Layers")
    print("="*60)
    
    for layer_idx in layers:
        layer_start_time = time.time()
        layer_timing = {}
        
        # Compute gradients
        grad_start = time.time()
        gradients = compute_gradients_for_layer(
            model=model,
            tokenizer=tokenizer,
            dataloader=dataloader,
            layer_idx=layer_idx,
            device=args.device,
            aggregation_mode=args.aggregation_mode,
            model_config=model_config,
            normalize=args.normalize_gradients
        )
        layer_timing['gradient_computation'] = time.time() - grad_start
        
        if gradients is not None:
            # Handle different combinations of aggregation and normalization modes
            if args.aggregation_mode == 'both' and args.normalize_gradients == 'both':
                # Both aggregation AND normalization = 4 variants
                for agg_mode in ['mean', 'mean_abs']:
                    for norm_mode in ['norm_sum', 'norm_sum_abs']:
                        all_results[agg_mode][norm_mode][layer_idx] = gradients[agg_mode][norm_mode]
                        
                        if output_dir is not None:
                            # Save each variant
                            filename = f'layer_{layer_idx}_{agg_mode}_{norm_mode}_gradients.npy'
                            np.save(os.path.join(output_dir, filename), gradients[agg_mode][norm_mode])
                            
                            # Create plots for each variant
                            plot_start = time.time()
                            mode_suffix = f'_{agg_mode}_{norm_mode}'
                            create_histogram(gradients[agg_mode][norm_mode], layer_idx, output_dir, mode_suffix)
                            create_feature_scatter_plot(gradients[agg_mode][norm_mode], layer_idx, output_dir, mode_suffix)
                            layer_timing[f'{agg_mode}_{norm_mode}_plots'] = time.time() - plot_start
            
            elif args.aggregation_mode == 'both':
                # Both aggregation modes, single normalization
                for agg_mode in ['mean', 'mean_abs']:
                    all_results[agg_mode][layer_idx] = gradients[agg_mode]
                
                if output_dir is not None:
                    # Create separate plots for each mode
                    for mode, mode_suffix in [('mean', '_mean'), ('mean_abs', '_mean_abs')]:
                        plot_start = time.time()
                        create_histogram(gradients[mode], layer_idx, output_dir, mode_suffix)
                        create_feature_scatter_plot(gradients[mode], layer_idx, output_dir, mode_suffix)
                        layer_timing[f'{mode}_plots'] = time.time() - plot_start
                        
                        # Save raw gradients
                        np.save(
                            os.path.join(output_dir, f'layer_{layer_idx}{mode_suffix}_gradients.npy'),
                            gradients[mode]
                        )
                    
                    # Create combined subplot figures
                    combined_start = time.time()
                    create_combined_single_layer_plot(gradients, layer_idx, output_dir, 'histogram')
                    create_combined_single_layer_plot(gradients, layer_idx, output_dir, 'scatter')
                    layer_timing['combined_plots'] = time.time() - combined_start
            
            elif args.normalize_gradients == 'both':
                # Single aggregation, both normalizations
                for norm_mode in ['norm_sum', 'norm_sum_abs']:
                    all_results[norm_mode][layer_idx] = gradients[norm_mode]
                
                if output_dir is not None:
                    for norm_mode in ['norm_sum', 'norm_sum_abs']:
                        plot_start = time.time()
                        mode_suffix = f'_{args.aggregation_mode}_{norm_mode}'
                        create_histogram(gradients[norm_mode], layer_idx, output_dir, mode_suffix)
                        create_feature_scatter_plot(gradients[norm_mode], layer_idx, output_dir, mode_suffix)
                        layer_timing[f'{norm_mode}_plots'] = time.time() - plot_start
                        
                        # Save raw gradients
                        filename = f'layer_{layer_idx}_{args.aggregation_mode}_{norm_mode}_gradients.npy'
                        np.save(os.path.join(output_dir, filename), gradients[norm_mode])
            
            else:
                # Single aggregation, single normalization
                all_results[layer_idx] = gradients
                mode_suffix = f"_{args.aggregation_mode}"
                
                if output_dir is not None:
                    # Create plots
                    plot_start = time.time()
                    create_histogram(gradients, layer_idx, output_dir, mode_suffix)
                    create_feature_scatter_plot(gradients, layer_idx, output_dir, mode_suffix)
                    layer_timing['plots'] = time.time() - plot_start
                    
                    # Save raw gradients
                    save_start = time.time()
                    np.save(
                        os.path.join(output_dir, f'layer_{layer_idx}{mode_suffix}_gradients.npy'),
                        gradients
                    )
                    layer_timing['save_data'] = time.time() - save_start
            
            layer_timing['total'] = time.time() - layer_start_time
            layer_timings[layer_idx] = layer_timing
            
            print(f"\nLayer {layer_idx} timing breakdown:")
            print(f"  Gradient computation: {layer_timing['gradient_computation']:.2f}s")
            if args.aggregation_mode == 'both':
                print(f"  Mean plots:          {layer_timing.get('mean_plots', 0):.2f}s")
                print(f"  Mean_abs plots:      {layer_timing.get('mean_abs_plots', 0):.2f}s")
                print(f"  Combined plots:      {layer_timing.get('combined_plots', 0):.2f}s")
            else:
                print(f"  Plots:               {layer_timing.get('plots', 0):.2f}s")
                print(f"  Save data:           {layer_timing.get('save_data', 0):.2f}s")
            print(f"  Total for layer:     {layer_timing['total']:.2f}s")
    
    timing_stats['total_runtime'] = time.time() - start_time_total
    
    # Print final timing summary
    summary_lines = [
        "\n" + "="*60,
        "FINAL TIMING SUMMARY",
        "="*60,
        f"Model loading:         {timing_stats['model_loading']:.2f} seconds",
        f"Dataset preparation:   {timing_stats['dataset_preparation']:.2f} seconds",
        f"Total runtime:         {timing_stats['total_runtime']:.2f} seconds",
        "="*60,
    ]
    
    # Print to screen
    for line in summary_lines:
        print(line)
    
    # Save timing stats and create combined plots if output directory exists
    if output_dir is not None:
        import json
        
        # Create enhanced metadata
        enhanced_metadata = create_enhanced_metadata(
            all_results=all_results,
            args=args,
            layers=layers,
            timing_stats=timing_stats,
            layer_timings=layer_timings,
            aggregation_mode=args.aggregation_mode
        )
        
        # Save enhanced metadata to JSON
        metadata_json_file = os.path.join(output_dir, 'metadata.json')
        with open(metadata_json_file, 'w') as f:
            json.dump(enhanced_metadata, f, indent=2)
        print(f"\nEnhanced metadata saved to {metadata_json_file}")
        
        # Save timing stats to JSON (backward compatibility)
        timing_json_file = os.path.join(output_dir, 'timing_stats.json')
        with open(timing_json_file, 'w') as f:
            json.dump({
                'overall': timing_stats,
                'per_layer': layer_timings
            }, f, indent=2)
        print(f"Timing statistics (JSON) saved to {timing_json_file}")
        
        # Save timing summary to text file
        timing_txt_file = os.path.join(output_dir, 'timing_summary.txt')
        with open(timing_txt_file, 'w') as f:
            for line in summary_lines:
                f.write(line + '\n')
        print(f"Timing summary (TXT) saved to {timing_txt_file}")
        
        # Create combined plots showing all layers
        if args.aggregation_mode == 'both' and args.normalize_gradients == 'both':
            # 4 variants: mean x (norm_sum, norm_sum_abs) + mean_abs x (norm_sum, norm_sum_abs)
            num_layers_check = len(all_results['mean']['norm_sum'])
            if num_layers_check > 1:
                print("\n" + "="*60)
                print("Creating combined plots for all layers...")
                print("="*60)
                
                combined_start = time.time()
                for agg_mode in ['mean', 'mean_abs']:
                    for norm_mode in ['norm_sum', 'norm_sum_abs']:
                        mode_suffix = f'_{agg_mode}_{norm_mode}'
                        create_combined_histograms(all_results[agg_mode][norm_mode], output_dir, mode_suffix)
                        create_combined_scatter_plots(all_results[agg_mode][norm_mode], output_dir, mode_suffix)
                combined_time = time.time() - combined_start
                
                print(f"Combined plots created in {combined_time:.2f} seconds")
        elif args.aggregation_mode == 'both':
            if len(all_results['mean']) > 1:
                print("\n" + "="*60)
                print("Creating combined plots for all layers...")
                print("="*60)
                
                combined_start = time.time()
                for mode, mode_suffix in [('mean', '_mean'), ('mean_abs', '_mean_abs')]:
                    create_combined_histograms(all_results[mode], output_dir, mode_suffix)
                    create_combined_scatter_plots(all_results[mode], output_dir, mode_suffix)
                combined_time = time.time() - combined_start
                
                print(f"Combined plots created in {combined_time:.2f} seconds")
        elif args.normalize_gradients == 'both':
            if len(all_results['norm_sum']) > 1:
                print("\n" + "="*60)
                print("Creating combined plots for all layers...")
                print("="*60)
                
                combined_start = time.time()
                for norm_mode in ['norm_sum', 'norm_sum_abs']:
                    mode_suffix = f'_{args.aggregation_mode}_{norm_mode}'
                    create_combined_histograms(all_results[norm_mode], output_dir, mode_suffix)
                    create_combined_scatter_plots(all_results[norm_mode], output_dir, mode_suffix)
                combined_time = time.time() - combined_start
                
                print(f"Combined plots created in {combined_time:.2f} seconds")
        else:
            if len(all_results) > 1:
                print("\n" + "="*60)
                print("Creating combined plots for all layers...")
                print("="*60)
                
                combined_start = time.time()
                mode_suffix = f"_{args.aggregation_mode}"
                create_combined_histograms(all_results, output_dir, mode_suffix)
                create_combined_scatter_plots(all_results, output_dir, mode_suffix)
                combined_time = time.time() - combined_start
                
                print(f"Combined plots created in {combined_time:.2f} seconds")
        
        # Calculate number of layers processed
        if args.aggregation_mode == 'both' and args.normalize_gradients == 'both':
            num_layers_processed = len(all_results['mean']['norm_sum'])
        elif args.aggregation_mode == 'both':
            num_layers_processed = len(all_results['mean'])
        elif args.normalize_gradients == 'both':
            num_layers_processed = len(all_results['norm_sum'])
        else:
            num_layers_processed = len(all_results)
        print(f"\nCompleted analysis for {num_layers_processed} layers")
        print(f"Results saved to {output_dir}")
    
    # Save consolidated .npz file if requested
    if args.save_consolidated is not None:
        consolidated_path = args.save_consolidated
        if not consolidated_path.endswith('.npz'):
            consolidated_path += '.npz'
        
        save_consolidated_results(
            all_results=all_results,
            args=args,
            layers=layers,
            output_path=consolidated_path,
            aggregation_mode=args.aggregation_mode
        )
    elif output_dir is not None:
        # Auto-save consolidated file in output directory
        consolidated_path = os.path.join(output_dir, 'consolidated_results.npz')
        save_consolidated_results(
            all_results=all_results,
            args=args,
            layers=layers,
            output_path=consolidated_path,
            aggregation_mode=args.aggregation_mode
        )
    
    if output_dir is None:
        # Calculate number of layers processed
        if args.aggregation_mode == 'both' and args.normalize_gradients == 'both':
            num_layers_processed = len(all_results['mean']['norm_sum'])
        elif args.aggregation_mode == 'both':
            num_layers_processed = len(all_results['mean'])
        elif args.normalize_gradients == 'both':
            num_layers_processed = len(all_results['norm_sum'])
        else:
            num_layers_processed = len(all_results)
        print(f"\nCompleted analysis for {num_layers_processed} layers")
        print("Results were NOT saved (use --save_outputs to enable saving)")


if __name__ == "__main__":
    main()

