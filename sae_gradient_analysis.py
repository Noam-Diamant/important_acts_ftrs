"""
SAE Feature Gradient Analysis Experiment

This script analyzes how SAE features affect the language model loss by computing
gradients and visualizing their distributions across all 24 layers of Qwen2.5-0.5B.
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
sys.path.insert(0, str(project_root / "dictionary_learning"))
from dictionary_learning.utils import load_dictionary


def get_sae_path(model_name: str, layer: int) -> str:
    """Get SAE path for a given model and layer.
    
    This is a local implementation to avoid complex AlphaEdit dependencies.
    """
    base_path = project_root / "dictionary_learning_demo" / "._qwen2.5_0.5B_Qwen_Qwen2.5-0.5B_batch_top_k_tokens500M"
    return str(base_path / f"mlp_out_layer_{layer}" / "trainer_0")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Analyze SAE feature gradients across layers')
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


def load_saes(model_name: str, layers: List[int], device: str) -> Dict[int, torch.nn.Module]:
    """Load SAE models for specified layers.
    
    Args:
        model_name: Name of the model
        layers: List of layer indices
        device: Device to load SAEs on
    
    Returns:
        Dictionary mapping layer index to SAE model
    """
    saes = {}
    print(f"Loading SAEs for layers: {layers}")
    
    for layer in tqdm(layers, desc="Loading SAEs"):
        try:
            sae_path = get_sae_path(model_name, layer)
            sae, config = load_dictionary(sae_path, device)
            saes[layer] = sae
        except Exception as e:
            print(f"Warning: Could not load SAE for layer {layer}: {e}")
            continue
    
    print(f"Successfully loaded {len(saes)} SAEs")
    return saes


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
    sae,
    device: str,
    aggregation_mode: str = 'mean_abs'
) -> Union[np.ndarray, Dict[str, np.ndarray]]:
    """Compute gradients of loss w.r.t. SAE features for a specific layer.
    
    Args:
        model: Language model
        tokenizer: Tokenizer
        dataloader: DataLoader with text samples
        layer_idx: Index of the layer to analyze
        sae: SAE model for this layer
        device: Device to use
        aggregation_mode: 'mean', 'mean_abs', or 'both'
    
    Returns:
        Array of aggregated gradients for each SAE feature, or dict with both if mode is 'both'
    """
    print(f"\nProcessing layer {layer_idx}")
    
    # Storage for gradients across all samples
    all_gradients_signed = []
    all_gradients_abs = []
    
    # Get the MLP output hook name
    # For Qwen models, the structure is model.model.layers[i].mlp
    hook_name = f"model.layers.{layer_idx}.mlp"
    
    for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Layer {layer_idx}")):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        # Skip if batch is too small
        if input_ids.size(0) == 0:
            continue
        
        # Storage for this batch
        batch_sae_features = []
        batch_activations = []
        
        # Hook to capture MLP output activations
        def capture_activation_hook(module, input, output):
            # output shape: [batch, seq_len, hidden_dim]
            batch_activations.append(output.detach().clone())
            return output
        
        # Register hook
        target_module = model.model.layers[layer_idx].mlp
        hook_handle = target_module.register_forward_hook(capture_activation_hook)
        
        try:
            # Forward pass to get activations
            with torch.no_grad():
                _ = model(input_ids=input_ids, attention_mask=attention_mask)
            
            # Get captured activations
            if len(batch_activations) == 0:
                hook_handle.remove()
                continue
                
            activations = batch_activations[0]  # [batch, seq_len, hidden_dim]
            
            # Remove hook
            hook_handle.remove()
            
            # Encode through SAE (with gradients enabled)
            # We'll process each position separately to compute gradients
            batch_size, seq_len, hidden_dim = activations.shape
            
            # Flatten to [batch * seq_len, hidden_dim]
            activations_flat = activations.reshape(-1, hidden_dim)
            
            # Only process non-padded positions
            mask_flat = attention_mask.reshape(-1)
            valid_indices = mask_flat.nonzero(as_tuple=True)[0]
            
            if len(valid_indices) == 0:
                continue
            
            # Take a subset of positions for efficiency (e.g., every 4th position)
            valid_indices = valid_indices[::4]
            if len(valid_indices) == 0:
                continue
            
            valid_activations = activations_flat[valid_indices]
            
            # Encode through SAE with gradients
            sae_features = sae.encode(valid_activations)
            sae_features.requires_grad_(True)
            
            # Decode back
            reconstructed = sae.decode(sae_features)
            
            # Now we need to compute loss
            # We'll do a forward pass with reconstructed activations
            
            # Create a hook to replace activations with reconstructed
            reconstructed_idx = 0
            
            def replace_activation_hook(module, input, output):
                nonlocal reconstructed_idx
                # Replace valid positions with reconstructed activations
                output_flat = output.reshape(-1, hidden_dim)
                output_flat[valid_indices] = reconstructed
                return output_flat.reshape(output.shape)
            
            # Register replacement hook
            hook_handle2 = target_module.register_forward_hook(replace_activation_hook)
            
            # Forward pass with replaced activations
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss
            
            # Remove hook
            hook_handle2.remove()
            
            # Compute gradients
            if loss.requires_grad and sae_features.requires_grad:
                grads = torch.autograd.grad(
                    loss,
                    sae_features,
                    retain_graph=False,
                    create_graph=False
                )
                
                # Get gradients for each mode
                # [num_valid_positions, num_features]
                
                if aggregation_mode in ['mean', 'both']:
                    # Average signed gradients across positions for this batch
                    grad_mean_signed = grads[0].mean(dim=0)  # [num_features]
                    all_gradients_signed.append(grad_mean_signed.detach().cpu().numpy())
                
                if aggregation_mode in ['mean_abs', 'both']:
                    # Average absolute gradients across positions for this batch
                    grad_abs = torch.abs(grads[0])  # [num_valid_positions, num_features]
                    grad_mean_abs = grad_abs.mean(dim=0)  # [num_features]
                    all_gradients_abs.append(grad_mean_abs.detach().cpu().numpy())
        
        except Exception as e:
            print(f"Error processing batch {batch_idx} for layer {layer_idx}: {e}")
            if 'hook_handle' in locals():
                hook_handle.remove()
            if 'hook_handle2' in locals():
                hook_handle2.remove()
            continue
    
    # Compute mean across all batches
    if aggregation_mode == 'mean':
        if len(all_gradients_signed) == 0:
            print(f"Warning: No gradients computed for layer {layer_idx}")
            return None
        mean_gradients = np.mean(all_gradients_signed, axis=0)
        print(f"Layer {layer_idx}: Mean gradient = {mean_gradients.mean():.6f}, "
              f"Max gradient = {np.abs(mean_gradients).max():.6f}, "
              f"Std gradient = {mean_gradients.std():.6f}")
        return mean_gradients
    
    elif aggregation_mode == 'mean_abs':
        if len(all_gradients_abs) == 0:
            print(f"Warning: No gradients computed for layer {layer_idx}")
            return None
        mean_gradients = np.mean(all_gradients_abs, axis=0)
        print(f"Layer {layer_idx}: Mean abs gradient = {mean_gradients.mean():.6f}, "
              f"Max gradient = {mean_gradients.max():.6f}, "
              f"Std gradient = {mean_gradients.std():.6f}")
        return mean_gradients
    
    else:  # both
        if len(all_gradients_signed) == 0 or len(all_gradients_abs) == 0:
            print(f"Warning: No gradients computed for layer {layer_idx}")
            return None
        mean_gradients_signed = np.mean(all_gradients_signed, axis=0)
        mean_gradients_abs = np.mean(all_gradients_abs, axis=0)
        print(f"Layer {layer_idx}:")
        print(f"  Mean gradient (signed) = {mean_gradients_signed.mean():.6f}, "
              f"Max = {np.abs(mean_gradients_signed).max():.6f}, Std = {mean_gradients_signed.std():.6f}")
        print(f"  Mean gradient (abs) = {mean_gradients_abs.mean():.6f}, "
              f"Max = {mean_gradients_abs.max():.6f}, Std = {mean_gradients_abs.std():.6f}")
        return {'mean': mean_gradients_signed, 'mean_abs': mean_gradients_abs}


def create_histogram(gradients: np.ndarray, layer_idx: int, output_dir: str, mode_suffix: str = ""):
    """Create and save histogram plot for gradient magnitudes.
    
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
    plt.ylabel('Number of Features', fontsize=12)
    plt.title(f'Layer {layer_idx}: SAE Feature Gradient Distribution ({mode_name})\n'
              f'Max: {max_grad:.6f}, Mean: {mean_grad:.6f}, Std: {std_grad:.6f}',
              fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save plot
    output_path = os.path.join(output_dir, f'layer_{layer_idx}{mode_suffix}_gradient_hist.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved histogram to {output_path}")


def create_feature_gradient_plot(gradients: np.ndarray, layer_idx: int, output_dir: str):
    """Create and save plot showing gradient magnitude per feature.
    
    Args:
        gradients: Array of mean absolute gradients (one per feature)
        layer_idx: Layer index
        output_dir: Directory to save plot
    """
    plt.figure(figsize=(14, 6))
    
    num_features = len(gradients)
    feature_indices = np.arange(num_features)
    
    # Plot gradient magnitude per feature
    plt.plot(feature_indices, gradients, linewidth=0.5, alpha=0.7, color='blue')
    plt.scatter(feature_indices[::max(1, num_features//1000)], 
                gradients[::max(1, num_features//1000)], 
                s=1, alpha=0.5, color='blue')
    
    # Add statistics
    mean_grad = gradients.mean()
    max_grad = gradients.max()
    std_grad = gradients.std()
    
    # Highlight top features
    top_k = min(10, num_features)
    top_indices = np.argsort(gradients)[-top_k:]
    plt.scatter(top_indices, gradients[top_indices], 
                s=50, color='red', alpha=0.7, 
                label=f'Top {top_k} features', zorder=5)
    
    plt.axhline(mean_grad, color='green', linestyle='--', linewidth=1, 
                label=f'Mean: {mean_grad:.6f}', alpha=0.7)
    
    plt.xlabel('Feature Index', fontsize=12)
    plt.ylabel('Mean Absolute Gradient Magnitude', fontsize=12)
    plt.title(f'Layer {layer_idx}: Gradient Magnitude per SAE Feature\n'
              f'Total Features: {num_features}, Max: {max_grad:.6f}, Mean: {mean_grad:.6f}, Std: {std_grad:.6f}',
              fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save plot
    output_path = os.path.join(output_dir, f'layer_{layer_idx}_feature_gradients.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved feature gradient plot to {output_path}")


def create_feature_scatter_plot(gradients: np.ndarray, layer_idx: int, output_dir: str, mode_suffix: str = ""):
    """Create and save scatter plot showing gradient magnitude per feature (points only).
    
    Args:
        gradients: Array of aggregated gradients (one per feature)
        layer_idx: Layer index
        output_dir: Directory to save plot
        mode_suffix: Suffix for filename (e.g., "_mean" or "_mean_abs")
    """
    plt.figure(figsize=(14, 6))
    
    num_features = len(gradients)
    feature_indices = np.arange(num_features)
    
    # Plot gradient magnitude per feature (scatter only, no line)
    plt.scatter(feature_indices, gradients, s=3, alpha=0.6, color='blue')
    
    # Add statistics
    mean_grad = gradients.mean()
    max_grad = np.abs(gradients).max()
    std_grad = gradients.std()
    
    # Highlight top features
    top_k = min(20, num_features)
    top_indices = np.argsort(np.abs(gradients))[-top_k:]
    plt.scatter(top_indices, gradients[top_indices], 
                s=80, color='red', alpha=0.8, 
                label=f'Top {top_k} features', zorder=5, marker='*')
    
    plt.axhline(mean_grad, color='green', linestyle='--', linewidth=1.5, 
                label=f'Mean: {mean_grad:.6f}', alpha=0.7)
    
    mode_name = "Signed Mean" if "mean" in mode_suffix and "abs" not in mode_suffix else "Mean Absolute"
    plt.xlabel('Feature Index', fontsize=12)
    plt.ylabel(f'Gradient Magnitude ({mode_name})', fontsize=12)
    plt.title(f'Layer {layer_idx}: Feature Gradient Scatter Plot ({mode_name})\n'
              f'Total Features: {num_features}, Max: {max_grad:.6f}, Mean: {mean_grad:.6f}, Std: {std_grad:.6f}',
              fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save plot
    output_path = os.path.join(output_dir, f'layer_{layer_idx}{mode_suffix}_feature_scatter.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved feature scatter plot to {output_path}")


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
            ax.set_ylabel('Number of Features', fontsize=12)
            ax.set_title(f'{mode_name}\nMean: {mean_grad:.6f}, Max: {np.abs(gradients).max():.6f}', fontsize=12)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        else:  # scatter
            num_features = len(gradients)
            feature_indices = np.arange(num_features)
            
            # Scatter plot
            ax.scatter(feature_indices, gradients, s=3, alpha=0.6, color='blue')
            
            # Add statistics
            mean_grad = gradients.mean()
            
            # Highlight top features
            top_k = min(20, num_features)
            top_indices = np.argsort(np.abs(gradients))[-top_k:]
            ax.scatter(top_indices, gradients[top_indices], 
                      s=80, color='red', alpha=0.8, 
                      label=f'Top {top_k}', zorder=5, marker='*')
            
            ax.axhline(mean_grad, color='green', linestyle='--', linewidth=1.5, 
                      label=f'Mean: {mean_grad:.6f}', alpha=0.7)
            
            ax.set_xlabel('Feature Index', fontsize=12)
            ax.set_ylabel(f'Gradient Magnitude ({mode_name})', fontsize=12)
            ax.set_title(f'{mode_name}\nMean: {mean_grad:.6f}, Max: {np.abs(gradients).max():.6f}', fontsize=12)
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    fig.suptitle(f'Layer {layer_idx}: SAE Feature Gradient Comparison', fontsize=16)
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
    fig.suptitle(f'SAE Feature Gradient Distribution - All Layers ({mode_name})', fontsize=16, y=0.995)
    
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
    fig.suptitle(f'SAE Feature Gradient Scatter - All Layers ({mode_name})', fontsize=16, y=0.995)
    
    # Flatten axes for easier iteration
    if num_layers == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx, layer_idx in enumerate(layers):
        ax = axes[idx]
        gradients = all_gradients[layer_idx]
        
        num_features = len(gradients)
        feature_indices = np.arange(num_features)
        
        # Scatter plot with smaller points
        ax.scatter(feature_indices, gradients, s=0.5, alpha=0.4, color='blue')
        
        # Add mean line
        mean_grad = gradients.mean()
        ax.axhline(mean_grad, color='green', linestyle='--', linewidth=1, alpha=0.7)
        
        # Highlight top features
        top_k = min(20, num_features)
        top_indices = np.argsort(gradients)[-top_k:]
        ax.scatter(top_indices, gradients[top_indices], 
                  s=15, color='red', alpha=0.8, marker='*', zorder=5)
        
        ax.set_xlabel('Feature Index', fontsize=8)
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
    output_path = os.path.join(output_dir, f'all_layers{mode_suffix}_feature_scatter.png')
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"Saved combined scatter plot to {output_path}")


def main():
    """Main function to run the experiment."""
    start_time_total = time.time()
    timing_stats = {}
    
    args = parse_args()
    
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
        output_dir = f"model_{model_safe}_num_samples_{args.num_samples}_batch_size_{args.batch_size}_layers_{layers_safe}_{timestamp}"
        #output_dir = os.path.join("important_feaures", output_dir)
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
    
    # Load SAEs
    print("\n" + "="*60)
    print("TIMING: Loading SAEs")
    print("="*60)
    start_time = time.time()
    saes = load_saes(args.model_name, layers, args.device)
    timing_stats['sae_loading'] = time.time() - start_time
    print(f"SAE loading took: {timing_stats['sae_loading']:.2f} seconds")
    
    if len(saes) == 0:
        print("Error: No SAEs loaded. Exiting.")
        return
    
    # Prepare dataset
    print("\n" + "="*60)
    print("TIMING: Preparing Dataset")
    print("="*60)
    start_time = time.time()
    dataloader = prepare_dataset(tokenizer, args.num_samples, args.max_length, args.batch_size)
    timing_stats['dataset_preparation'] = time.time() - start_time
    print(f"Dataset preparation took: {timing_stats['dataset_preparation']:.2f} seconds")
    
    # Process each layer
    all_results = {} if args.aggregation_mode != 'both' else {'mean': {}, 'mean_abs': {}}
    layer_timings = {}
    
    print("\n" + "="*60)
    print("TIMING: Processing Layers")
    print("="*60)
    
    for layer_idx in layers:
        if layer_idx not in saes:
            print(f"Skipping layer {layer_idx} (SAE not loaded)")
            continue
        
        layer_start_time = time.time()
        layer_timing = {}
        
        # Compute gradients
        grad_start = time.time()
        gradients = compute_gradients_for_layer(
            model=model,
            tokenizer=tokenizer,
            dataloader=dataloader,
            layer_idx=layer_idx,
            sae=saes[layer_idx],
            device=args.device,
            aggregation_mode=args.aggregation_mode
        )
        layer_timing['gradient_computation'] = time.time() - grad_start
        
        if gradients is not None:
            if args.aggregation_mode == 'both':
                # Save results for both modes
                all_results['mean'][layer_idx] = gradients['mean']
                all_results['mean_abs'][layer_idx] = gradients['mean_abs']
                
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
                
            else:
                # Single mode
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
    
    # Calculate summary statistics
    total_grad_time = 0
    total_plot_time = 0
    if layer_timings:
        total_grad_time = sum(t['gradient_computation'] for t in layer_timings.values())
        # total_plot_time = sum(t['histogram_plot'] + t['feature_plot'] + t['scatter_plot'] 
        #                     for t in layer_timings.values())
        total_plot_time = sum(t['histogram_plot'] + t['scatter_plot'] 
                            for t in layer_timings.values())    
    # Print final timing summary
    summary_lines = [
        "\n" + "="*60,
        "FINAL TIMING SUMMARY",
        "="*60,
        f"Model loading:         {timing_stats['model_loading']:.2f} seconds",
        f"SAE loading:           {timing_stats['sae_loading']:.2f} seconds",
        f"Dataset preparation:   {timing_stats['dataset_preparation']:.2f} seconds",
    ]
    
    if layer_timings:
        summary_lines.extend([
            f"Total gradient comp:   {total_grad_time:.2f} seconds",
            f"Total plotting:        {total_plot_time:.2f} seconds",
        ])
    
    summary_lines.extend([
        f"Total runtime:         {timing_stats['total_runtime']:.2f} seconds",
        "="*60,
    ])
    
    # Print to screen
    for line in summary_lines:
        print(line)
    
    # Save timing stats if output directory exists
    if output_dir is not None:
        # Save timing stats to JSON
        timing_json_file = os.path.join(output_dir, 'timing_stats.json')
        import json
        with open(timing_json_file, 'w') as f:
            json.dump({
                'overall': timing_stats,
                'per_layer': layer_timings
            }, f, indent=2)
        print(f"\nTiming statistics (JSON) saved to {timing_json_file}")
        
        # Save timing summary to text file
        timing_txt_file = os.path.join(output_dir, 'timing_summary.txt')
        with open(timing_txt_file, 'w') as f:
            # Write summary
            for line in summary_lines:
                f.write(line + '\n')
            
            # Write detailed per-layer breakdown
            if layer_timings:
                f.write('\n\nDETAILED PER-LAYER TIMING\n')
                f.write('='*60 + '\n')
                for layer_idx in sorted(layer_timings.keys()):
                    lt = layer_timings[layer_idx]
                    f.write(f'\nLayer {layer_idx}:\n')
                    f.write(f'  Gradient computation: {lt["gradient_computation"]:.2f}s\n')
                    f.write(f'  Histogram plot:       {lt["histogram_plot"]:.2f}s\n')
                    # f.write(f'  Feature plot:         {lt["feature_plot"]:.2f}s\n')
                    f.write(f'  Scatter plot:         {lt["scatter_plot"]:.2f}s\n')
                    f.write(f'  Save data:            {lt["save_data"]:.2f}s\n')
                    f.write(f'  Total for layer:      {lt["total"]:.2f}s\n')
        
        print(f"Timing summary (TXT) saved to {timing_txt_file}")
        
        # Create combined plots showing all layers
        if args.aggregation_mode == 'both':
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
        
        num_layers_processed = len(all_results) if args.aggregation_mode != 'both' else len(all_results['mean'])
        print(f"\nCompleted analysis for {num_layers_processed} layers")
        print(f"Results saved to {output_dir}")
    else:
        num_layers_processed = len(all_results) if args.aggregation_mode != 'both' else len(all_results['mean'])
        print(f"\nCompleted analysis for {num_layers_processed} layers")
        print("Results were NOT saved (use --save_outputs to enable saving)")


if __name__ == "__main__":
    main()

