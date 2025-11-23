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


class TeeOutput:
    """Writes output to both stdout and a file simultaneously."""
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w', buffering=1)  # Line buffered
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    
    def close(self):
        self.log.close()

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
    parser.add_argument('--normalize_mode', type=str, default='sum_abs',
                        choices=['none', 'sum_abs', 'sum', 'both', 'all'],
                        help='Normalization mode: none (no normalization), sum_abs (sum of absolute values = dim_size), sum (sum = dim_size), both (apply both sum and sum_abs), all (apply all three: none, sum_abs, and sum) (default: sum_abs)')
    parser.add_argument('--aggregate_by', type=str, default='values',
                        choices=['gradients', 'values', 'hessian_diagonal_sum', 'hessian_diagonal'],
                        help='What to aggregate: gradients (compute loss gradients), values (raw activation values), hessian_diagonal_sum (fast approximation using sum of gradients), or hessian_diagonal (true second derivatives - slower but accurate) (default: values)')
    parser.add_argument('--power', type=str, default=None,
                        help='Power to raise aggregated values to before normalization. Can be a single value (e.g., "2.0") or comma-separated list (e.g., "0.5,1.0,2.0"). If list provided, only first value is used. (optional, default: None)')
    parser.add_argument('--save_consolidated', type=str, default=None,
                        help='Path to save consolidated .npz file with all results (optional)')
    args = parser.parse_args()
    
    # Parse power argument - convert to single value or None
    if args.power is not None:
        power_values = [float(p.strip()) for p in args.power.split(',')]
        if len(power_values) > 1:
            # Multiple values provided - store for later, will use first one
            args.power_list = power_values
            args.power = power_values[0]
        else:
            args.power_list = None
            args.power = power_values[0]
    else:
        args.power_list = None
    
    return args


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


def normalize_values(values: np.ndarray, dimension_size: int, mode: str = 'sum_abs') -> np.ndarray:
    """Normalize array values based on the specified mode.
    
    Args:
        values: Array of values (gradients or activations)
        dimension_size: Target dimension size for normalization
        mode: Normalization mode ('none', 'sum_abs', or 'sum')
            - 'none': No normalization, returns original values
            - 'sum_abs': Normalize so sum(abs(values)) == dimension_size
            - 'sum': Normalize so sum(values) == dimension_size
    
    Returns:
        Normalized array based on the specified mode
    """
    if mode == 'none':
        return values
    
    elif mode == 'sum_abs':
        abs_sum = np.sum(np.abs(values))
        if abs_sum == 0:
            print("Warning: Sum of absolute values is zero, returning original values")
            return values
        normalized = values * (dimension_size / abs_sum)
        return normalized
    
    elif mode == 'sum':
        total_sum = np.sum(values)
        if total_sum == 0:
            print("Warning: Sum of values is zero, returning original values")
            return values
        normalized = values * (dimension_size / total_sum)
        return normalized
    
    else:
        raise ValueError(f"Unknown normalization mode: {mode}. Use 'none', 'sum_abs', or 'sum'")


def load_model_and_tokenizer(model_name: str, device: str, use_eager_attention: bool = False):
    """Load the language model and tokenizer.
    
    Args:
        model_name: Name of the model to load
        device: Device to load model on
        use_eager_attention: If True, force eager attention (supports second derivatives)
    
    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Set padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Prepare model loading kwargs
    model_kwargs = {
        'torch_dtype': torch.float32,  # Use float32 for gradient computation
        'device_map': device,
    }
    
    # Force eager attention if needed (supports second derivatives)
    if use_eager_attention:
        model_kwargs['attn_implementation'] = 'eager'
        print("Forcing eager attention implementation (supports second derivatives)")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        **model_kwargs
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


def compute_layer_aggregates(
    model,
    tokenizer,
    dataloader,
    layer_idx: int,
    device: str,
    aggregation_mode: str,
    model_config: Dict[str, str],
    normalize: str = 'none',
    aggregate_by: str = 'values',
    power: Optional[float] = None
) -> Union[np.ndarray, Dict[str, np.ndarray]]:
    """Compute aggregated statistics (gradients, values, or Hessian diagonal) for raw MLP output activations.
    
    This function computes either:
    - Gradients of loss w.r.t. raw MLP output activations (when aggregate_by='gradients')
    - Raw activation values themselves (when aggregate_by='values')
    - Diagonal of Hessian matrix approximation (when aggregate_by='hessian_diagonal_sum')
    - True diagonal of Hessian matrix (second derivatives) (when aggregate_by='hessian_diagonal')
    
    Args:
        model: Language model
        tokenizer: Tokenizer
        dataloader: DataLoader with text samples
        layer_idx: Index of the layer to analyze
        device: Device to use
        aggregation_mode: 'mean', 'mean_abs', or 'both'
        model_config: Model configuration dictionary
        normalize: Normalization mode ('none', 'sum_abs', or 'sum')
        aggregate_by: 'gradients' (first derivatives), 'values' (raw activations), 'hessian_diagonal_sum' (fast approximation), or 'hessian_diagonal' (true second derivatives)
        power: Optional power to raise aggregated values to before normalization
    
    Returns:
        Array of aggregated values for each activation dimension, or dict with both if mode is 'both'
    """
    print(f"\nProcessing layer {layer_idx} (aggregate_by={aggregate_by}, power={power})")
    
    # NOTE: SDP kernel settings for Hessian computation are set in main() before model loading
    # This is required because the settings must be applied before the model is loaded
    
    # Storage for gradients/values across all samples
    all_gradients_signed = []
    all_gradients_abs = []
    
    for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Layer {layer_idx}")):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        # Skip if batch is too small
        if input_ids.size(0) == 0:
            continue
        
        # Register hook
        target_module = get_layer_module(model, layer_idx, model_config)
        
        try:
            if aggregate_by == 'values':
                # Simpler path: just aggregate activation values
                batch_activations = []
                
                def capture_activation_hook(module, input, output):
                    batch_activations.append(output.detach().clone())
                    return output
                
                hook_handle = target_module.register_forward_hook(capture_activation_hook)
                
                # Forward pass to capture activations
                with torch.no_grad():
                    _ = model(input_ids=input_ids, attention_mask=attention_mask)
                
                hook_handle.remove()
                
                if len(batch_activations) == 0:
                    continue
                
                activations = batch_activations[0]  # [batch, seq_len, hidden_dim]
                batch_size, seq_len, hidden_dim = activations.shape
                mask_expanded = attention_mask.unsqueeze(-1).expand_as(activations)
                
                # Apply mask and compute mean across valid positions
                valid_acts = activations * mask_expanded
                num_valid = attention_mask.sum()
                
                if num_valid > 0:
                    # Compute mean across batch and sequence dimensions
                    if aggregation_mode in ['mean', 'both']:
                        act_mean_signed = valid_acts.sum(dim=(0, 1)) / num_valid  # [hidden_dim]
                        all_gradients_signed.append(act_mean_signed.cpu().numpy())
                    
                    if aggregation_mode in ['mean_abs', 'both']:
                        act_mean_abs = torch.abs(valid_acts).sum(dim=(0, 1)) / num_valid  # [hidden_dim]
                        all_gradients_abs.append(act_mean_abs.cpu().numpy())
            
            elif aggregate_by == 'gradients':
                # Original gradient computation path
                # Storage for this batch
                batch_activations = []
                
                # Hook to capture MLP output activations
                def capture_activation_hook(module, input, output):
                    # output shape: [batch, seq_len, hidden_dim]
                    batch_activations.append(output)
                    return output
                
                hook_handle = target_module.register_forward_hook(capture_activation_hook)
                
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
            
            elif aggregate_by == 'hessian_diagonal_sum':
                # Hessian diagonal approximation (fast but approximate)
                # Computes grad(sum(first_grad), activations) which gives curvature information
                # but not the true diagonal Hessian d²L/d(act_i)²
                activations_with_grad = None
                
                # Hook to replace activations with gradient-enabled version
                def replace_activation_hook(module, input, output):
                    nonlocal activations_with_grad
                    activations_with_grad = output.detach().clone()
                    activations_with_grad.requires_grad_(True)
                    return activations_with_grad
                
                hook_handle = target_module.register_forward_hook(replace_activation_hook)
                
                # Forward pass with replaced activations
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
                loss = outputs.loss
                
                # Remove hook
                hook_handle.remove()
                
                if activations_with_grad is None:
                    continue
                
                # Compute Hessian diagonal approximation efficiently
                if loss.requires_grad and activations_with_grad.requires_grad:
                    # First, compute gradient with create_graph=True and retain_graph=True
                    # We need retain_graph=True because we'll do a second backward pass
                    grads = torch.autograd.grad(
                        loss,
                        activations_with_grad,
                        retain_graph=True,  # Must retain graph for second backward pass
                        create_graph=True
                    )
                    
                    # grads[0] shape: [batch, seq_len, hidden_dim]
                    first_grad = grads[0]
                    batch_size, seq_len, hidden_dim = first_grad.shape
                    mask_expanded = attention_mask.unsqueeze(-1).expand_as(first_grad)
                    
                    # Compute Hessian-based curvature information
                    # This computes grad(sum(first_grad), activations) which gives us
                    # curvature information but not the true diagonal Hessian d²L/d(act_i)²
                    # This approximation captures how activations affect gradient magnitude
                    hessian_diag = torch.autograd.grad(
                        outputs=first_grad,
                        inputs=activations_with_grad,
                        grad_outputs=torch.ones_like(first_grad),
                        retain_graph=False,  # Can free graph after second backward
                        create_graph=False,
                        allow_unused=True
                    )[0]
                    
                    # If gradient is None (unused), create zeros
                    if hessian_diag is None:
                        hessian_diag = torch.zeros_like(activations_with_grad)
                    
                    # Apply mask and compute mean across valid positions
                    valid_hessian = hessian_diag * mask_expanded
                    num_valid = attention_mask.sum()
                    
                    if num_valid > 0:
                        # Compute mean across batch and sequence dimensions
                        if aggregation_mode in ['mean', 'both']:
                            hess_mean_signed = valid_hessian.sum(dim=(0, 1)) / num_valid  # [hidden_dim]
                            all_gradients_signed.append(hess_mean_signed.detach().cpu().numpy())
                        
                        if aggregation_mode in ['mean_abs', 'both']:
                            hess_mean_abs = torch.abs(valid_hessian).sum(dim=(0, 1)) / num_valid  # [hidden_dim]
                            all_gradients_abs.append(hess_mean_abs.detach().cpu().numpy())
            
            else:  # aggregate_by == 'hessian_diagonal'
                # True Hessian diagonal computation (slow but accurate)
                # Computes d²L/d(act[i])² for each dimension i
                activations_with_grad = None
                
                # Hook to replace activations with gradient-enabled version
                def replace_activation_hook(module, input, output):
                    nonlocal activations_with_grad
                    activations_with_grad = output.detach().clone()
                    activations_with_grad.requires_grad_(True)
                    return activations_with_grad
                
                hook_handle = target_module.register_forward_hook(replace_activation_hook)
                
                # Forward pass with replaced activations
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
                loss = outputs.loss
                
                # Remove hook
                hook_handle.remove()
                
                if activations_with_grad is None:
                    continue
                
                # Compute true Hessian diagonal (d²L/d(act[i])² for each i)
                if loss.requires_grad and activations_with_grad.requires_grad:
                    # First, compute gradient with create_graph=True and retain_graph=True
                    # We need retain_graph=True because we'll do multiple second backward passes
                    grads = torch.autograd.grad(
                        loss,
                        activations_with_grad,
                        retain_graph=True,  # Must retain graph for second backward passes
                        create_graph=True
                    )
                    
                    # grads[0] shape: [batch, seq_len, hidden_dim]
                    first_grad = grads[0]
                    batch_size, seq_len, hidden_dim = first_grad.shape
                    mask_expanded = attention_mask.unsqueeze(-1).expand_as(first_grad)
                    
                    # Compute true Hessian diagonal: d²L/d(act[i])² for each dimension i
                    # This requires computing the gradient of each component of first_grad
                    # with respect to the corresponding component of activations
                    # NOTE: This is computationally expensive (O(batch * seq * hidden_dim) backward passes)
                    # but gives the true second derivatives
                    hessian_diag = torch.zeros_like(activations_with_grad)
                    
                    # Get valid positions from mask
                    valid_positions = attention_mask.nonzero(as_tuple=False)  # [num_valid, 2] with (b, s) indices
                    num_valid = valid_positions.shape[0]
                    
                    if num_valid > 0 and batch_idx == 0:
                        print(f"  Computing true Hessian diagonal for {num_valid} positions × {hidden_dim} dimensions "
                              f"(this will be slow but accurate)...")
                    
                    # For each valid position and dimension, compute d²L/d(act[b, s, i])²
                    for pos_idx, (b, s) in enumerate(valid_positions):
                        b, s = b.item(), s.item()
                        
                        if pos_idx > 0 and pos_idx % 100 == 0:
                            print(f"    Processed {pos_idx}/{num_valid} positions...")
                        
                        for dim_idx in range(hidden_dim):
                            # Compute second derivative for this specific position and dimension
                            grad_element = first_grad[b, s, dim_idx]
                            act_element = activations_with_grad[b, s, dim_idx]
                            
                            # Compute d(grad_element)/d(act_element) = d²L/d(act[b, s, dim_idx])²
                            second_grad = torch.autograd.grad(
                                outputs=grad_element,
                                inputs=act_element,
                                grad_outputs=torch.ones_like(grad_element),
                                retain_graph=True,  # Keep graph for next position/dimension
                                create_graph=False,
                                allow_unused=True
                            )[0]
                            
                            if second_grad is not None:
                                hessian_diag[b, s, dim_idx] = second_grad
                    
                    # Apply mask and compute mean across valid positions
                    valid_hessian = hessian_diag * mask_expanded
                    num_valid = attention_mask.sum()
                    
                    if num_valid > 0:
                        # Compute mean across batch and sequence dimensions
                        if aggregation_mode in ['mean', 'both']:
                            hess_mean_signed = valid_hessian.sum(dim=(0, 1)) / num_valid  # [hidden_dim]
                            all_gradients_signed.append(hess_mean_signed.detach().cpu().numpy())
                        
                        if aggregation_mode in ['mean_abs', 'both']:
                            hess_mean_abs = torch.abs(valid_hessian).sum(dim=(0, 1)) / num_valid  # [hidden_dim]
                            all_gradients_abs.append(hess_mean_abs.detach().cpu().numpy())
        
        except Exception as e:
            print(f"Error processing batch {batch_idx} for layer {layer_idx}: {e}")
            if 'hook_handle' in locals():
                hook_handle.remove()
            continue
    
    # Compute mean across all batches
    if aggregation_mode == 'mean':
        if len(all_gradients_signed) == 0:
            print(f"Warning: No values computed for layer {layer_idx}")
            return None
        mean_gradients = np.mean(all_gradients_signed, axis=0)
        dimension_size = len(mean_gradients)
        
        value_type = "value" if aggregate_by == 'values' else ("Hessian diag (sum)" if aggregate_by == 'hessian_diagonal_sum' else ("Hessian diag" if aggregate_by == 'hessian_diagonal' else "gradient"))
        print(f"Layer {layer_idx} (before power/normalization): Mean {value_type} = {mean_gradients.mean():.6f}, "
              f"Max {value_type} = {np.abs(mean_gradients).max():.6f}, "
              f"Std {value_type} = {mean_gradients.std():.6f}")
        
        # Apply power transformation if specified
        if power is not None:
            mean_gradients = np.power(np.abs(mean_gradients), power) * np.sign(mean_gradients)
            print(f"Layer {layer_idx} (after power={power}): Mean = {mean_gradients.mean():.6f}, "
                  f"Max = {np.abs(mean_gradients).max():.6f}")
        
        if normalize == 'none':
            return mean_gradients
        elif normalize == 'both':
            # Apply both normalizations
            norm_sum = normalize_values(mean_gradients, dimension_size, mode='sum')
            norm_sum_abs = normalize_values(mean_gradients, dimension_size, mode='sum_abs')
            print(f"Layer {layer_idx} (after normalization):")
            print(f"  norm_sum: Sum = {np.sum(norm_sum):.1f} (target: {dimension_size})")
            print(f"  norm_sum_abs: Sum of abs = {np.sum(np.abs(norm_sum_abs)):.1f} (target: {dimension_size})")
            return {'norm_sum': norm_sum, 'norm_sum_abs': norm_sum_abs}
        elif normalize == 'all':
            # Apply all three: none, sum, and sum_abs
            norm_none = mean_gradients.copy()
            norm_sum = normalize_values(mean_gradients, dimension_size, mode='sum')
            norm_sum_abs = normalize_values(mean_gradients, dimension_size, mode='sum_abs')
            print(f"Layer {layer_idx} (after all normalizations):")
            print(f"  none: No normalization applied")
            print(f"  norm_sum: Sum = {np.sum(norm_sum):.1f} (target: {dimension_size})")
            print(f"  norm_sum_abs: Sum of abs = {np.sum(np.abs(norm_sum_abs)):.1f} (target: {dimension_size})")
            return {'none': norm_none, 'norm_sum': norm_sum, 'norm_sum_abs': norm_sum_abs}
        else:
            mean_gradients = normalize_values(mean_gradients, dimension_size, mode=normalize)
            if normalize == 'sum_abs':
                print(f"Layer {layer_idx} (after {normalize}): Sum of abs values = {np.sum(np.abs(mean_gradients)):.1f} (target: {dimension_size})")
            elif normalize == 'sum':
                print(f"Layer {layer_idx} (after {normalize}): Sum = {np.sum(mean_gradients):.1f} (target: {dimension_size})")
            return mean_gradients
    
    elif aggregation_mode == 'mean_abs':
        if len(all_gradients_abs) == 0:
            print(f"Warning: No values computed for layer {layer_idx}")
            return None
        mean_gradients = np.mean(all_gradients_abs, axis=0)
        dimension_size = len(mean_gradients)
        
        value_type = "value" if aggregate_by == 'values' else ("Hessian diag (sum)" if aggregate_by == 'hessian_diagonal_sum' else ("Hessian diag" if aggregate_by == 'hessian_diagonal' else "gradient"))
        print(f"Layer {layer_idx} (before power/normalization): Mean abs {value_type} = {mean_gradients.mean():.6f}, "
              f"Max {value_type} = {mean_gradients.max():.6f}, "
              f"Std {value_type} = {mean_gradients.std():.6f}")
        
        # Apply power transformation if specified
        if power is not None:
            mean_gradients = np.power(mean_gradients, power)
            print(f"Layer {layer_idx} (after power={power}): Mean = {mean_gradients.mean():.6f}, "
                  f"Max = {mean_gradients.max():.6f}")
        
        if normalize == 'none':
            return mean_gradients
        elif normalize == 'both':
            # Apply both normalizations
            norm_sum = normalize_values(mean_gradients, dimension_size, mode='sum')
            norm_sum_abs = normalize_values(mean_gradients, dimension_size, mode='sum_abs')
            print(f"Layer {layer_idx} (after normalization):")
            print(f"  norm_sum: Sum = {np.sum(norm_sum):.1f} (target: {dimension_size})")
            print(f"  norm_sum_abs: Sum of abs = {np.sum(np.abs(norm_sum_abs)):.1f} (target: {dimension_size})")
            return {'norm_sum': norm_sum, 'norm_sum_abs': norm_sum_abs}
        elif normalize == 'all':
            # Apply all three: none, sum, and sum_abs
            norm_none = mean_gradients.copy()
            norm_sum = normalize_values(mean_gradients, dimension_size, mode='sum')
            norm_sum_abs = normalize_values(mean_gradients, dimension_size, mode='sum_abs')
            print(f"Layer {layer_idx} (after all normalizations):")
            print(f"  none: No normalization applied")
            print(f"  norm_sum: Sum = {np.sum(norm_sum):.1f} (target: {dimension_size})")
            print(f"  norm_sum_abs: Sum of abs = {np.sum(np.abs(norm_sum_abs)):.1f} (target: {dimension_size})")
            return {'none': norm_none, 'norm_sum': norm_sum, 'norm_sum_abs': norm_sum_abs}
        else:
            mean_gradients = normalize_values(mean_gradients, dimension_size, mode=normalize)
            if normalize == 'sum_abs':
                print(f"Layer {layer_idx} (after {normalize}): Sum of abs values = {np.sum(np.abs(mean_gradients)):.1f} (target: {dimension_size})")
            elif normalize == 'sum':
                print(f"Layer {layer_idx} (after {normalize}): Sum = {np.sum(mean_gradients):.1f} (target: {dimension_size})")
            return mean_gradients
    
    else:  # both aggregation modes
        if len(all_gradients_signed) == 0 or len(all_gradients_abs) == 0:
            print(f"Warning: No values computed for layer {layer_idx}")
            return None
        mean_gradients_signed = np.mean(all_gradients_signed, axis=0)
        mean_gradients_abs = np.mean(all_gradients_abs, axis=0)
        dimension_size = len(mean_gradients_signed)
        
        value_type = "value" if aggregate_by == 'values' else ("Hessian diag (sum)" if aggregate_by == 'hessian_diagonal_sum' else ("Hessian diag" if aggregate_by == 'hessian_diagonal' else "gradient"))
        print(f"Layer {layer_idx} (before power/normalization):")
        print(f"  Mean {value_type} (signed) = {mean_gradients_signed.mean():.6f}, "
              f"Max = {np.abs(mean_gradients_signed).max():.6f}, Std = {mean_gradients_signed.std():.6f}")
        print(f"  Mean {value_type} (abs) = {mean_gradients_abs.mean():.6f}, "
              f"Max = {mean_gradients_abs.max():.6f}, Std = {mean_gradients_abs.std():.6f}")
        
        # Apply power transformation if specified
        if power is not None:
            mean_gradients_signed = np.power(np.abs(mean_gradients_signed), power) * np.sign(mean_gradients_signed)
            mean_gradients_abs = np.power(mean_gradients_abs, power)
            print(f"Layer {layer_idx} (after power={power}):")
            print(f"  Mean {value_type} (signed): Mean = {mean_gradients_signed.mean():.6f}, "
                  f"Max = {np.abs(mean_gradients_signed).max():.6f}")
            print(f"  Mean {value_type} (abs): Mean = {mean_gradients_abs.mean():.6f}, "
                  f"Max = {mean_gradients_abs.max():.6f}")
        
        if normalize == 'none':
            return {'mean': mean_gradients_signed, 'mean_abs': mean_gradients_abs}
        elif normalize == 'both':
            # Apply both normalizations to both aggregation modes
            result = {
                'mean': {
                    'norm_sum': normalize_values(mean_gradients_signed, dimension_size, mode='sum'),
                    'norm_sum_abs': normalize_values(mean_gradients_signed, dimension_size, mode='sum_abs')
                },
                'mean_abs': {
                    'norm_sum': normalize_values(mean_gradients_abs, dimension_size, mode='sum'),
                    'norm_sum_abs': normalize_values(mean_gradients_abs, dimension_size, mode='sum_abs')
                }
            }
            print(f"Layer {layer_idx} (after both normalizations):")
            print(f"  Mean (norm_sum): Sum = {np.sum(result['mean']['norm_sum']):.1f}")
            print(f"  Mean (norm_sum_abs): Sum of abs = {np.sum(np.abs(result['mean']['norm_sum_abs'])):.1f}")
            print(f"  Mean_abs (norm_sum): Sum = {np.sum(result['mean_abs']['norm_sum']):.1f}")
            print(f"  Mean_abs (norm_sum_abs): Sum of abs = {np.sum(np.abs(result['mean_abs']['norm_sum_abs'])):.1f}")
            return result
        elif normalize == 'all':
            # Apply all three normalizations to both aggregation modes
            result = {
                'mean': {
                    'none': mean_gradients_signed.copy(),
                    'norm_sum': normalize_values(mean_gradients_signed, dimension_size, mode='sum'),
                    'norm_sum_abs': normalize_values(mean_gradients_signed, dimension_size, mode='sum_abs')
                },
                'mean_abs': {
                    'none': mean_gradients_abs.copy(),
                    'norm_sum': normalize_values(mean_gradients_abs, dimension_size, mode='sum'),
                    'norm_sum_abs': normalize_values(mean_gradients_abs, dimension_size, mode='sum_abs')
                }
            }
            print(f"Layer {layer_idx} (after all normalizations):")
            print(f"  Mean (none): No normalization applied")
            print(f"  Mean (norm_sum): Sum = {np.sum(result['mean']['norm_sum']):.1f}")
            print(f"  Mean (norm_sum_abs): Sum of abs = {np.sum(np.abs(result['mean']['norm_sum_abs'])):.1f}")
            print(f"  Mean_abs (none): No normalization applied")
            print(f"  Mean_abs (norm_sum): Sum = {np.sum(result['mean_abs']['norm_sum']):.1f}")
            print(f"  Mean_abs (norm_sum_abs): Sum of abs = {np.sum(np.abs(result['mean_abs']['norm_sum_abs'])):.1f}")
            return result
        else:
            mean_gradients_signed = normalize_values(mean_gradients_signed, dimension_size, mode=normalize)
            mean_gradients_abs = normalize_values(mean_gradients_abs, dimension_size, mode=normalize)
            print(f"Layer {layer_idx} (after {normalize}):")
            if normalize == 'sum_abs':
                print(f"  Mean {value_type} (signed): Sum of abs = {np.sum(np.abs(mean_gradients_signed)):.1f} (target: {dimension_size})")
                print(f"  Mean {value_type} (abs): Sum of abs = {np.sum(np.abs(mean_gradients_abs)):.1f} (target: {dimension_size})")
            elif normalize == 'sum':
                print(f"  Mean {value_type} (signed): Sum = {np.sum(mean_gradients_signed):.1f} (target: {dimension_size})")
                print(f"  Mean {value_type} (abs): Sum = {np.sum(mean_gradients_abs):.1f} (target: {dimension_size})")
            return {'mean': mean_gradients_signed, 'mean_abs': mean_gradients_abs}


def create_histogram(gradients: np.ndarray, layer_idx: int, output_dir: str, mode_suffix: str = "", aggregate_by: str = "gradients"):
    """Create and save histogram plot for gradient values.
    
    Args:
        gradients: Array of aggregated gradients
        layer_idx: Layer index
        output_dir: Directory to save plot
        mode_suffix: Suffix for filename (e.g., "_mean" or "_mean_abs")
        aggregate_by: What was aggregated ('gradients', 'values', 'hessian_diagonal_sum', or 'hessian_diagonal')
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
    
    # Replace 'none' with 'norm_none' in mode_suffix for filenames
    filename_suffix = mode_suffix.replace('_none', '_norm_none') if '_none' in mode_suffix else mode_suffix
    
    # Save plot with aggregate_by in filename
    output_path = os.path.join(output_dir, f'layer_{layer_idx}{filename_suffix}_{aggregate_by}_hist.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved histogram to {output_path}")


def create_feature_scatter_plot(gradients: np.ndarray, layer_idx: int, output_dir: str, mode_suffix: str = "", aggregate_by: str = "gradients"):
    """Create and save scatter plot showing gradient per activation dimension.
    
    Args:
        gradients: Array of aggregated gradients (one per activation dimension)
        layer_idx: Layer index
        output_dir: Directory to save plot
        mode_suffix: Suffix for filename (e.g., "_mean" or "_mean_abs")
        aggregate_by: What was aggregated ('gradients', 'values', 'hessian_diagonal_sum', or 'hessian_diagonal')
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
    
    # Replace 'none' with 'norm_none' in mode_suffix for filenames
    filename_suffix = mode_suffix.replace('_none', '_norm_none') if '_none' in mode_suffix else mode_suffix
    
    # Save plot with aggregate_by in filename
    output_path = os.path.join(output_dir, f'layer_{layer_idx}{filename_suffix}_{aggregate_by}_scatter.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved activation scatter plot to {output_path}")


def create_combined_single_layer_plot(gradients_dict: Dict[str, np.ndarray], layer_idx: int, 
                                     output_dir: str, plot_type: str = 'histogram', aggregate_by: str = "gradients"):
    """Create combined plot showing both mean and mean_abs side by side for a single layer.
    
    Args:
        gradients_dict: Dict with 'mean' and 'mean_abs' gradient arrays
        layer_idx: Layer index
        output_dir: Directory to save plot
        plot_type: 'histogram' or 'scatter'
        aggregate_by: What was aggregated ('gradients', 'values', 'hessian_diagonal_sum', or 'hessian_diagonal')
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
    output_path = os.path.join(output_dir, f'layer_{layer_idx}_{aggregate_by}_{suffix}_combined.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved combined {plot_type} plot to {output_path}")


def create_combined_histograms(all_gradients: Dict[int, np.ndarray], output_dir: str, mode_suffix: str = "", aggregate_by: str = "gradients"):
    """Create combined plot showing histograms for all layers.
    
    Args:
        all_gradients: Dictionary mapping layer_idx to gradient arrays
        output_dir: Directory to save plot
        mode_suffix: Suffix for filename (e.g., "_mean" or "_mean_abs")
        aggregate_by: What was aggregated ('gradients', 'values', 'hessian_diagonal_sum', or 'hessian_diagonal')
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
    
    # Replace 'none' with 'norm_none' in mode_suffix for filenames
    filename_suffix = mode_suffix.replace('_none', '_norm_none') if '_none' in mode_suffix else mode_suffix
    
    # Save plot with aggregate_by in filename
    output_path = os.path.join(output_dir, f'all_layers{filename_suffix}_{aggregate_by}_histograms.png')
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"\nSaved combined histogram plot to {output_path}")


def create_combined_scatter_plots(all_gradients: Dict[int, np.ndarray], output_dir: str, mode_suffix: str = "", aggregate_by: str = "gradients"):
    """Create combined plot showing scatter plots for all layers.
    
    Args:
        all_gradients: Dictionary mapping layer_idx to gradient arrays
        output_dir: Directory to save plot
        mode_suffix: Suffix for filename (e.g., "_mean" or "_mean_abs")
        aggregate_by: What was aggregated ('gradients', 'values', 'hessian_diagonal_sum', or 'hessian_diagonal')
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
    
    # Replace 'none' with 'norm_none' in mode_suffix for filenames
    filename_suffix = mode_suffix.replace('_none', '_norm_none') if '_none' in mode_suffix else mode_suffix
    
    # Save plot with aggregate_by in filename
    output_path = os.path.join(output_dir, f'all_layers{filename_suffix}_{aggregate_by}_scatter.png')
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
    
    if aggregation_mode == 'both' and args.normalize_mode == 'both':
        # Both aggregation AND normalization
        for agg_mode in ['mean', 'mean_abs']:
            for norm_mode in ['norm_sum', 'norm_sum_abs']:
                for layer_idx in sorted(all_results[agg_mode][norm_mode].keys()):
                    key = f'layer_{layer_idx}_{agg_mode}_{norm_mode}'
                    save_dict[key] = all_results[agg_mode][norm_mode][layer_idx]
    elif aggregation_mode == 'both' and args.normalize_mode == 'all':
        # Both aggregation AND all normalizations (6 variants)
        for agg_mode in ['mean', 'mean_abs']:
            for norm_mode in ['none', 'norm_sum', 'norm_sum_abs']:
                for layer_idx in sorted(all_results[agg_mode][norm_mode].keys()):
                    key = f'layer_{layer_idx}_{agg_mode}_{norm_mode}'
                    save_dict[key] = all_results[agg_mode][norm_mode][layer_idx]
    elif aggregation_mode == 'both':
        # Both aggregation modes, single normalization
        for mode in ['mean', 'mean_abs']:
            for layer_idx in sorted(all_results[mode].keys()):
                key = f'layer_{layer_idx}_{mode}'
                save_dict[key] = all_results[mode][layer_idx]
    elif args.normalize_mode == 'both':
        # Single aggregation, both normalizations
        for norm_mode in ['norm_sum', 'norm_sum_abs']:
            for layer_idx in sorted(all_results[norm_mode].keys()):
                key = f'layer_{layer_idx}_{aggregation_mode}_{norm_mode}'
                save_dict[key] = all_results[norm_mode][layer_idx]
    elif args.normalize_mode == 'all':
        # Single aggregation, all normalizations (3 variants)
        for norm_mode in ['none', 'norm_sum', 'norm_sum_abs']:
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
        'normalization_mode': args.normalize_mode,
        'aggregate_by': args.aggregate_by,
        'power': args.power,
        'device': args.device,
        'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
    }
    
    # Add dimension info per layer
    if aggregation_mode == 'both' and args.normalize_mode in ['both', 'all']:
        # Use first available normalization mode
        first_norm = 'norm_sum' if 'norm_sum' in all_results['mean'] else list(all_results['mean'].keys())[0]
        dimension_info = {layer_idx: len(all_results['mean'][first_norm][layer_idx]) 
                         for layer_idx in all_results['mean'][first_norm].keys()}
    elif aggregation_mode == 'both':
        dimension_info = {layer_idx: len(all_results['mean'][layer_idx]) 
                         for layer_idx in all_results['mean'].keys()}
    elif args.normalize_mode in ['both', 'all']:
        # Use first available normalization mode
        first_norm = 'norm_sum' if 'norm_sum' in all_results else list(all_results.keys())[0]
        dimension_info = {layer_idx: len(all_results[first_norm][layer_idx]) 
                         for layer_idx in all_results[first_norm].keys()}
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
            'normalization_mode': args.normalize_mode,
            'aggregate_by': args.aggregate_by,
            'power': args.power,
        },
        'timing_stats': timing_stats,
        'per_layer_timing': layer_timings,
        'layer_statistics': {},
        'dimension_info': {}
    }
    
    # Add per-layer statistics and dimension info
    if aggregation_mode == 'both' and args.normalize_mode == 'both':
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
    elif aggregation_mode == 'both' and args.normalize_mode == 'all':
        for layer_idx in all_results['mean']['none'].keys():
            dimension_size = len(all_results['mean']['none'][layer_idx])
            metadata['dimension_info'][f'layer_{layer_idx}'] = dimension_size
            metadata['layer_statistics'][f'layer_{layer_idx}'] = {}
            for agg_mode in ['mean', 'mean_abs']:
                metadata['layer_statistics'][f'layer_{layer_idx}'][agg_mode] = {}
                for norm_mode in ['none', 'norm_sum', 'norm_sum_abs']:
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
    elif args.normalize_mode == 'both':
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
    elif args.normalize_mode == 'all':
        for layer_idx in all_results['none'].keys():
            dimension_size = len(all_results['none'][layer_idx])
            metadata['dimension_info'][f'layer_{layer_idx}'] = dimension_size
            metadata['layer_statistics'][f'layer_{layer_idx}'] = {}
            for norm_mode in ['none', 'norm_sum', 'norm_sum_abs']:
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
        'individual_npy': 'Use np.load(layer_X_mode_aggregate_by.npy) for individual layers',
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
        
        # Find all .npy files (supports both old '_gradients.npy' and new naming)
        gradients = {}
        for file in os.listdir(path):
            if file.endswith('_gradients.npy') or file.endswith('_values.npy'):
                # Remove the file extension and store
                key = file.replace('_gradients.npy', '').replace('_values.npy', '')
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
    
    # Print power info if multiple values were provided
    if args.power_list is not None:
        print(f"Multiple power values provided: {args.power_list}")
        print(f"Using first power value: {args.power}")
    
    # Get model configuration
    model_config = get_model_config(args.model_name)
    print(f"Model type detected: {model_config['model_type']}")
    
    # Parse layer range
    layers = parse_layer_range(args.layers)
    print(f"Analyzing layers: {layers}")
    print(f"Aggregation mode: {args.aggregation_mode}")
    
    # Determine output directory
    output_dir: Optional[str] = None
    tee_output = None
    original_stdout = sys.stdout
    
    if args.save_outputs:
        # Generate directory name from parameters
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_safe = args.model_name.replace("/", "_")
        layers_safe = args.layers.replace("-", "to").replace(",", "_")
        power_str = f"_power_{args.power}" if args.power is not None else f"_power_None"
        output_dir = f"layer_activations_model_{model_safe}_num_samples_{args.num_samples}_batch_size_{args.batch_size}_layers_{layers_safe}_agg_{args.aggregate_by}{power_str}_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        print(f"\nOutputs will be saved to: {output_dir}")
        
        # Set up output logging to file
        runtime_log_file = os.path.join(output_dir, 'runtime_output.txt')
        tee_output = TeeOutput(runtime_log_file)
        sys.stdout = tee_output
        print(f"Runtime output will be saved to: {runtime_log_file}")
    else:
        print("\nOutputs will NOT be saved (use --save_outputs to enable)")
        output_dir = None
    
    # Force eager attention for Hessian computation (must be done BEFORE model loading)
    # Eager attention supports second derivatives, unlike flash/memory-efficient attention
    use_eager_attention = (args.aggregate_by in ['hessian_diagonal_sum', 'hessian_diagonal'])
    
    if use_eager_attention:
        print("Forcing eager attention implementation for Hessian computation")
        print("  (eager attention supports second derivatives, unlike efficient attention)")
        
        # Also disable SDP kernels as backup (though model config should be sufficient)
        original_sdp_settings = None
        try:
            # Try to disable efficient kernels (old API - deprecated but may still work)
            original_sdp_settings = {
                'enable_flash': torch.backends.cuda.sdp_kernel.enable_flash,
                'enable_math': torch.backends.cuda.sdp_kernel.enable_math,
                'enable_mem_efficient': torch.backends.cuda.sdp_kernel.enable_mem_efficient
            }
            torch.backends.cuda.sdp_kernel(
                enable_flash=False,
                enable_math=True,
                enable_mem_efficient=False
            )
        except (AttributeError, TypeError):
            # Old API not available or deprecated - that's okay, model config will handle it
            original_sdp_settings = None
    
    # Load model and tokenizer
    print("\n" + "="*60)
    print("TIMING: Loading Model and Tokenizer")
    print("="*60)
    start_time = time.time()
    model, tokenizer = load_model_and_tokenizer(args.model_name, args.device, use_eager_attention=use_eager_attention)
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
    if args.aggregation_mode == 'both' and args.normalize_mode == 'both':
        all_results = {
            'mean': {'norm_sum': {}, 'norm_sum_abs': {}},
            'mean_abs': {'norm_sum': {}, 'norm_sum_abs': {}}
        }
    elif args.aggregation_mode == 'both' and args.normalize_mode == 'all':
        all_results = {
            'mean': {'none': {}, 'norm_sum': {}, 'norm_sum_abs': {}},
            'mean_abs': {'none': {}, 'norm_sum': {}, 'norm_sum_abs': {}}
        }
    elif args.aggregation_mode == 'both':
        all_results = {'mean': {}, 'mean_abs': {}}
    elif args.normalize_mode == 'both':
        all_results = {'norm_sum': {}, 'norm_sum_abs': {}}
    elif args.normalize_mode == 'all':
        all_results = {'none': {}, 'norm_sum': {}, 'norm_sum_abs': {}}
    else:
        all_results = {}
    layer_timings = {}
    
    print("\n" + "="*60)
    print("TIMING: Processing Layers")
    print("="*60)
    
    for layer_idx in layers:
        layer_start_time = time.time()
        layer_timing = {}
        
        # Compute aggregated values (gradients or activations)
        grad_start = time.time()
        gradients = compute_layer_aggregates(
            model=model,
            tokenizer=tokenizer,
            dataloader=dataloader,
            layer_idx=layer_idx,
            device=args.device,
            aggregation_mode=args.aggregation_mode,
            model_config=model_config,
            normalize=args.normalize_mode,
            aggregate_by=args.aggregate_by,
            power=args.power
        )
        layer_timing['gradient_computation'] = time.time() - grad_start
        
        if gradients is not None:
            # Handle different combinations of aggregation and normalization modes
            if args.aggregation_mode == 'both' and args.normalize_mode == 'both':
                # Both aggregation AND normalization = 4 variants
                for agg_mode in ['mean', 'mean_abs']:
                    for norm_mode in ['norm_sum', 'norm_sum_abs']:
                        all_results[agg_mode][norm_mode][layer_idx] = gradients[agg_mode][norm_mode]
                        
                        if output_dir is not None:
                            # Save each variant
                            # Replace 'none' with 'norm_none' in filename
                            norm_mode_filename = 'norm_none' if norm_mode == 'none' else norm_mode
                            filename = f'layer_{layer_idx}_{agg_mode}_{norm_mode_filename}_{args.aggregate_by}.npy'
                            np.save(os.path.join(output_dir, filename), gradients[agg_mode][norm_mode])
                            
                            # Create plots for each variant
                            plot_start = time.time()
                            mode_suffix = f'_{agg_mode}_{norm_mode}'
                            create_histogram(gradients[agg_mode][norm_mode], layer_idx, output_dir, mode_suffix, aggregate_by=args.aggregate_by)
                            create_feature_scatter_plot(gradients[agg_mode][norm_mode], layer_idx, output_dir, mode_suffix, aggregate_by=args.aggregate_by)
                            layer_timing[f'{agg_mode}_{norm_mode}_plots'] = time.time() - plot_start
            
            elif args.aggregation_mode == 'both' and args.normalize_mode == 'all':
                # Both aggregation AND all normalizations = 6 variants
                for agg_mode in ['mean', 'mean_abs']:
                    for norm_mode in ['none', 'norm_sum', 'norm_sum_abs']:
                        all_results[agg_mode][norm_mode][layer_idx] = gradients[agg_mode][norm_mode]
                        
                        if output_dir is not None:
                            # Save each variant
                            # Replace 'none' with 'norm_none' in filename
                            norm_mode_filename = 'norm_none' if norm_mode == 'none' else norm_mode
                            filename = f'layer_{layer_idx}_{agg_mode}_{norm_mode_filename}_{args.aggregate_by}.npy'
                            np.save(os.path.join(output_dir, filename), gradients[agg_mode][norm_mode])
                            
                            # Create plots for each variant
                            plot_start = time.time()
                            mode_suffix = f'_{agg_mode}_{norm_mode}'
                            create_histogram(gradients[agg_mode][norm_mode], layer_idx, output_dir, mode_suffix, aggregate_by=args.aggregate_by)
                            create_feature_scatter_plot(gradients[agg_mode][norm_mode], layer_idx, output_dir, mode_suffix, aggregate_by=args.aggregate_by)
                            layer_timing[f'{agg_mode}_{norm_mode}_plots'] = time.time() - plot_start
            
            elif args.aggregation_mode == 'both':
                # Both aggregation modes, single normalization
                for agg_mode in ['mean', 'mean_abs']:
                    all_results[agg_mode][layer_idx] = gradients[agg_mode]
                
                if output_dir is not None:
                    # Create separate plots for each mode
                    for mode, mode_suffix in [('mean', '_mean'), ('mean_abs', '_mean_abs')]:
                        plot_start = time.time()
                        create_histogram(gradients[mode], layer_idx, output_dir, mode_suffix, aggregate_by=args.aggregate_by)
                        create_feature_scatter_plot(gradients[mode], layer_idx, output_dir, mode_suffix, aggregate_by=args.aggregate_by)
                        layer_timing[f'{mode}_plots'] = time.time() - plot_start
                        
                        # Save raw gradients/values
                        np.save(
                            os.path.join(output_dir, f'layer_{layer_idx}{mode_suffix}_{args.aggregate_by}.npy'),
                            gradients[mode]
                        )
                    
                    # Create combined subplot figures
                    combined_start = time.time()
                    create_combined_single_layer_plot(gradients, layer_idx, output_dir, 'histogram', aggregate_by=args.aggregate_by)
                    create_combined_single_layer_plot(gradients, layer_idx, output_dir, 'scatter', aggregate_by=args.aggregate_by)
                    layer_timing['combined_plots'] = time.time() - combined_start
            
            elif args.normalize_mode == 'both':
                # Single aggregation, both normalizations
                for norm_mode in ['norm_sum', 'norm_sum_abs']:
                    all_results[norm_mode][layer_idx] = gradients[norm_mode]
                
                if output_dir is not None:
                    for norm_mode in ['norm_sum', 'norm_sum_abs']:
                        plot_start = time.time()
                        mode_suffix = f'_{args.aggregation_mode}_{norm_mode}'
                        create_histogram(gradients[norm_mode], layer_idx, output_dir, mode_suffix, aggregate_by=args.aggregate_by)
                        create_feature_scatter_plot(gradients[norm_mode], layer_idx, output_dir, mode_suffix, aggregate_by=args.aggregate_by)
                        layer_timing[f'{norm_mode}_plots'] = time.time() - plot_start
                        
                        # Save raw gradients/values
                        # Replace 'none' with 'norm_none' in filename
                        norm_mode_filename = 'norm_none' if norm_mode == 'none' else norm_mode
                        filename = f'layer_{layer_idx}_{args.aggregation_mode}_{norm_mode_filename}_{args.aggregate_by}.npy'
                        np.save(os.path.join(output_dir, filename), gradients[norm_mode])
            
            elif args.normalize_mode == 'all':
                # Single aggregation, all normalizations
                for norm_mode in ['none', 'norm_sum', 'norm_sum_abs']:
                    all_results[norm_mode][layer_idx] = gradients[norm_mode]
                
                if output_dir is not None:
                    for norm_mode in ['none', 'norm_sum', 'norm_sum_abs']:
                        plot_start = time.time()
                        mode_suffix = f'_{args.aggregation_mode}_{norm_mode}'
                        create_histogram(gradients[norm_mode], layer_idx, output_dir, mode_suffix, aggregate_by=args.aggregate_by)
                        create_feature_scatter_plot(gradients[norm_mode], layer_idx, output_dir, mode_suffix, aggregate_by=args.aggregate_by)
                        layer_timing[f'{norm_mode}_plots'] = time.time() - plot_start
                        
                        # Save raw gradients/values
                        # Replace 'none' with 'norm_none' in filename
                        norm_mode_filename = 'norm_none' if norm_mode == 'none' else norm_mode
                        filename = f'layer_{layer_idx}_{args.aggregation_mode}_{norm_mode_filename}_{args.aggregate_by}.npy'
                        np.save(os.path.join(output_dir, filename), gradients[norm_mode])
            
            else:
                # Single aggregation, single normalization
                all_results[layer_idx] = gradients
                mode_suffix = f"_{args.aggregation_mode}"
                
                if output_dir is not None:
                    # Create plots
                    plot_start = time.time()
                    create_histogram(gradients, layer_idx, output_dir, mode_suffix, aggregate_by=args.aggregate_by)
                    create_feature_scatter_plot(gradients, layer_idx, output_dir, mode_suffix, aggregate_by=args.aggregate_by)
                    layer_timing['plots'] = time.time() - plot_start
                    
                    # Save raw gradients/values
                    save_start = time.time()
                    np.save(
                        os.path.join(output_dir, f'layer_{layer_idx}{mode_suffix}_{args.aggregate_by}.npy'),
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
        if args.aggregation_mode == 'both' and args.normalize_mode in ['both', 'all']:
            # Get first available normalization mode to check layer count
            first_norm = 'norm_sum' if 'norm_sum' in all_results['mean'] else list(all_results['mean'].keys())[0]
            num_layers_check = len(all_results['mean'][first_norm])
            if num_layers_check > 1:
                print("\n" + "="*60)
                print("Creating combined plots for all layers...")
                print("="*60)
                
                combined_start = time.time()
                for agg_mode in ['mean', 'mean_abs']:
                    # Get all normalization modes available for this aggregation mode
                    norm_modes = list(all_results[agg_mode].keys())
                    for norm_mode in norm_modes:
                        mode_suffix = f'_{agg_mode}_{norm_mode}'
                        create_combined_histograms(all_results[agg_mode][norm_mode], output_dir, mode_suffix, aggregate_by=args.aggregate_by)
                        create_combined_scatter_plots(all_results[agg_mode][norm_mode], output_dir, mode_suffix, aggregate_by=args.aggregate_by)
                combined_time = time.time() - combined_start
                
                print(f"Combined plots created in {combined_time:.2f} seconds")
        elif args.aggregation_mode == 'both':
            if len(all_results['mean']) > 1:
                print("\n" + "="*60)
                print("Creating combined plots for all layers...")
                print("="*60)
                
                combined_start = time.time()
                for mode, mode_suffix in [('mean', '_mean'), ('mean_abs', '_mean_abs')]:
                    create_combined_histograms(all_results[mode], output_dir, mode_suffix, aggregate_by=args.aggregate_by)
                    create_combined_scatter_plots(all_results[mode], output_dir, mode_suffix, aggregate_by=args.aggregate_by)
                combined_time = time.time() - combined_start
                
                print(f"Combined plots created in {combined_time:.2f} seconds")
        elif args.normalize_mode in ['both', 'all']:
            # Get first available normalization mode to check layer count
            first_norm = list(all_results.keys())[0]
            if len(all_results[first_norm]) > 1:
                print("\n" + "="*60)
                print("Creating combined plots for all layers...")
                print("="*60)
                
                combined_start = time.time()
                # Get all normalization modes
                norm_modes = list(all_results.keys())
                for norm_mode in norm_modes:
                    mode_suffix = f'_{args.aggregation_mode}_{norm_mode}'
                    create_combined_histograms(all_results[norm_mode], output_dir, mode_suffix, aggregate_by=args.aggregate_by)
                    create_combined_scatter_plots(all_results[norm_mode], output_dir, mode_suffix, aggregate_by=args.aggregate_by)
                combined_time = time.time() - combined_start
                
                print(f"Combined plots created in {combined_time:.2f} seconds")
        else:
            if len(all_results) > 1:
                print("\n" + "="*60)
                print("Creating combined plots for all layers...")
                print("="*60)
                
                combined_start = time.time()
                mode_suffix = f"_{args.aggregation_mode}"
                create_combined_histograms(all_results, output_dir, mode_suffix, aggregate_by=args.aggregate_by)
                create_combined_scatter_plots(all_results, output_dir, mode_suffix, aggregate_by=args.aggregate_by)
                combined_time = time.time() - combined_start
                
                print(f"Combined plots created in {combined_time:.2f} seconds")
        
        # Calculate number of layers processed
        if args.aggregation_mode == 'both' and args.normalize_mode in ['both', 'all']:
            first_norm = list(all_results['mean'].keys())[0]
            num_layers_processed = len(all_results['mean'][first_norm])
        elif args.aggregation_mode == 'both':
            num_layers_processed = len(all_results['mean'])
        elif args.normalize_mode in ['both', 'all']:
            first_norm = list(all_results.keys())[0]
            num_layers_processed = len(all_results[first_norm])
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
        if args.aggregation_mode == 'both' and args.normalize_mode in ['both', 'all']:
            first_norm = list(all_results['mean'].keys())[0]
            num_layers_processed = len(all_results['mean'][first_norm])
        elif args.aggregation_mode == 'both':
            num_layers_processed = len(all_results['mean'])
        elif args.normalize_mode in ['both', 'all']:
            first_norm = list(all_results.keys())[0]
            num_layers_processed = len(all_results[first_norm])
        else:
            num_layers_processed = len(all_results)
        print(f"\nCompleted analysis for {num_layers_processed} layers")
        print("Results were NOT saved (use --save_outputs to enable saving)")
    
    # Restore SDP kernel settings if they were changed
    if args.aggregate_by in ['hessian_diagonal_sum', 'hessian_diagonal'] and original_sdp_settings is not None:
        try:
            torch.backends.cuda.sdp_kernel(**original_sdp_settings)
            print("Restored original SDP kernel settings")
        except (AttributeError, TypeError):
            pass
    
    # Cleanup: restore stdout and close log file
    if tee_output is not None:
        sys.stdout = original_stdout
        tee_output.close()
        print(f"\nRuntime output saved to: {os.path.join(output_dir, 'runtime_output.txt')}")


if __name__ == "__main__":
    main()

