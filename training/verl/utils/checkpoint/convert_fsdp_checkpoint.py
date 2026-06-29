#!/usr/bin/env python3
"""
Convert FSDP checkpoint from one world_size to another.

This script converts FSDP checkpoints by:
1. Loading all sharded checkpoints from source world_size
2. Merging them into full state dicts
3. Resharding to target world_size
4. Saving new checkpoint files

Usage:
    python convert_fsdp_checkpoint.py \
        --source_ckpt_dir /path/to/source/checkpoint \
        --target_ckpt_dir /path/to/target/checkpoint \
        --source_world_size 8 \
        --target_world_size 16
"""

import argparse
import json
import os
import shutil
from collections import OrderedDict
from pathlib import Path

import torch


def load_sharded_state_dict(sharded_dict: dict, source_world_size: int) -> dict:
    """
    Merge sharded FSDP state dict into full state dict.
    
    FSDP sharded state dicts contain keys like:
    - _fsdp_wrapped_module.{module_name}.{param_name} with flat_param_X tensors
    
    This function reconstructs the full state dict by concatenating flat_params.
    """
    full_state_dict = OrderedDict()
    
    # Group parameters by their base name (without rank suffix)
    param_groups = {}
    
    for key, value in sharded_dict.items():
        if isinstance(value, torch.Tensor):
            # For flat_params, we need to concatenate them
            if "flat_param" in key:
                base_key = key.rsplit(".", 1)[0]  # Remove .flat_param_X
                if base_key not in param_groups:
                    param_groups[base_key] = []
                param_groups[base_key].append((key, value))
            else:
                # For non-flat params, we can use any rank's copy (they should be the same)
                if key not in full_state_dict:
                    full_state_dict[key] = value.clone()
        else:
            # Non-tensor values (like metadata)
            if key not in full_state_dict:
                full_state_dict[key] = value
    
    # Concatenate flat_params
    for base_key, param_list in param_groups.items():
        # Sort by rank (extract from key if possible, or use order)
        param_list.sort(key=lambda x: x[0])
        # Concatenate all shards
        full_tensor = torch.cat([v for _, v in param_list], dim=0)
        # Find the original parameter name
        # For FSDP, flat_params are concatenated parameters, we need to split them
        # This is complex, so we'll keep them as flat_params for now
        full_state_dict[base_key] = full_tensor
    
    return full_state_dict


def reshard_state_dict(full_state_dict: dict, target_world_size: int, target_rank: int) -> dict:
    """
    Reshard a full state dict for a specific rank in target world_size.
    
    Args:
        full_state_dict: Full (merged) state dict
        target_world_size: Target number of GPUs
        target_rank: Rank in target world_size
    
    Returns:
        Sharded state dict for target_rank
    """
    sharded_dict = OrderedDict()
    
    for key, value in full_state_dict.items():
        if isinstance(value, torch.Tensor):
            # Calculate shard size
            total_size = value.numel()
            shard_size = (total_size + target_world_size - 1) // target_world_size
            start_idx = target_rank * shard_size
            end_idx = min(start_idx + shard_size, total_size)
            
            if start_idx < total_size:
                # Flatten, shard, then reshape
                flat_value = value.flatten()
                shard = flat_value[start_idx:end_idx]
                # Try to reshape to original shape if possible
                # For simplicity, we keep it flat (FSDP will handle reshaping)
                sharded_dict[key] = shard
            else:
                # Empty shard
                sharded_dict[key] = torch.tensor([], dtype=value.dtype, device=value.device)
        else:
            # Non-tensor values are copied as-is
            sharded_dict[key] = value
    
    return sharded_dict


def convert_checkpoint(
    source_ckpt_dir: str,
    target_ckpt_dir: str,
    source_world_size: int,
    target_world_size: int,
    component: str = "actor",
):
    """
    Convert FSDP checkpoint from source_world_size to target_world_size.
    
    Args:
        source_ckpt_dir: Directory containing source checkpoint
        target_ckpt_dir: Directory to save converted checkpoint
        source_world_size: Original number of GPUs
        target_world_size: Target number of GPUs
        component: Checkpoint component name (e.g., 'actor', 'critic')
    """
    source_dir = Path(source_ckpt_dir) / component
    target_dir = Path(target_ckpt_dir) / component
    
    if not source_dir.exists():
        raise ValueError(f"Source checkpoint directory not found: {source_dir}")
    
    # Create target directory
    target_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Converting checkpoint from {source_world_size} GPUs to {target_world_size} GPUs")
    print(f"Source: {source_dir}")
    print(f"Target: {target_dir}")
    
    # Load all source checkpoints
    print("\nLoading source checkpoints...")
    model_shards = []
    optim_shards = []
    extra_states = []
    
    for rank in range(source_world_size):
        model_path = source_dir / f"model_world_size_{source_world_size}_rank_{rank}.pt"
        optim_path = source_dir / f"optim_world_size_{source_world_size}_rank_{rank}.pt"
        extra_path = source_dir / f"extra_state_world_size_{source_world_size}_rank_{rank}.pt"
        
        if model_path.exists():
            model_shards.append(torch.load(model_path, weights_only=False, map_location="cpu"))
            print(f"  Loaded model shard {rank}")
        
        if optim_path.exists():
            optim_shards.append(torch.load(optim_path, weights_only=False, map_location="cpu"))
            print(f"  Loaded optim shard {rank}")
        
        if extra_path.exists():
            extra_states.append(torch.load(extra_path, weights_only=False, map_location="cpu"))
            print(f"  Loaded extra_state shard {rank}")
    
    # Merge model state dicts
    print("\nMerging model state dicts...")
    full_model_state = OrderedDict()
    
    # For FSDP sharded checkpoints, we need to handle flat_params specially
    # The simplest approach is to use the first rank's structure and concatenate flat_params
    if model_shards:
        # Collect all unique keys
        all_keys = set()
        for shard in model_shards:
            all_keys.update(shard.keys())
        
        # For each key, merge if it's a tensor
        for key in sorted(all_keys):
            values = [shard.get(key) for shard in model_shards if key in shard]
            if not values:
                continue
            
            # Check if all values are tensors
            if all(isinstance(v, torch.Tensor) for v in values):
                # Check if this is a flat_param (needs concatenation)
                if "flat_param" in key or any(v.numel() > 0 for v in values):
                    # Try to concatenate along first dimension
                    try:
                        # Filter out empty tensors
                        non_empty = [v for v in values if v.numel() > 0]
                        if non_empty:
                            # Concatenate all shards
                            full_model_state[key] = torch.cat(non_empty, dim=0)
                        else:
                            # Use first value if all are empty
                            full_model_state[key] = values[0]
                    except Exception as e:
                        print(f"  Warning: Could not concatenate {key}: {e}, using first shard")
                        full_model_state[key] = values[0]
                else:
                    # Use first non-empty value
                    full_model_state[key] = next(v for v in values if v.numel() > 0) if any(v.numel() > 0 for v in values) else values[0]
            else:
                # Non-tensor: use first rank's value
                full_model_state[key] = values[0]
    
    # Merge optimizer state dicts (similar approach)
    print("\nMerging optimizer state dicts...")
    full_optim_state = OrderedDict()
    
    if optim_shards:
        all_keys = set()
        for shard in optim_shards:
            all_keys.update(shard.keys())
        
        for key in sorted(all_keys):
            values = [shard.get(key) for shard in optim_shards if key in shard]
            if not values:
                continue
            
            if all(isinstance(v, torch.Tensor) for v in values):
                non_empty = [v for v in values if v.numel() > 0]
                if non_empty:
                    try:
                        full_optim_state[key] = torch.cat(non_empty, dim=0)
                    except:
                        full_optim_state[key] = values[0]
                else:
                    full_optim_state[key] = values[0]
            else:
                full_optim_state[key] = values[0]
    
    # For extra_state, we can use rank 0's (they should be the same for scheduler/RNG)
    print("\nProcessing extra states...")
    if extra_states:
        # Use rank 0's extra_state as template
        full_extra_state = extra_states[0].copy()
    else:
        full_extra_state = {}
    
    # Reshard to target world_size
    print(f"\nResharding to {target_world_size} GPUs...")
    for target_rank in range(target_world_size):
        print(f"  Processing rank {target_rank}...")
        
        # Reshard model
        sharded_model = reshard_state_dict(full_model_state, target_world_size, target_rank)
        model_path = target_dir / f"model_world_size_{target_world_size}_rank_{target_rank}.pt"
        torch.save(sharded_model, model_path)
        print(f"    Saved model shard to {model_path}")
        
        # Reshard optimizer
        sharded_optim = reshard_state_dict(full_optim_state, target_world_size, target_rank)
        optim_path = target_dir / f"optim_world_size_{target_world_size}_rank_{target_rank}.pt"
        torch.save(sharded_optim, optim_path)
        print(f"    Saved optim shard to {optim_path}")
        
        # Copy extra_state (same for all ranks)
        extra_path = target_dir / f"extra_state_world_size_{target_world_size}_rank_{target_rank}.pt"
        torch.save(full_extra_state, extra_path)
        print(f"    Saved extra_state to {extra_path}")
    
    # Copy other files
    print("\nCopying additional files...")
    
    # Copy fsdp_config.json and update world_size
    fsdp_config_path = source_dir / "fsdp_config.json"
    if fsdp_config_path.exists():
        with open(fsdp_config_path, "r") as f:
            fsdp_config = json.load(f)
        fsdp_config["world_size"] = target_world_size
        target_fsdp_config_path = target_dir / "fsdp_config.json"
        with open(target_fsdp_config_path, "w") as f:
            json.dump(fsdp_config, f, indent=4)
        print(f"  Updated fsdp_config.json: world_size -> {target_world_size}")
    
    # Copy huggingface directory
    hf_source = source_dir / "huggingface"
    if hf_source.exists():
        hf_target = target_dir / "huggingface"
        if hf_target.exists():
            shutil.rmtree(hf_target)
        shutil.copytree(hf_source, hf_target)
        print(f"  Copied huggingface/ directory")
    
    print(f"\n✅ Conversion complete! Checkpoint saved to {target_dir}")


def main():
    parser = argparse.ArgumentParser(description="Convert FSDP checkpoint between different world sizes")
    parser.add_argument(
        "--source_ckpt_dir",
        type=str,
        required=True,
        help="Source checkpoint directory (e.g., .../global_step_12)",
    )
    parser.add_argument(
        "--target_ckpt_dir",
        type=str,
        required=True,
        help="Target checkpoint directory",
    )
    parser.add_argument(
        "--source_world_size",
        type=int,
        required=True,
        help="Source world size (number of GPUs)",
    )
    parser.add_argument(
        "--target_world_size",
        type=int,
        required=True,
        help="Target world size (number of GPUs)",
    )
    parser.add_argument(
        "--component",
        type=str,
        default="actor",
        help="Checkpoint component name (default: actor)",
    )
    
    args = parser.parse_args()
    
    convert_checkpoint(
        args.source_ckpt_dir,
        args.target_ckpt_dir,
        args.source_world_size,
        args.target_world_size,
        args.component,
    )


if __name__ == "__main__":
    main()





