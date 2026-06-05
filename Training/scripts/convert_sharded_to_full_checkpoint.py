#!/usr/bin/env python3
"""
Convert FSDP sharded checkpoint (DTensor format) to full checkpoint format.

This handles the case where checkpoints were saved with world_size=N GPUs
(as per-rank DTensor shards) and need to be loaded on a different number of GPUs.

The script:
1. Loads all per-rank DTensor model shards
2. Extracts _local_tensor and concatenates along shard dimension to get full params
3. Saves as model_full.pt (plain tensor state dict, loadable on any world_size)
4. Copies extra_state (LR scheduler + RNG) from rank 0
5. Updates fsdp_config.json to indicate "full" checkpoint format

Usage:
    python scripts/convert_sharded_to_full_checkpoint.py \
        --ckpt_dir /path/to/global_step_35/actor
    
    # Or convert the entire step directory (actor + critic)
    python scripts/convert_sharded_to_full_checkpoint.py \
        --ckpt_dir /path/to/global_step_35 \
        --components actor
"""

import argparse
import json
import os
import sys
from collections import OrderedDict
from pathlib import Path

import torch


def extract_local_tensor(value):
    """Extract plain tensor from DTensor or return as-is."""
    if hasattr(value, '_local_tensor'):
        return value._local_tensor.clone()
    elif isinstance(value, torch.Tensor):
        return value.clone()
    return value


def merge_dtensor_shards(shard_dir: Path, source_world_size: int) -> OrderedDict:
    """
    Load all per-rank DTensor model shards and merge into a full state dict.
    
    Each DTensor has:
    - _local_tensor: the actual data shard for that rank
    - placements: (Shard(dim=0),) indicating sharding along dim 0
    
    We concatenate all local tensors along dim 0 to reconstruct the full parameter.
    """
    print(f"  Loading {source_world_size} model shards...")
    
    # Load all shards and extract local tensors
    all_local_tensors = []
    for rank in range(source_world_size):
        path = shard_dir / f"model_world_size_{source_world_size}_rank_{rank}.pt"
        if not path.exists():
            raise FileNotFoundError(f"Missing shard file: {path}")
        
        shard = torch.load(path, map_location='cpu', weights_only=False)
        local_shard = OrderedDict()
        for k, v in shard.items():
            local_shard[k] = extract_local_tensor(v)
        all_local_tensors.append(local_shard)
        print(f"    Loaded rank {rank} ({len(shard)} keys)")
        del shard  # Free memory
    
    # Merge: concatenate all local tensors along dim 0
    print(f"  Merging into full state dict...")
    full_state = OrderedDict()
    keys = list(all_local_tensors[0].keys())
    
    for key in keys:
        values = [all_local_tensors[r][key] for r in range(source_world_size)]
        
        if isinstance(values[0], torch.Tensor) and values[0].dim() > 0:
            # Concatenate along shard dimension (dim 0 for Shard(dim=0))
            full_state[key] = torch.cat(values, dim=0)
        elif isinstance(values[0], torch.Tensor):
            # Scalar tensors - just use rank 0's value
            full_state[key] = values[0].clone()
        else:
            full_state[key] = values[0]
    
    # Free shard memory
    del all_local_tensors
    
    return full_state


def convert_component(component_dir: Path, source_world_size: int):
    """Convert a single component (actor or critic) from sharded to full format."""
    
    print(f"\n{'='*60}")
    print(f"Converting: {component_dir}")
    print(f"Source world_size: {source_world_size}")
    print(f"{'='*60}")
    
    # 1. Merge model shards
    full_model_state = merge_dtensor_shards(component_dir, source_world_size)
    
    # Verify: check a known parameter shape
    first_key = list(full_model_state.keys())[0]
    first_val = full_model_state[first_key]
    print(f"  Full state dict: {len(full_model_state)} keys")
    print(f"  Example: {first_key} shape={first_val.shape}")
    
    # 2. Save model_full.pt
    model_full_path = component_dir / "model_full.pt"
    print(f"  Saving model_full.pt...")
    torch.save(full_model_state, model_full_path)
    file_size_gb = model_full_path.stat().st_size / (1024**3)
    print(f"  Saved: {model_full_path} ({file_size_gb:.2f} GB)")
    del full_model_state
    
    # 3. Copy extra_state from rank 0
    extra_src = component_dir / f"extra_state_world_size_{source_world_size}_rank_0.pt"
    extra_dst = component_dir / "extra_state_full.pt"
    if extra_src.exists():
        extra_state = torch.load(extra_src, map_location='cpu', weights_only=False)
        torch.save(extra_state, extra_dst)
        print(f"  Copied extra_state from rank 0 -> extra_state_full.pt")
        if 'lr_scheduler' in extra_state and extra_state['lr_scheduler']:
            print(f"    LR scheduler: last_epoch={extra_state['lr_scheduler'].get('last_epoch', 'N/A')}")
    else:
        print(f"  Warning: No extra_state found at {extra_src}")
    
    # 4. Update fsdp_config.json
    fsdp_config_path = component_dir / "fsdp_config.json"
    if fsdp_config_path.exists():
        with open(fsdp_config_path) as f:
            config = json.load(f)
        print(f"  Original fsdp_config: {config}")
        
        # Update to indicate full format is available
        config["checkpoint_format"] = "full"
        config["original_world_size"] = source_world_size
        
        with open(fsdp_config_path, 'w') as f:
            json.dump(config, f, indent=4)
        print(f"  Updated fsdp_config: {config}")
    
    print(f"  Done!")


def main():
    parser = argparse.ArgumentParser(
        description="Convert FSDP sharded DTensor checkpoint to full checkpoint format"
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        required=True,
        help="Path to checkpoint directory. Can be component dir (e.g., .../actor) "
             "or step dir (e.g., .../global_step_35)"
    )
    parser.add_argument(
        "--components",
        type=str,
        nargs="+",
        default=["actor"],
        help="Components to convert (default: actor). E.g., --components actor critic"
    )
    args = parser.parse_args()
    
    ckpt_dir = Path(args.ckpt_dir)
    if not ckpt_dir.exists():
        print(f"Error: Directory does not exist: {ckpt_dir}")
        sys.exit(1)
    
    # Detect if ckpt_dir is a component dir or step dir
    fsdp_config = ckpt_dir / "fsdp_config.json"
    if fsdp_config.exists():
        # This is a component directory (e.g., /actor)
        with open(fsdp_config) as f:
            config = json.load(f)
        source_world_size = config["world_size"]
        convert_component(ckpt_dir, source_world_size)
    else:
        # This is a step directory, convert each requested component
        for comp in args.components:
            comp_dir = ckpt_dir / comp
            if not comp_dir.exists():
                print(f"Warning: Component directory not found: {comp_dir}, skipping")
                continue
            
            comp_fsdp_config = comp_dir / "fsdp_config.json"
            if not comp_fsdp_config.exists():
                print(f"Warning: No fsdp_config.json in {comp_dir}, skipping")
                continue
            
            with open(comp_fsdp_config) as f:
                config = json.load(f)
            source_world_size = config["world_size"]
            convert_component(comp_dir, source_world_size)
    
    print(f"\n{'='*60}")
    print("Conversion complete!")
    print("The full checkpoint files (model_full.pt, extra_state_full.pt) are now")
    print("available alongside the original sharded files.")
    print("The original sharded files are preserved (not deleted).")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
