#!/usr/bin/env python3
"""
Convert FSDP sharded checkpoint to a different world_size, including optimizer state.

This handles the case where checkpoints were saved with world_size=N GPUs and need
to be loaded on world_size=M GPUs. Unlike convert_sharded_to_full_checkpoint.py,
this script ALSO converts optimizer states.

The script:
1. Model: Loads all per-rank DTensor model shards, merges into model_full.pt
2. Optimizer: Loads all per-rank flat optimizer shards, concatenates and re-splits
   for edge world_size, saves as optim_world_size_{M}_rank_{r}.pt
3. Extra state: Copies rank 0's extra_state for each target rank
4. Updates fsdp_config.json

Usage:
    # Convert 16-GPU checkpoint to be loadable on 8 GPUs
    python scripts/convert_checkpoint_worldsize.py \
        --ckpt_dir /path/to/global_step_420/actor \
        --target_world_size 8

    # Convert the entire step directory
    python scripts/convert_checkpoint_worldsize.py \
        --ckpt_dir /path/to/global_step_420 \
        --target_world_size 8 \
        --components actor
"""

import argparse
import json
import math
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


def merge_model_shards(shard_dir: Path, source_world_size: int) -> OrderedDict:
    """
    Load all per-rank DTensor model shards and merge into a full state dict.
    Same logic as convert_sharded_to_full_checkpoint.py.
    """
    print(f"  [Model] Loading {source_world_size} shards...")

    all_local_tensors = []
    for rank in range(source_world_size):
        path = shard_dir / f"model_world_size_{source_world_size}_rank_{rank}.pt"
        if not path.exists():
            raise FileNotFoundError(f"Missing model shard: {path}")
        shard = torch.load(path, map_location='cpu', weights_only=False)
        local_shard = OrderedDict()
        for k, v in shard.items():
            local_shard[k] = extract_local_tensor(v)
        all_local_tensors.append(local_shard)
        del shard

    print(f"  [Model] Merging into full state dict...")
    full_state = OrderedDict()
    keys = list(all_local_tensors[0].keys())
    for key in keys:
        values = [all_local_tensors[r][key] for r in range(source_world_size)]
        if isinstance(values[0], torch.Tensor) and values[0].dim() > 0:
            full_state[key] = torch.cat(values, dim=0)
        elif isinstance(values[0], torch.Tensor):
            full_state[key] = values[0].clone()
        else:
            full_state[key] = values[0]
    del all_local_tensors

    first_key = list(full_state.keys())[0]
    print(f"  [Model] Full state dict: {len(full_state)} keys, e.g. {first_key} shape={full_state[first_key].shape}")
    return full_state


def merge_and_reshard_optimizer(shard_dir: Path, source_world_size: int, target_world_size: int):
    """
    Load all per-rank optimizer shards, merge, and re-split for target world_size.

    FSDP SHARDED_STATE_DICT optimizer format:
    - state: {int_key: {step: scalar, exp_avg: flat_1d_tensor, exp_avg_sq: flat_1d_tensor}}
    - param_groups: [{lr, betas, ..., params: [0, 1, 2, ...]}]

    Each rank stores a shard of each flat tensor. Merging = concatenate along dim 0.
    Re-splitting = divide evenly for target world_size.
    """
    print(f"  [Optimizer] Loading {source_world_size} shards...")

    all_optim_states = []
    for rank in range(source_world_size):
        path = shard_dir / f"optim_world_size_{source_world_size}_rank_{rank}.pt"
        if not path.exists():
            raise FileNotFoundError(f"Missing optimizer shard: {path}")
        state = torch.load(path, map_location='cpu', weights_only=False)
        all_optim_states.append(state)

    # Use rank 0 as template for param_groups
    template = all_optim_states[0]
    state_keys = sorted(template['state'].keys())
    print(f"  [Optimizer] {len(state_keys)} parameter groups, merging...")

    # Merge: concatenate flat tensors across all ranks
    merged_state = {}
    for key in state_keys:
        merged_entry = {}
        sub_keys = list(template['state'][key].keys())
        for sk in sub_keys:
            values = [all_optim_states[r]['state'][key][sk] for r in range(source_world_size)]
            if isinstance(values[0], torch.Tensor) and values[0].dim() >= 1:
                # Flat 1D tensor (exp_avg, exp_avg_sq) → concatenate
                merged_entry[sk] = torch.cat(values, dim=0)
            elif isinstance(values[0], torch.Tensor):
                # Scalar tensor (step) → use rank 0's value
                merged_entry[sk] = values[0].clone()
            else:
                merged_entry[sk] = values[0]
        merged_state[key] = merged_entry

    del all_optim_states

    # Verify merge
    for key in state_keys[:2]:
        for sk, sv in merged_state[key].items():
            if isinstance(sv, torch.Tensor):
                print(f"  [Optimizer] Merged key={key}/{sk}: shape={sv.shape}")

    # Re-split for target world_size
    print(f"  [Optimizer] Re-splitting for target world_size={target_world_size}...")
    target_shards = [{} for _ in range(target_world_size)]

    for key in state_keys:
        for target_rank in range(target_world_size):
            target_shards[target_rank][key] = {}

        for sk, sv in merged_state[key].items():
            if isinstance(sv, torch.Tensor) and sv.dim() >= 1:
                total_size = sv.shape[0]
                # FSDP splits flat params evenly, with possible padding for last rank
                chunk_size = math.ceil(total_size / target_world_size)
                for target_rank in range(target_world_size):
                    start = target_rank * chunk_size
                    end = min(start + chunk_size, total_size)
                    if start < total_size:
                        chunk = sv[start:end]
                        # Pad last chunk if needed to maintain equal sizes
                        if chunk.shape[0] < chunk_size:
                            padding = torch.zeros(chunk_size - chunk.shape[0], dtype=chunk.dtype)
                            chunk = torch.cat([chunk, padding], dim=0)
                        target_shards[target_rank][key][sk] = chunk
                    else:
                        # Edge case: more ranks than elements (shouldn't happen in practice)
                        target_shards[target_rank][key][sk] = torch.zeros(chunk_size, dtype=sv.dtype)
            else:
                # Scalar (step) → replicate to all ranks
                for target_rank in range(target_world_size):
                    target_shards[target_rank][key][sk] = sv.clone() if isinstance(sv, torch.Tensor) else sv

    del merged_state

    # Build full optimizer state dicts for each target rank
    result = []
    for target_rank in range(target_world_size):
        optim_state_dict = {
            'state': target_shards[target_rank],
            'param_groups': template['param_groups'],  # Same across all ranks
        }
        result.append(optim_state_dict)

    return result


def convert_component(component_dir: Path, source_world_size: int, target_world_size: int):
    """Convert a single component (actor or critic) for new world_size."""

    print(f"\n{'='*60}")
    print(f"Converting: {component_dir}")
    print(f"Source world_size: {source_world_size} → Target world_size: {target_world_size}")
    print(f"{'='*60}")

    # 1. Merge model shards → model_full.pt
    model_full_path = component_dir / "model_full.pt"
    if model_full_path.exists():
        print(f"  [Model] model_full.pt already exists, skipping merge")
    else:
        full_model_state = merge_model_shards(component_dir, source_world_size)
        print(f"  [Model] Saving model_full.pt...")
        torch.save(full_model_state, model_full_path)
        file_size_gb = model_full_path.stat().st_size / (1024**3)
        print(f"  [Model] Saved: {model_full_path} ({file_size_gb:.2f} GB)")
        del full_model_state

    # 2. Merge & reshard optimizer → optim_world_size_{target}_rank_{r}.pt
    has_optim = (component_dir / f"optim_world_size_{source_world_size}_rank_0.pt").exists()
    if has_optim:
        target_optim_shards = merge_and_reshard_optimizer(
            component_dir, source_world_size, target_world_size
        )
        print(f"  [Optimizer] Saving {target_world_size} resharded optimizer files...")
        for rank, shard in enumerate(target_optim_shards):
            out_path = component_dir / f"optim_world_size_{target_world_size}_rank_{rank}.pt"
            torch.save(shard, out_path)
        file_size_mb = out_path.stat().st_size / (1024**2)
        print(f"  [Optimizer] Saved {target_world_size} files (~{file_size_mb:.1f} MB each)")
        del target_optim_shards
    else:
        print(f"  [Optimizer] No optimizer shards found, skipping")

    # 3. Extra state → copy rank 0's state for each target rank
    extra_src = component_dir / f"extra_state_world_size_{source_world_size}_rank_0.pt"
    extra_full_path = component_dir / "extra_state_full.pt"
    if extra_src.exists():
        extra_state = torch.load(extra_src, map_location='cpu', weights_only=False)

        # Save extra_state_full.pt (for _load_full_checkpoint path)
        torch.save(extra_state, extra_full_path)
        print(f"  [Extra] Saved extra_state_full.pt")

        # Also save per-rank files for target world_size
        for rank in range(target_world_size):
            out_path = component_dir / f"extra_state_world_size_{target_world_size}_rank_{rank}.pt"
            torch.save(extra_state, out_path)
        print(f"  [Extra] Saved {target_world_size} extra_state files")

        if 'lr_scheduler' in extra_state and extra_state['lr_scheduler']:
            print(f"    LR scheduler: last_epoch={extra_state['lr_scheduler'].get('last_epoch', 'N/A')}")
    else:
        print(f"  [Extra] No extra_state found")

    # 4. Update fsdp_config.json
    fsdp_config_path = component_dir / "fsdp_config.json"
    if fsdp_config_path.exists():
        with open(fsdp_config_path) as f:
            config = json.load(f)
        print(f"  [Config] Original: {config}")

        config["checkpoint_format"] = "full"
        config["original_world_size"] = source_world_size
        config["target_world_size"] = target_world_size

        with open(fsdp_config_path, 'w') as f:
            json.dump(config, f, indent=4)
        print(f"  [Config] Updated: {config}")

    print(f"  Done!")


def main():
    parser = argparse.ArgumentParser(
        description="Convert FSDP checkpoint to different world_size (with optimizer state)"
    )
    parser.add_argument(
        "--ckpt_dir", type=str, required=True,
        help="Path to checkpoint directory. Can be component dir (e.g., .../actor) "
             "or step dir (e.g., .../global_step_420)"
    )
    parser.add_argument(
        "--target_world_size", type=int, required=True,
        help="Target number of GPUs (world_size) for the converted checkpoint"
    )
    parser.add_argument(
        "--components", type=str, nargs="+", default=["actor"],
        help="Components to convert (default: actor). E.g., --components actor critic"
    )
    args = parser.parse_args()

    ckpt_dir = Path(args.ckpt_dir)
    if not ckpt_dir.exists():
        print(f"Error: Directory does not exist: {ckpt_dir}")
        sys.exit(1)

    target_ws = args.target_world_size

    # Detect if ckpt_dir is a component dir or step dir
    fsdp_config = ckpt_dir / "fsdp_config.json"
    if fsdp_config.exists():
        with open(fsdp_config) as f:
            config = json.load(f)
        source_ws = config["world_size"]
        if source_ws == target_ws:
            print(f"Source and target world_size are the same ({source_ws}), nothing to do.")
            return
        convert_component(ckpt_dir, source_ws, target_ws)
    else:
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
            source_ws = config["world_size"]
            if source_ws == target_ws:
                print(f"Source and target world_size are the same ({source_ws}), nothing to do.")
                continue
            convert_component(comp_dir, source_ws, target_ws)

    print(f"\n{'='*60}")
    print(f"Conversion complete!")
    print(f"You can now resume training with target_world_size={target_ws} GPUs.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
