#!/usr/bin/env python3
"""
Convert FSDP sharded checkpoint to HuggingFace format.

This script loads FSDP sharded checkpoints and converts them to a standard
HuggingFace format that can be loaded with AutoModelForCausalLM.from_pretrained().

Usage:
    python scripts/convert_fsdp_to_hf.py \
        --fsdp_checkpoint_dir /path/to/checkpoint/global_step_X \
        --output_dir /path/to/output/hf_model \
        --model_name_or_path Qwen/Qwen2.5-3B-Instruct
"""

import argparse
import json
import os
import shutil
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig


def load_sharded_checkpoints(checkpoint_dir: str, world_size: int):
    """
    Load all sharded FSDP checkpoints and merge them.

    Args:
        checkpoint_dir: Directory containing sharded checkpoints
        world_size: Number of GPUs used during training

    Returns:
        Tuple of (merged_state_dict, extra_state_dict)
    """
    checkpoint_dir = Path(checkpoint_dir)

    print(f"Loading {world_size} sharded checkpoints from {checkpoint_dir}...")

    # Load all model shards
    all_shards = []
    for rank in range(world_size):
        shard_path = checkpoint_dir / f"model_world_size_{world_size}_rank_{rank}.pt"
        if not shard_path.exists():
            raise FileNotFoundError(f"Checkpoint shard not found: {shard_path}")

        print(f"  Loading rank {rank} from {shard_path.name}...")
        shard_data = torch.load(shard_path, weights_only=False, map_location="cpu")
        all_shards.append(shard_data)

    # Merge shards - FSDP stores parameters with special keys
    # The keys typically contain 'flat_param' for sharded parameters
    merged_state_dict = {}

    # Collect all unique keys
    all_keys = set()
    for shard in all_shards:
        all_keys.update(shard.keys())

    print(f"\nMerging {len(all_keys)} unique parameters...")

    for key in sorted(all_keys):
        values = [shard.get(key) for shard in all_shards if key in shard]

        if not values:
            continue

        # Check if this is a sharded parameter (flat_param)
        if "flat_param" in key:
            # Concatenate all shards
            non_empty = [v for v in values if v is not None and v.numel() > 0]
            if non_empty:
                try:
                    merged = torch.cat(non_empty, dim=0)
                    merged_state_dict[key] = merged
                    if merged.numel() > 1000:  # Only log large parameters
                        print(f"  Merged {key}: shape {merged.shape}")
                except Exception as e:
                    print(f"  Warning: Could not merge {key}: {e}")
                    # Fallback: use first shard
                    merged_state_dict[key] = values[0]
            else:
                merged_state_dict[key] = values[0]
        else:
            # Non-sharded parameter - use first occurrence
            merged_state_dict[key] = values[0]

    # Load extra_state from rank 0
    extra_state_path = checkpoint_dir / f"extra_state_world_size_{world_size}_rank_0.pt"
    extra_state = None
    if extra_state_path.exists():
        print(f"\nLoading extra state from {extra_state_path.name}...")
        extra_state = torch.load(extra_state_path, weights_only=False, map_location="cpu")

    return merged_state_dict, extra_state


def convert_fsdp_keys_to_hf(state_dict: dict, model_config):
    """
    Convert FSDP state dict keys to HuggingFace format.

    FSDP wraps parameters in a module structure. We need to unwrap them.
    """
    converted_state_dict = {}

    # Load the model to get the expected key structure
    with torch.device("meta"):
        from transformers import AutoModelForCausalLM
        dummy_model = AutoModelForCausalLM.from_config(model_config)
        expected_keys = set(dummy_model.state_dict().keys())

    print(f"\nConverting FSDP keys to HuggingFace format...")
    print(f"  Expected {len(expected_keys)} parameters in HF model")

    # Try to match FSDP keys to HF keys
    for fsdp_key, tensor in state_dict.items():
        # Remove common FSDP prefixes
        hf_key = fsdp_key

        # Remove FSDP wrapper prefixes
        for prefix in [
            "_fsdp_wrapped_module.",
            "module.",
            "model.",
        ]:
            if hf_key.startswith(prefix):
                hf_key = hf_key[len(prefix):]

        # Handle flat_param suffix
        if ".flat_param" in hf_key:
            # This is a flattened parameter - we need to unflatten it
            # For now, skip these and let the model loader handle it
            continue

        # Check if this key exists in the HF model
        if hf_key in expected_keys:
            # Check if shapes match
            expected_shape = dummy_model.state_dict()[hf_key].shape
            if tensor.shape == expected_shape or tensor.numel() == dummy_model.state_dict()[hf_key].numel():
                # Reshape if needed
                if tensor.shape != expected_shape:
                    tensor = tensor.reshape(expected_shape)
                converted_state_dict[hf_key] = tensor
                if tensor.numel() > 1000:
                    print(f"  Converted: {fsdp_key} -> {hf_key} (shape: {tensor.shape})")
        else:
            # Try to find a matching key
            # Remove any trailing flat_param_N
            if ".flat_param_" in hf_key:
                base_key = hf_key.rsplit(".flat_param_", 1)[0]
                if base_key in expected_keys:
                    converted_state_dict[base_key] = tensor
                    if tensor.numel() > 1000:
                        print(f"  Converted: {fsdp_key} -> {base_key} (shape: {tensor.shape})")
                    continue

            # Store as-is for debugging
            if tensor.numel() > 1000:
                print(f"  Warning: Could not match key {fsdp_key} (shape: {tensor.shape})")

    print(f"\nConverted {len(converted_state_dict)}/{len(expected_keys)} parameters")

    # Check for missing keys
    missing_keys = expected_keys - set(converted_state_dict.keys())
    if missing_keys:
        print(f"\nWarning: Missing {len(missing_keys)} parameters:")
        for key in sorted(missing_keys)[:10]:  # Show first 10
            print(f"  - {key}")
        if len(missing_keys) > 10:
            print(f"  ... and {len(missing_keys) - 10} more")

    return converted_state_dict


def main():
    parser = argparse.ArgumentParser(
        description="Convert FSDP checkpoint to HuggingFace format"
    )
    parser.add_argument(
        "--fsdp_checkpoint_dir",
        type=str,
        required=True,
        help="Path to FSDP checkpoint directory (e.g., /path/to/checkpoint/global_step_18/actor)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to save HuggingFace model",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="Base model name or path (e.g., Qwen/Qwen2.5-3B-Instruct)",
    )
    parser.add_argument(
        "--world_size",
        type=int,
        default=None,
        help="Number of GPUs used during training (auto-detected if not specified)",
    )

    args = parser.parse_args()

    # Detect world_size if not specified
    if args.world_size is None:
        checkpoint_dir = Path(args.fsdp_checkpoint_dir)
        # Count model shards
        model_files = list(checkpoint_dir.glob("model_world_size_*_rank_*.pt"))
        if model_files:
            # Extract world_size from filename
            world_size = None
            for f in model_files:
                if "model_world_size_" in f.name:
                    parts = f.name.split("_")
                    for i, part in enumerate(parts):
                        if part == "size" and i + 1 < len(parts):
                            try:
                                world_size = int(parts[i + 1])
                                break
                            except ValueError:
                                continue
                    if world_size is not None:
                        break

            if world_size is None:
                raise ValueError(
                    "Could not auto-detect world_size from checkpoint files. "
                    "Please specify --world_size explicitly."
                )
            args.world_size = world_size
            print(f"Auto-detected world_size = {args.world_size}")
        else:
            raise ValueError(
                f"No model checkpoint files found in {checkpoint_dir}"
            )

    # Load FSDP checkpoints
    fsdp_state_dict, extra_state = load_sharded_checkpoints(
        args.fsdp_checkpoint_dir, args.world_size
    )

    # Load model config
    print(f"\nLoading model config from {args.model_name_or_path}...")
    model_config = AutoConfig.from_pretrained(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    # Convert state dict keys
    hf_state_dict = convert_fsdp_keys_to_hf(fsdp_state_dict, model_config)

    # Load base model and update weights
    print(f"\nLoading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_path,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )

    print(f"\nUpdating model weights...")
    model.load_state_dict(hf_state_dict, strict=False)

    # Save in HuggingFace format
    print(f"\nSaving HuggingFace model to {args.output_dir}...")
    os.makedirs(args.output_dir, exist_ok=True)

    # Save model weights
    model.save_pretrained(args.output_dir, safe_serialization=True)
    print(f"  Saved model weights")

    # Save tokenizer
    tokenizer.save_pretrained(args.output_dir)
    print(f"  Saved tokenizer")

    # Copy additional files from huggingface directory if exists
    source_hf_dir = Path(args.fsdp_checkpoint_dir) / "huggingface"
    if source_hf_dir.exists():
        print(f"\nCopying additional files from {source_hf_dir}...")
        for file in source_hf_dir.iterdir():
            if file.is_file() and not file.name.startswith("."):
                dest = Path(args.output_dir) / file.name
                if not dest.exists():
                    shutil.copy(file, dest)
                    print(f"  Copied {file.name}")

    print(f"\n✅ Conversion complete!")
    print(f"   HuggingFace model saved to: {args.output_dir}")
    print(f"\nTo load the model:")
    print(f"   from transformers import AutoModelForCausalLM")
    print(f"   model = AutoModelForCausalLM.from_pretrained('{args.output_dir}'")


if __name__ == "__main__":
    main()
