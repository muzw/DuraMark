#!/usr/bin/env python3
"""Strip training metadata from model checkpoint files, keeping only weights.

Training checkpoints often contain optimizer states, epoch/step counters,
and other metadata. This script extracts only the model state_dict to
reduce file size for distribution.
"""

import os
import sys
import argparse
import torch


def clean_and_save(src_path, dst_path=None):
    """Load a checkpoint, strip to weights only, and save."""
    if dst_path is None:
        dst_path = src_path

    data = torch.load(src_path, map_location="cpu")

    print(f"\n  {os.path.basename(src_path)}")
    print(f"    Original size: {os.path.getsize(src_path) / 1024 / 1024:.1f} MB")

    if not isinstance(data, dict):
        print(f"    Type: {type(data).__name__} (keeping as-is)")
        return

    keys = list(data.keys())
    print(f"    Keys: {keys}")

    # Check if it's a training checkpoint or pure state_dict
    is_state_dict = all("." in k or k.startswith("module.") for k in keys)
    has_optimizer = any("optimizer" in k.lower() for k in keys)
    has_metadata = any(k in ["epoch", "step", "loss", "lr", "global_step"] for k in keys)

    if not has_optimizer and not has_metadata:
        weight_keys = [k for k in keys if k in ["model", "state_dict", "module"]]
        if weight_keys:
            # Nested: {'model': state_dict, 'epoch': 10, ...}
            print(f"    Contains nested weight key: {weight_keys[0]}")
            actual_weights = data[weight_keys[0]]
            if isinstance(actual_weights, dict):
                torch.save(actual_weights, dst_path)
                print(f"    Stripped to state_dict ({len(actual_weights)} tensors)")
                print(f"    New size: {os.path.getsize(dst_path) / 1024 / 1024:.1f} MB")
                return
        print(f"    Already clean (state_dict only), no changes needed")
        return

    # Try to extract model weights
    # Common patterns in training checkpoints:
    model_keys = ["model", "state_dict", "module", "generator", "net"]
    for mk in model_keys:
        if mk in data:
            actual = data[mk]
            if isinstance(actual, dict):
                torch.save(actual, dst_path)
                print(f"    Extracted '{mk}' ({len(actual)} tensors)")
                print(f"    Removed: {[k for k in keys if k not in [mk, 'state_dict']]}")
                print(f"    New size: {os.path.getsize(dst_path) / 1024 / 1024:.1f} MB")
                return

    # If no nested model key, try stripping non-weight keys
    weight_only = {k: v for k, v in data.items()
                   if isinstance(v, torch.Tensor) or (
                       isinstance(v, dict) and all(
                           isinstance(vv, torch.Tensor) for vv in v.values()
                       )
                   )}
    if weight_only:
        torch.save(weight_only, dst_path)
        removed = set(keys) - set(weight_only.keys())
        print(f"    Removed non-weight keys: {removed}")
        print(f"    New size: {os.path.getsize(dst_path) / 1024 / 1024:.1f} MB")
    else:
        print(f"    Could not determine weight structure, keeping as-is")


def main():
    parser = argparse.ArgumentParser(
        description="Strip training metadata from model checkpoints")
    parser.add_argument("--input_dir", required=True, help="Model directory")
    parser.add_argument("--output_dir", default=None,
                        help="Output directory (default: overwrite in-place)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for f in sorted(os.listdir(args.input_dir)):
        src = os.path.join(args.input_dir, f)
        if f.endswith(".pt") or f.endswith(".pth"):
            dst = os.path.join(args.output_dir, f) if args.output_dir else src + ".tmp"
            try:
                clean_and_save(src, dst)
            except Exception as e:
                print(f"    Error: {e}")
        else:
            # Copy non-weight files as-is (yaml, onnx, etc.)
            if args.output_dir:
                import shutil
                shutil.copy2(src, os.path.join(args.output_dir, f))
                print(f"  Copied {f}")

    # Replace originals
    if args.output_dir is None:
        for f in os.listdir(args.input_dir):
            if f.endswith(".tmp"):
                os.replace(os.path.join(args.input_dir, f),
                          os.path.join(args.input_dir, f.replace(".tmp", "")))

    print(f"\nDone.")


if __name__ == "__main__":
    main()
