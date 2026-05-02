"""Device utilities for CUDA-based inference."""

import torch


def get_device(gpu_id: int = 0) -> torch.device:
    """Return a CUDA device if available, otherwise CPU.

    Args:
        gpu_id: GPU index to use (default: 0).

    Returns:
        torch.device for the requested GPU or CPU.
    """
    if torch.cuda.is_available():
        return torch.device(f"cuda:{gpu_id}")
    return torch.device("cpu")
