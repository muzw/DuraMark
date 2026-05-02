"""File I/O utilities for SCP files, ID parsing, and list splitting."""

import os
from typing import Dict, List, Optional, Tuple


def split_list(lst: list, n: int) -> list:
    """Split a list into n roughly equal-sized chunks.

    Args:
        lst: The list to split.
        n: Number of chunks. If <= 0, returns the whole list in a single chunk.

    Returns:
        List of sub-lists.
    """
    if n <= 0:
        return [lst]
    if not lst:
        return []
    avg = len(lst) // n
    remainder = len(lst) % n
    out = []
    start = 0
    for i in range(n):
        length = avg + 1 if i < remainder else avg
        if length > 0:
            out.append(lst[start : start + length])
            start += length
    return out


def extract_id(filename: str) -> str:
    """Extract the utterance ID from a filename.

    If the filename ends with an all-binary suffix (e.g. 'utt_1010.wav'),
    the suffix is stripped and the part before it is returned as the ID.
    Otherwise the entire name without extension is returned.

    Args:
        filename: The filename (with or without path).

    Returns:
        Extracted utterance ID.
    """
    name_no_ext = os.path.splitext(filename)[0]
    parts = name_no_ext.split('_')
    last_part = parts[-1]
    # If the last segment is all 0s and 1s, it's likely watermark bits
    if len(last_part) > 0 and all(c in '01' for c in last_part):
        return "_".join(parts[:-1])
    return name_no_ext


def parse_id_msg_map(watermarked_dir: str) -> Dict[str, List[int]]:
    """Parse watermark bitstrings from watermarked audio filenames.

    Filenames are expected to follow the pattern: <utt_id>_<bits>.wav
    where <bits> is a string of 0s and 1s.

    Args:
        watermarked_dir: Directory containing watermarked WAV files.

    Returns:
        Dict mapping utterance ID to list of integer bits.
    """
    mapping = {}
    if not os.path.exists(watermarked_dir):
        return mapping
    for f in os.listdir(watermarked_dir):
        if not f.endswith('.wav'):
            continue
        try:
            name_no_ext = os.path.splitext(f)[0]
            parts = name_no_ext.rsplit('_', 1)
            if len(parts) == 2:
                key = parts[0]
                bits_str = parts[1]
                if all(c in '01' for c in bits_str):
                    mapping[key] = [int(c) for c in bits_str]
        except (ValueError, IndexError):
            continue
    return mapping


def read_scp_file(scp_path: str) -> Dict[str, str]:
    """Read a Kaldi-style SCP file.

    Each line format: <key> <value>
    The key and value are separated by whitespace (first occurrence only).

    Args:
        scp_path: Path to the SCP file.

    Returns:
        Dict mapping key to value string.
    """
    result = {}
    if not os.path.exists(scp_path):
        return result
    with open(scp_path, 'r') as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                result[parts[0]] = parts[1]
    return result
