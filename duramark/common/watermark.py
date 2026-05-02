"""Core watermark algorithms: polarity, voting, detection scoring."""

import numpy as np


def compute_soft_polarity(duration: float) -> float:
    """Compute the soft polarity of a duration value.

    Positive result means the duration leans toward an odd integer (bit 1),
    negative means it leans toward an even integer (bit 0).

    Args:
        duration: A float duration value (e.g. in frames).

    Returns:
        dist_even - dist_odd: positive -> bit 1, negative -> bit 0.
    """
    dist_even = abs(duration - 2 * round(duration / 2))
    dist_odd = abs(duration - (2 * round((duration - 1) / 2) + 1))
    return dist_even - dist_odd


def generate_random_watermark_bits(length: int) -> list:
    """Generate random binary watermark bits.

    Args:
        length: Number of bits.

    Returns:
        List of random 0/1 integers.
    """
    random_bits = np.random.randint(0, 2, size=length).tolist()
    return random_bits


def soft_voting_extraction(
    soft_durations: list, watermark_len: int = 8
) -> list:
    """Decode watermark bits via soft voting over duration polarities.

    Folds the duration sequence modulo watermark_len and accumulates
    soft polarity scores. Each accumulated score's sign determines the bit.

    Args:
        soft_durations: List of float durations (one per character/frame).
        watermark_len: Number of watermark bits to extract.

    Returns:
        List of extracted bits (0 or 1).
    """
    if not soft_durations:
        return []

    bit_scores = np.zeros(watermark_len)

    for i, duration in enumerate(soft_durations):
        bit_idx = i % watermark_len
        score = compute_soft_polarity(duration)
        bit_scores[bit_idx] += score

    final_bits = [1 if s > 0 else 0 for s in bit_scores]
    return final_bits


def calculate_unfolded_detection_score(
    soft_durations: list, target_bits: list
) -> float:
    """Compute a detection score by comparing durations against target bits.

    For each duration, polarity is compared to the corresponding target bit.
    The total score is normalized by sequence length.

    Args:
        soft_durations: List of float durations.
        target_bits: List of target watermark bits.

    Returns:
        Normalized detection score (higher = more likely watermarked).
    """
    if not soft_durations or not target_bits:
        return 0.0

    total_score = 0.0
    watermark_len = len(target_bits)

    for i, duration in enumerate(soft_durations):
        target_bit = target_bits[i % watermark_len]
        polarity = compute_soft_polarity(duration)

        if target_bit == 1:
            total_score += polarity
        else:
            total_score -= polarity

    return total_score / len(soft_durations)
