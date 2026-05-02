"""Numba-accelerated DP alignment and blind watermark extraction.

These functions implement the ASR-blind detection mode from
watermark_eval_asr.py.
"""

import numpy as np
from numba import jit


@jit(nopython=True)
def compute_soft_polarity_jit(duration):
    """JIT-compiled soft polarity computation.

    Returns:
        Positive -> leans toward odd (bit 1), negative -> leans toward even (bit 0).
    """
    dist_even = abs(duration - 2 * round(duration / 2))
    dist_odd = abs(duration - (2 * round((duration - 1) / 2) + 1))
    return dist_even - dist_odd


@jit(nopython=True, fastmath=True)
def robust_alignment_anchored_jit(
    audio_soft_scores,
    extended_targets,
    extended_bit_indices,
    n,
    m,
    L,
    penalty_del,
    penalty_ins,
    anchor_len=4,
    anchor_scale=5.0,
):
    """Anchored Needleman-Wunsch alignment with traceback.

    The first `anchor_len` positions receive extra penalty for
    insertions/deletions, forcing a hard alignment at the beginning.

    Args:
        audio_soft_scores: shape (n,) float64 array of soft polarities.
        extended_targets: shape (m,) float64 array of mapped target bits
                          (1.0 for bit 1, -1.0 for bit 0), extended cyclically.
        extended_bit_indices: shape (m,) int32 array mapping each target position
                              to its original bit index (0..L-1).
        n: Length of audio sequence.
        m: Length of extended targets.
        L: Watermark length (number of bits).
        penalty_del: Deletion penalty (negative value).
        penalty_ins: Insertion penalty (negative value).
        anchor_len: Number of initial positions considered anchored.
        anchor_scale: Penalty multiplier within anchor region.

    Returns:
        Tuple of (detection_score, vote_buckets).
        detection_score: float, normalized alignment score.
        vote_buckets: shape (L,) float64 array, accumulated scores per bit.
    """
    dp = np.full((n + 1, m + 1), -1e10, dtype=np.float64)
    ptr = np.zeros((n + 1, m + 1), dtype=np.int32)

    dp[0, 0] = 0.0

    for j in range(1, m + 1):
        p = penalty_del * anchor_scale if j <= anchor_len else penalty_del
        dp[0, j] = dp[0, j - 1] + p
        ptr[0, j] = 2

    for i in range(1, n + 1):
        p = penalty_ins * anchor_scale if i <= anchor_len else penalty_ins
        dp[i, 0] = dp[i - 1, 0] + p
        ptr[i, 0] = 1

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            is_in_anchor = (i <= anchor_len) or (j <= anchor_len)
            scale = anchor_scale if is_in_anchor else 1.0

            curr_p_ins = penalty_ins * scale
            curr_p_del = penalty_del * scale

            score_curr = audio_soft_scores[i - 1]
            target_val = extended_targets[j - 1]
            match_score = score_curr * target_val

            s_match = dp[i - 1, j - 1] + match_score
            s_skip_audio = dp[i - 1, j] + curr_p_ins
            s_skip_target = dp[i, j - 1] + curr_p_del

            best_s = s_match
            direction = 0
            if s_skip_audio > best_s:
                best_s = s_skip_audio
                direction = 1
            if s_skip_target > best_s:
                best_s = s_skip_target
                direction = 2
            dp[i, j] = best_s
            ptr[i, j] = direction

    # Find best endpoint
    best_end_j = 0
    max_val = -1e10
    for j in range(m + 1):
        if dp[n, j] > max_val:
            max_val = dp[n, j]
            best_end_j = j

    detection_score = max_val / n if n > 0 else 0.0

    # Traceback
    vote_buckets = np.zeros(L, dtype=np.float64)
    curr_i = n
    curr_j = best_end_j

    while curr_i > 0 and curr_j > 0:
        direction = ptr[curr_i, curr_j]
        if direction == 0:
            a_score = audio_soft_scores[curr_i - 1]
            bit_idx = extended_bit_indices[curr_j - 1]
            vote_buckets[bit_idx] += a_score
            curr_i -= 1
            curr_j -= 1
        elif direction == 1:
            curr_i -= 1
        elif direction == 2:
            curr_j -= 1

    return detection_score, vote_buckets


def robust_blind_extraction_with_scores(
    soft_durations,
    watermark_len=8,
    anchor_len=4,
    anchor_scale=2.0,
    penalty_del=-6.0,
    penalty_ins=-6.0,
):
    """Blind watermark extraction via brute-force search over all candidates.

    Enumerates all 2^watermark_len candidates and selects the one with the
    highest DP alignment score against the audio's soft duration polarities.

    Args:
        soft_durations: List of float durations from the duration detector.
        watermark_len: Number of watermark bits.
        anchor_len: Anchored alignment length.
        anchor_scale: Anchor penalty multiplier.
        penalty_del: Deletion penalty.
        penalty_ins: Insertion penalty.

    Returns:
        Tuple of (best_candidate_bits, polarity_scores).
    """
    if not soft_durations:
        return [0] * watermark_len, []

    audio_soft_scores = np.array(
        [compute_soft_polarity_jit(x) for x in soft_durations], dtype=np.float64
    )
    n = len(audio_soft_scores)

    best_score = -1e10
    best_candidate_bits = [0] * watermark_len

    num_candidates = 1 << watermark_len
    max_extend = max(n + watermark_len, 200)

    for i in range(num_candidates):
        candidate_bits = [(i >> (watermark_len - 1 - k)) & 1 for k in range(watermark_len)]
        mapped_targets = [1.0 if b == 1 else -1.0 for b in candidate_bits]

        extended_targets = []
        extended_bit_indices = []

        while len(extended_targets) < max_extend:
            for idx, val in enumerate(mapped_targets):
                extended_targets.append(val)
                extended_bit_indices.append(idx)
                if len(extended_targets) >= max_extend:
                    break

        ext_targets_arr = np.array(extended_targets, dtype=np.float64)
        ext_indices_arr = np.array(extended_bit_indices, dtype=np.int32)

        det_score, _ = robust_alignment_anchored_jit(
            audio_soft_scores,
            ext_targets_arr,
            ext_indices_arr,
            n,
            len(ext_targets_arr),
            watermark_len,
            penalty_del,
            penalty_ins,
            anchor_len=anchor_len,
            anchor_scale=anchor_scale,
        )

        if det_score > best_score:
            best_score = det_score
            best_candidate_bits = candidate_bits

    return best_candidate_bits, audio_soft_scores.tolist()


def robust_alignment_and_extraction_asymmetric(
    durations,
    target_bits,
    penalty_del=-6.0,
    penalty_ins=-6.0,
    anchor_len=4,
    anchor_scale=2.0,
):
    """Detection scoring with known watermark bits.

    Computes the anchored DP alignment between audio durations and a known
    target watermark, returning both the detection score and extracted bits.

    Args:
        durations: List of float durations from the duration detector.
        target_bits: List of known watermark bits.
        penalty_del: Deletion penalty.
        penalty_ins: Insertion penalty.
        anchor_len: Anchored alignment length.
        anchor_scale: Anchor penalty multiplier.

    Returns:
        Tuple of (det_score, extracted_bits).
    """
    audio_soft_scores = np.array(
        [compute_soft_polarity_jit(dur) for dur in durations], dtype=np.float64
    )
    n = len(audio_soft_scores)
    L = len(target_bits)

    mapped_targets = [1.0 if b == 1 else -1.0 for b in target_bits]

    max_extend = max(n + L, 200)
    extended_targets = []
    extended_bit_indices = []

    while len(extended_targets) < max_extend:
        for idx, val in enumerate(mapped_targets):
            extended_targets.append(val)
            extended_bit_indices.append(idx)
            if len(extended_targets) >= max_extend:
                break

    ext_targets_arr = np.array(extended_targets, dtype=np.float64)
    ext_indices_arr = np.array(extended_bit_indices, dtype=np.int32)

    det_score, vote_buckets = robust_alignment_anchored_jit(
        audio_soft_scores,
        ext_targets_arr,
        ext_indices_arr,
        n,
        len(ext_targets_arr),
        L,
        penalty_del,
        penalty_ins,
        anchor_len=anchor_len,
        anchor_scale=anchor_scale,
    )

    extracted_bits = [1 if s > 0 else 0 for s in vote_buckets]
    return det_score, extracted_bits


def calculate_accuracy(extracted, target):
    """Compute bitwise accuracy between extracted and target bits.

    Args:
        extracted: List of extracted bits.
        target: List of ground-truth bits.

    Returns:
        Fraction of matching bits (0.0 to 1.0).
    """
    n = len(target)
    if len(extracted) != n:
        return 0.0
    match_count = sum(1 for x, y in zip(extracted, target) if x == y)
    return match_count / n
