"""Audio processing utilities."""

import torch
import torchaudio


def load_wav(path: str, target_sample_rate: int) -> torch.Tensor:
    """Load a WAV file and resample to the target sample rate.

    Args:
        path: Path to the WAV file.
        target_sample_rate: Desired sample rate in Hz.

    Returns:
        Audio waveform tensor of shape (1, num_samples).
    """
    waveform, sample_rate = torchaudio.load(path)
    if sample_rate != target_sample_rate:
        waveform = torchaudio.transforms.Resample(
            orig_freq=sample_rate, new_freq=target_sample_rate
        )(waveform)
    return waveform


def extract_speech_feat(
    audio_path: str,
    feat_extractor,
    resample_rate: int,
    device: torch.device,
):
    """Extract speech features from an audio file.

    Loads the audio, resamples if needed, normalizes amplitude, and runs
    the feature extractor.

    Args:
        audio_path: Path to the audio file.
        feat_extractor: A callable that takes (waveform) -> features (T, D).
        resample_rate: Target sample rate for the feature extractor.
        device: Target torch device.

    Returns:
        Tuple of (speech_feat [1, T, D], speech_feat_len [1]).
    """
    waveform, sample_rate = torchaudio.load(audio_path)
    if sample_rate != resample_rate:
        waveform = torchaudio.transforms.Resample(
            orig_freq=sample_rate, new_freq=resample_rate
        )(waveform)

    # Amplitude normalization
    max_val = waveform.abs().max()
    if max_val > 0:
        waveform = waveform / max_val

    speech_feat = feat_extractor(waveform).squeeze(dim=0).transpose(0, 1)
    speech_feat = speech_feat.unsqueeze(dim=0).to(device)
    speech_feat_len = torch.tensor([speech_feat.shape[1]], dtype=torch.int32).to(device)
    return speech_feat, speech_feat_len
