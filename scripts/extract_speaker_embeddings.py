#!/usr/bin/env python3
"""Build a speaker embedding pool for no-reference synthesis mode.

Extracts embeddings from a directory of reference audio files using the
campplus.onnx model and saves them as a .pt file. During no-ref inference,
a random embedding from this pool is used to control the voice timbre.

Usage:
    python scripts/extract_speaker_embeddings.py \
        --model_dir ./models/tts \
        --audio_dir ./test_data/audio \
        --output ./models/tts/spk_embeddings.pt \
        --num_speakers 50
"""

import os
import sys
import argparse
import glob
import random

import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
import onnxruntime
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Extract speaker embedding pool")
    parser.add_argument("--model_dir", required=True,
                        help="Path to TTS model directory (contains campplus.onnx)")
    parser.add_argument("--audio_dir", required=True,
                        help="Directory of 16kHz WAV files")
    parser.add_argument("--output", required=True,
                        help="Output .pt file path")
    parser.add_argument("--num_speakers", type=int, default=100,
                        help="Max number of speakers to extract (default: 100)")
    args = parser.parse_args()

    # Load campplus ONNX session
    onnx_path = os.path.join(args.model_dir, "campplus.onnx")
    if not os.path.exists(onnx_path):
        print(f"Error: campplus.onnx not found at {onnx_path}")
        sys.exit(1)

    providers = ["CUDAExecutionProvider"] if torch.cuda.is_available() else ["CPUExecutionProvider"]
    option = onnxruntime.SessionOptions()
    option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    option.intra_op_num_threads = 1
    session = onnxruntime.InferenceSession(onnx_path, sess_options=option, providers=providers)

    # Collect audio files
    wav_files = sorted(glob.glob(os.path.join(args.audio_dir, "*.wav")))
    if not wav_files:
        print(f"Error: no WAV files found in {args.audio_dir}")
        sys.exit(1)

    if len(wav_files) > args.num_speakers:
        random.seed(42)
        wav_files = random.sample(wav_files, args.num_speakers)

    print(f"Extracting embeddings from {len(wav_files)} audio files...")

    embeddings = []
    for i, wav_path in enumerate(wav_files):
        try:
            waveform, sample_rate = torchaudio.load(wav_path)
            if sample_rate != 16000:
                waveform = torchaudio.transforms.Resample(sample_rate, 16000)(waveform)

            feat = kaldi.fbank(waveform, num_mel_bins=80, dither=0, sample_frequency=16000)
            feat = feat - feat.mean(dim=0, keepdim=True)
            emb = session.run(
                None,
                {session.get_inputs()[0].name: feat.unsqueeze(dim=0).numpy()}
            )[0].flatten()
            embeddings.append(torch.tensor(emb, dtype=torch.float32))
        except Exception as e:
            print(f"  Skip {os.path.basename(wav_path)}: {e}")
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(wav_files)}")

    if not embeddings:
        print("Error: no embeddings extracted")
        sys.exit(1)

    embeddings = torch.stack(embeddings)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save(embeddings, args.output)
    print(f"Saved {embeddings.shape[0]} speaker embeddings to {args.output}")


if __name__ == "__main__":
    main()
