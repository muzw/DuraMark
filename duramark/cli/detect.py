"""Watermark detection: evaluate audio files for watermark presence.

Supports two modes:
  - 'text': Uses ground-truth text, simple soft voting.
  - 'asr':  Uses ASR transcription, anchored DP alignment.

Adapted from watermark_eval.py and watermark_eval_asr.py.
"""

import os

import numpy as np
import torch

from duramark.common.text import keep_chinese
from duramark.common.file_io import parse_id_msg_map, read_scp_file
from duramark.detector.evaluator import DurationWatermarkEvaluator


def run_detect(args):
    """Main entry point for the detect subcommand."""
    model_dir = args.model_dir
    input_dir = args.input_dir
    text_scp = args.text_scp
    watermark_length = args.watermark_length
    mode = args.mode
    device_str = args.device
    output_path = args.output

    # Determine WM directory for parsing target bits
    wm_dir = args.wm_dir if args.wm_dir else input_dir

    # Read text SCP
    text_dict = {}
    if os.path.exists(text_scp):
        text_lines = open(text_scp).read().splitlines()
        text_dict = {
            line.split(" ", 1)[0]: keep_chinese(line.split(" ", 1)[1])
            for line in text_lines
            if len(line.split(" ", 1)) == 2
        }
    else:
        print(f"[Error] Text SCP not found: {text_scp}")
        return

    # Parse target watermark bits from filenames
    id_msg_map = parse_id_msg_map(wm_dir)
    if not id_msg_map:
        print(f"[Error] No watermarked files found in {wm_dir}")
        return

    # Initialize evaluator
    evaluator = DurationWatermarkEvaluator(model_dir, device_str)

    # Evaluate folder
    print(f"\n--- Detecting watermarks (mode={mode}) ---")
    print(f"  Input dir: {input_dir}")
    print(f"  Model dir: {model_dir}")
    print(f"  Device: {device_str}")
    print(f"  Watermark length: {watermark_length} bits")

    results = evaluator.evaluate_folder(
        folder_path=input_dir,
        text_dict=text_dict,
        id_message_map=id_msg_map,
        mode=mode,
        output_txt=output_path,
    )

    if not results:
        print("[Skip] No results.")
        return

    # Summary statistics
    det_scores = [r['det_score'] for r in results]
    ext_accs = [r['ext_acc'] for r in results]

    valid_scores = [s for s in det_scores if s > -999.0]
    if valid_scores:
        print(f"\n--- Results ({len(results)} files) ---")
        print(f"  Avg Detection Score: {np.mean(valid_scores):.4f}")
        print(f"  Avg Extraction Acc:  {np.mean(ext_accs):.4f}")
        print(f"  Min Detection Score: {np.min(valid_scores):.4f}")
        print(f"  Max Detection Score: {np.max(valid_scores):.4f}")

        # Per-file details
        print(f"\n--- Per-file Details ---")
        for r in results:
            if r['det_score'] <= -999.0:
                continue
            tgt_str = "".join(map(str, r['target_bits']))
            ext_str = "".join(map(str, r['extracted_bits']))
            match = sum(1 for a, b in zip(r['target_bits'], r['extracted_bits']) if a == b)
            print(f"  {r['file']}")
            print(f"    Detect Score: {r['det_score']:.4f}  |  Ext Acc: {r['ext_acc']:.2%} ({match}/{len(r['target_bits'])})")
            print(f"    Target:    {tgt_str}")
            print(f"    Extracted: {ext_str}")

    if output_path:
        print(f"\n  Results saved to: {output_path}")
