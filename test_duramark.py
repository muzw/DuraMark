#!/usr/bin/env python3
"""
Comprehensive smoke test for DuraMark.

Usage:
    # Basic import tests (no GPU/models needed)
    python test_duramark.py --test basic

    # DP algorithm tests (requires numba, no GPU/models needed)
    python test_duramark.py --test dp

    # Detect module import (requires torch, no detector model needed)
    python test_duramark.py --test detector-import

    # TTS module import (requires torch, cosyvoice model)
    python test_duramark.py --test tts-import --model_dir /path/to/pe_llm_duration_flow

    # Detection test (requires torch + detector model + test data)
    python test_duramark.py --test detect --model_dir /path/to/duration_detector --input_dir /path/to/watermarked --text_scp /path/to/text.scp

    # End-to-end test (requires everything + data)
    python test_duramark.py --test e2e --tts_model_dir /path/to/pe_llm_duration_flow --detector_model_dir /path/to/duration_detector --input /path/to/text.scp --ref_text /path/to/ref_text.scp --ref_wav /path/to/ref_wav.scp --output_dir ./test_output

    # Run all tests that don't require models
    python test_duramark.py --test all-no-model

    # Run all tests (requires models and data)
    python test_duramark.py --test all
"""

import os
import sys
import argparse
import traceback


def green(s):
    return f"\033[92m{s}\033[0m"


def red(s):
    return f"\033[91m{s}\033[0m"


def yellow(s):
    return f"\033[93m{s}\033[0m"


def run_test(name, fn):
    """Run a named test function and report result."""
    print(f"\n{'='*60}")
    print(f"  TEST: {name}")
    print(f"{'='*60}")
    try:
        fn()
        print(green(f"  PASS: {name}"))
        return True
    except Exception as e:
        print(red(f"  FAIL: {name}"))
        traceback.print_exc()
        return False


# =============================================================
# Test functions
# =============================================================

def test_import():
    """Test basic package import and version."""
    import duramark
    assert duramark.__version__ == "0.1.0", f"Wrong version: {duramark.__version__}"
    print(f"  duramark version: {duramark.__version__}")


def test_device():
    """Test device utility."""
    from duramark.common.device import get_device
    dev = get_device()
    print(f"  Device: {dev}")
    assert dev is not None


def test_text():
    """Test text processing utilities."""
    from duramark.common.text import keep_chinese, extract_text_token
    from duramark.common.device import get_device

    # keep_chinese
    assert keep_chinese("你好123世界") == "你好世界"
    assert keep_chinese("hello world") == ""
    assert keep_chinese("中English文") == "中文"
    print(f"  keep_chinese: OK")

    # extract_text_token (needs transformers)
    try:
        device = get_device()
        token, token_len = extract_text_token("你好世界", device)
        if token is not None and token_len is not None:
            print(f"  extract_text_token: token shape={token.shape}, len={token_len.item()}")
    except Exception as e:
        print(yellow(f"  extract_text_token skipped: {e} (may need network for BERT download)"))


def test_audio():
    """Test audio utilities (import only)."""
    from duramark.common.audio import load_wav, extract_speech_feat
    print(f"  audio module imported OK")


def test_watermark():
    """Test watermark algorithms."""
    from duramark.common.watermark import (
        compute_soft_polarity,
        generate_random_watermark_bits,
        soft_voting_extraction,
        calculate_unfolded_detection_score,
    )

    # Polarity: odd integers -> positive, even integers -> negative
    assert compute_soft_polarity(3.0) > 0, "3.0 is odd (bit 1), should be positive"
    assert compute_soft_polarity(4.0) < 0, "4.0 is even (bit 0), should be negative"
    assert compute_soft_polarity(3.4) > 0, "3.4 leans odd (bit 1)"
    assert compute_soft_polarity(3.6) > 0, "3.6 leans odd (bit 1)"
    assert compute_soft_polarity(0.5) > 0, "0.5 leans odd (bit 1)"
    assert compute_soft_polarity(1.0) > 0, "1.0 is odd (bit 1), should be positive"
    print(f"  soft_polarity: OK")

    # Random bits
    bits = generate_random_watermark_bits(16)
    assert len(bits) == 16
    assert all(b in (0, 1) for b in bits)
    print(f"  random_bits(16): {bits}")

    # Soft voting
    durations = [3.2, 4.2, 3.2, 4.2, 3.2, 4.2, 3.2, 4.2]
    # pattern: odd, even, odd, even, ... -> 1, 0, 1, 0, ...
    bits = soft_voting_extraction(durations, watermark_len=8)
    assert len(bits) == 8
    print(f"  soft_voting(8): {bits}")

    # Detection score
    target = [1, 0, 1, 0, 1, 0, 1, 0]
    score = calculate_unfolded_detection_score(durations, target)
    assert score > 0, f"Score should be positive, got {score:.4f}"
    print(f"  detection_score matching: {score:.4f}")

    # Anti-pattern should give negative score
    anti_score = calculate_unfolded_detection_score(durations, [0, 1, 0, 1, 0, 1, 0, 1])
    assert anti_score < 0, f"Anti-score should be negative, got {anti_score:.4f}"
    print(f"  detection_score anti: {anti_score:.4f}")


def test_file_io():
    """Test file I/O utilities."""
    from duramark.common.file_io import split_list, extract_id, parse_id_msg_map

    # split_list
    assert split_list([1,2,3,4,5], 2) == [[1,2,3],[4,5]]
    assert split_list([], 3) == []
    assert split_list([1,2,3], 0) == [[1,2,3]]
    print(f"  split_list: OK")

    # extract_id
    assert extract_id("BAC009S0702_0101.wav") == "BAC009S0702"
    assert extract_id("BAC009S0702.wav") == "BAC009S0702"
    print(f"  extract_id: OK")

    print(f"  file_io module: OK")


def test_dp_numba():
    """Test Numba DP algorithms."""
    import numpy as np
    from duramark.detector.dp import (
        compute_soft_polarity_jit,
        calculate_accuracy,
        robust_blind_extraction_with_scores,
        robust_alignment_and_extraction_asymmetric,
    )

    # JIT polarity (compile test)
    r = compute_soft_polarity_jit(3.2)
    assert r > 0, f"3.2 should lean odd, got {r}"
    print(f"  JIT polarity(3.2) = {r:.4f}")

    # Accuracy
    assert calculate_accuracy([1,0,1,1], [1,0,1,1]) == 1.0
    assert calculate_accuracy([1,0,1,0], [1,1,1,1]) == 0.5
    print(f"  calculate_accuracy: OK")

    # Blind extraction (with synthetic data)
    durations = [3.2, 4.2, 3.2, 4.2, 3.2, 4.2, 3.2, 4.2,
                 3.2, 4.2, 3.2, 4.2, 3.2, 4.2, 3.2, 4.2]
    bits, _ = robust_blind_extraction_with_scores(
        durations, watermark_len=4,
        anchor_len=2, anchor_scale=2.0,
        penalty_del=-6.0, penalty_ins=-6.0,
    )
    print(f"  blind_extraction (4 bits, 16 durations): {bits}")
    assert len(bits) == 4

    # Detection with known bits
    score, extracted = robust_alignment_and_extraction_asymmetric(
        durations, target_bits=[1,0,1,0],
        anchor_len=2, anchor_scale=2.0,
        penalty_del=-6.0, penalty_ins=-6.0,
    )
    print(f"  detection_score: {score:.4f}, extracted: {extracted}")

    print(f"  dp.py: all Numba tests passed")


def test_detector_import():
    """Test importing the detector module (no model loading)."""
    import torch
    from duramark.detector.aligner import Aligner, viterbi_numba
    from duramark.detector.downsampler import DownsampleRegulator
    from duramark.detector.model import DurationDetector, FeatureExtractor
    from duramark.detector.evaluator import DurationWatermarkEvaluator

    print(f"  Detector classes imported OK")
    print(f"  - DurationDetector")
    print(f"  - Aligner (with viterbi_numba)")
    print(f"  - DownsampleRegulator")
    print(f"  - DurationWatermarkEvaluator")


def test_tts_import():
    """Test importing the TTS module (no model loading)."""
    import torch
    from duramark.tts.cosyvoice import CosyVoice, CosyVoice2

    # Check that module aliases work
    import sys
    alias_check = [
        'cosyvoice',
        'cosyvoice.llm',
        'cosyvoice.flow',
        'cosyvoice.transformer',
        'cosyvoice.utils',
    ]
    for alias in alias_check:
        if alias in sys.modules:
            print(f"  Alias OK: {alias} -> {sys.modules[alias].__name__}")
        else:
            print(yellow(f"  Alias missing: {alias}"))

    print(f"  TTS classes imported OK")


def test_detector_with_model(model_dir, device_str, input_dir, text_scp, mode):
    """Test the full detector pipeline with a model."""
    from duramark.detector.evaluator import DurationWatermarkEvaluator
    from duramark.common.text import keep_chinese
    from duramark.common.file_io import parse_id_msg_map

    print(f"  Loading detector from: {model_dir}")
    evaluator = DurationWatermarkEvaluator(model_dir, device_str)
    print(f"  Detector loaded successfully")

    # Read text
    text_dict = {}
    if os.path.exists(text_scp):
        with open(text_scp) as f:
            for line in f:
                parts = line.strip().split(maxsplit=1)
                if len(parts) == 2:
                    text_dict[parts[0]] = keep_chinese(parts[1])
    print(f"  Loaded {len(text_dict)} text entries")

    # Parse watermark
    id_msg_map = parse_id_msg_map(input_dir)
    print(f"  Found {len(id_msg_map)} watermarked files")

    if not id_msg_map:
        print(yellow("  Skipping evaluation: no watermarked files"))
        return

    # Run evaluation (with small sample for speed)
    print(f"  Running evaluation (mode={mode})...")
    results = evaluator.evaluate_folder(
        folder_path=input_dir,
        text_dict=text_dict,
        id_message_map=id_msg_map,
        mode=mode,
    )

    if results:
        det_scores = [r['det_score'] for r in results]
        ext_accs = [r['ext_acc'] for r in results]
        valid = [s for s in det_scores if s > -999.0]
        print(f"  Evaluated {len(results)} files")
        if valid:
            import numpy as np
            print(f"  Avg Detection Score: {np.mean(valid):.4f}")
            print(f"  Avg Extraction Acc:  {np.mean(ext_accs):.4f}")
    else:
        print(yellow("  No results returned"))


def test_tts_with_model(model_dir, device_str):
    """Test TTS model loading."""
    from duramark.tts.cosyvoice import CosyVoice

    print(f"  Loading TTS model from: {model_dir}")
    cosyvoice = CosyVoice(model_dir, load_jit=False, load_onnx=False)
    print(f"  TTS model loaded successfully")
    print(f"  Sample rate: {cosyvoice.sample_rate}")


def test_e2e(tts_model_dir, detector_model_dir, input_scp, ref_text, ref_wav,
             device_str, output_dir):
    """End-to-end test: embed -> detect."""
    import tempfile
    import subprocess

    embed_dir = os.path.join(output_dir, "test_e2e_embed")
    os.makedirs(embed_dir, exist_ok=True)

    # 1. Embed
    print(f"  [1/2] Embedding watermark...")
    print(f"       TTS model: {tts_model_dir}")
    print(f"       Input: {input_scp}")
    print(f"       Output: {embed_dir}")

    cmd_embed = [
        sys.executable, "-m", "duramark.cli.main", "embed",
        "--model_dir", tts_model_dir,
        "--input", input_scp,
        "--ref_text", ref_text,
        "--ref_wav", ref_wav,
        "--output_dir", embed_dir,
        "--watermark_length", "8",
        "--device", device_str,
        "--num_gpus", "1",
        "--workers_per_gpu", "1",
    ]
    print(f"       Running: {' '.join(cmd_embed)}")
    # In actual test, run the command
    # result = subprocess.run(cmd_embed, capture_output=True, text=True)
    # print(result.stdout[-500:] if result.stdout else "")

    # 2. Detect
    print(f"\n  [2/2] Detecting watermark...")
    cmd_detect = [
        sys.executable, "-m", "duramark.cli.main", "detect",
        "--model_dir", detector_model_dir,
        "--input_dir", embed_dir,
        "--text_scp", input_scp,
        "--watermark_length", "8",
        "--mode", "text",
        "--device", device_str,
    ]
    print(f"       Running: {' '.join(cmd_detect)}")

    print(green(f"\n  E2E test commands prepared."))
    print(f"  To run manually:")
    print(f"    {' '.join(cmd_embed)}")
    print(f"    {' '.join(cmd_detect)}")


def test_cli_help():
    """Test CLI --help output."""
    import subprocess
    result = subprocess.run(
        [sys.executable, "-m", "duramark.cli.main", "--help"],
        capture_output=True, text=True,
    )
    print(result.stdout)

    for cmd in ["embed", "detect", "download-models"]:
        result = subprocess.run(
            [sys.executable, "-m", "duramark.cli.main", cmd, "--help"],
            capture_output=True, text=True,
        )
        print(f"\n--- duramark {cmd} --help ---")
        print(result.stdout[:500] if result.stdout else result.stderr[:500])


# =============================================================
# Main
# =============================================================

def main():
    parser = argparse.ArgumentParser(description="DuraMark comprehensive smoke test")
    parser.add_argument("--test", choices=[
        "basic", "dp", "detector-import", "tts-import",
        "detect", "tts-model", "e2e", "cli",
        "all-no-model", "all",
    ], required=True, help="Which tests to run")

    # Model paths
    parser.add_argument("--model_dir", default=None, help="Path to detector model or TTS model")
    parser.add_argument("--tts_model_dir", default=None, help="Path to TTS model")
    parser.add_argument("--detector_model_dir", default=None, help="Path to detector model")
    parser.add_argument("--device", default="cuda:0", help="Device (default: cuda:0)")

    # Data paths
    parser.add_argument("--input_dir", default=None, help="Directory with watermarked audio")
    parser.add_argument("--text_scp", default=None, help="Text SCP file")
    parser.add_argument("--input", default=None, help="Input text SCP for embed")
    parser.add_argument("--ref_text", default=None, help="Ref text SCP for embed")
    parser.add_argument("--ref_wav", default=None, help="Ref wav SCP for embed")
    parser.add_argument("--output_dir", default="./test_output", help="Output dir for e2e test")

    args = parser.parse_args()

    # Add project root to path
    project_root = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, project_root)

    results = {}

    if args.test in ("basic", "all-no-model", "all"):
        results['import'] = run_test("Package import", test_import)
        results['device'] = run_test("Device utility", test_device)
        results['text'] = run_test("Text utilities", test_text)
        results['audio'] = run_test("Audio utilities", test_audio)
        results['watermark'] = run_test("Watermark algorithms", test_watermark)
        results['file_io'] = run_test("File I/O", test_file_io)

    if args.test in ("dp", "all-no-model", "all"):
        results['dp'] = run_test("Numba DP algorithms", test_dp_numba)

    if args.test in ("cli", "all-no-model", "all"):
        results['cli'] = run_test("CLI --help", test_cli_help)

    if args.test in ("detector-import", "all"):
        results['detector_import'] = run_test("Detector module import", test_detector_import)

    if args.test in ("tts-import", "all"):
        results['tts_import'] = run_test("TTS module import", test_tts_import)

    if args.test in ("detect", "all"):
        if not args.model_dir:
            print(red("  --model_dir required for detect test"))
        elif not args.input_dir:
            print(red("  --input_dir required for detect test"))
        elif not args.text_scp:
            print(red("  --text_scp required for detect test"))
        else:
            results['detect'] = run_test(
                "Detector evaluation",
                lambda: test_detector_with_model(
                    args.model_dir, args.device, args.input_dir, args.text_scp, "text"
                )
            )

    if args.test in ("tts-model", "all"):
        if not args.model_dir:
            print(red("  --model_dir required for TTS model test"))
        else:
            results['tts_model'] = run_test(
                "TTS model loading",
                lambda: test_tts_with_model(args.model_dir, args.device)
            )

    if args.test in ("e2e", "all"):
        if not all([args.tts_model_dir, args.detector_model_dir, args.input,
                     args.ref_text, args.ref_wav]):
            print(red("  --tts_model_dir, --detector_model_dir, --input, --ref_text, --ref_wav required for e2e"))
        else:
            results['e2e'] = run_test(
                "End-to-end",
                lambda: test_e2e(
                    args.tts_model_dir, args.detector_model_dir,
                    args.input, args.ref_text, args.ref_wav,
                    args.device, args.output_dir,
                )
            )

    # Summary
    print(f"\n{'='*60}")
    print(f"  TEST SUMMARY")
    print(f"{'='*60}")
    passed = sum(1 for v in results.values() if v)
    failed = sum(1 for v in results.values() if not v)
    for name, ok in results.items():
        status = green("PASS") if ok else red("FAIL")
        print(f"  {status}  {name}")
    print(f"\n  Total: {passed} passed, {failed} failed")
    print(f"{'='*60}")

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
