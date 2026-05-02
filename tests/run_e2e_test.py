#!/usr/bin/env python3
"""One-click end-to-end test for DuraMark."""
import os, sys, subprocess, shutil, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "test_e2e_output")
DEV = "cuda:0"

def run(cmd, label):
    print(f"\n{'='*50}\n  {label}\n{'='*50}")
    t0 = time.time()
    ok = subprocess.run(cmd).returncode == 0
    print(f"  {'PASS' if ok else 'FAIL'} ({time.time()-t0:.0f}s)")
    return ok

def main():
    if os.path.exists(OUT):
        shutil.rmtree(OUT)

    ok = True
    # Embed + detect with ref
    ok &= run(["duramark", "embed", "--model_dir", f"{ROOT}/models/tts", "--input", f"{ROOT}/test_data/text.scp", "--ref_text", f"{ROOT}/test_data/ref_text.scp", "--ref_wav", f"{ROOT}/test_data/ref_wav.scp", "--output_dir", f"{OUT}/ref", "--watermark_length", "8", "--device", DEV, "--num_gpus", "1", "--workers_per_gpu", "1"], "1/4 Embed (with ref)")
    ok &= run(["duramark", "detect", "--model_dir", f"{ROOT}/models/duration_detector", "--input_dir", f"{OUT}/ref", "--text_scp", f"{ROOT}/test_data/text.scp", "--mode", "text", "--device", DEV], "2/4 Detect (with ref)")
    # Embed + detect without ref
    ok &= run(["duramark", "embed", "--model_dir", f"{ROOT}/models/tts", "--input", f"{ROOT}/test_data/text.scp", "--output_dir", f"{OUT}/no_ref", "--watermark_length", "8", "--device", DEV, "--num_gpus", "1", "--workers_per_gpu", "1"], "3/4 Embed (no ref)")
    ok &= run(["duramark", "detect", "--model_dir", f"{ROOT}/models/duration_detector", "--input_dir", f"{OUT}/no_ref", "--text_scp", f"{ROOT}/test_data/text.scp", "--mode", "text", "--device", DEV], "4/4 Detect (no ref)")

    print(f"\n{'='*50}\n  {'ALL PASS' if ok else 'SOME FAILED'}\n{'='*50}")
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
