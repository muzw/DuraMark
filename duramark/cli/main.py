"""CLI entry point for DuraMark."""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        prog="duramark",
        description="DuraMark: Duration-based audio watermarking for speech synthesis",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # embed
    embed_parser = subparsers.add_parser("embed", help="Embed watermark into synthesized speech")
    embed_parser.add_argument("--model_dir", required=True, help="Path to TTS model directory")
    embed_parser.add_argument("--input", required=True, help="Input text SCP file (utt_id text)")
    embed_parser.add_argument("--ref_text", default=None, help="Reference text SCP file (optional)")
    embed_parser.add_argument("--ref_wav", default=None, help="Reference audio SCP file (optional)")
    embed_parser.add_argument("--output_dir", required=True, help="Output directory for generated audio")
    embed_parser.add_argument("--watermark_length", type=int, default=8, help="Number of watermark bits (default: 8)")
    embed_parser.add_argument("--watermark_bits", type=str, default=None, help="Fixed watermark bits (e.g. '10101010'). If not set, random bits are used.")
    embed_parser.add_argument("--device", type=str, default="cuda:0", help="Device for inference (default: cuda:0)")
    embed_parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs to use (default: 1)")
    embed_parser.add_argument("--workers_per_gpu", type=int, default=1, help="Workers per GPU (default: 1)")

    # detect
    detect_parser = subparsers.add_parser("detect", help="Detect watermark in audio files")
    detect_parser.add_argument("--model_dir", required=True, help="Path to detector model directory")
    detect_parser.add_argument("--input_dir", required=True, help="Directory containing WAV files to evaluate")
    detect_parser.add_argument("--text_scp", required=True, help="Text SCP file (utt_id text)")
    detect_parser.add_argument("--wm_dir", default=None, help="Directory with watermarked files (for parsing target bits from filenames). Defaults to --input_dir.")
    detect_parser.add_argument("--watermark_length", type=int, default=8, help="Number of watermark bits (default: 8)")
    detect_parser.add_argument("--mode", choices=["text", "asr"], default="text", help="Detection mode: text or asr (default: text)")
    detect_parser.add_argument("--output", type=str, default=None, help="Output path for results CSV")
    detect_parser.add_argument("--device", type=str, default="cuda:0", help="Device for inference (default: cuda:0)")

    # download-models
    download_parser = subparsers.add_parser("download-models", help="Download pre-trained model weights")
    download_parser.add_argument("--output_dir", default="./models", help="Output directory for models (default: ./models)")
    download_parser.add_argument("--models", choices=["tts", "detector", "all"], default="all", help="Which models to download (default: all)")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "embed":
        from duramark.cli.embed import run_embed
        run_embed(args)
    elif args.command == "detect":
        from duramark.cli.detect import run_detect
        run_detect(args)
    elif args.command == "download-models":
        from duramark.cli.download import run_download
        run_download(args)


if __name__ == "__main__":
    main()
