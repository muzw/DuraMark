"""Model weight downloader for DuraMark.

Downloads pre-trained TTS and detector model weights from a hosting service.
The actual download URLs should be filled in before release.
"""

import os
import sys
import argparse
from typing import Dict, List


# Model manifest: mapping model_set -> list of (relative_path, url_or_google_drive_id, description)
# TODO: Fill in actual download URLs before release.
# Supported sources: Google Drive (gdown), HuggingFace Hub (huggingface_hub), or direct URL (wget).

MODEL_MANIFEST = {
    "tts": [
        # Main TTS model for watermark embedding
        # ("tts/llm.pt", "GOOGLE_DRIVE_ID_OR_URL", "LLM checkpoint"),
        # ("tts/flow.pt", "GOOGLE_DRIVE_ID_OR_URL", "Flow model checkpoint"),
        # ("tts/hift.pt", "GOOGLE_DRIVE_ID_OR_URL", "HiFiGAN checkpoint"),
        # ("tts/cosyvoice.yaml", "GOOGLE_DRIVE_ID_OR_URL", "Model config"),
        # ("tts/campplus.onnx", "GOOGLE_DRIVE_ID_OR_URL", "Speaker embedding"),
        # ("tts/speech_tokenizer_v1.onnx", "GOOGLE_DRIVE_ID_OR_URL", "Speech tokenizer"),
    ],
    "detector": [
        # Duration detector for watermark extraction
        # ("duration_detector/duration_detector.pt", "GOOGLE_DRIVE_ID_OR_URL", "Detector checkpoint"),
        # ("duration_detector/duration_detector.yaml", "GOOGLE_DRIVE_ID_OR_URL", "Detector config"),
    ],
}


def download_file(url: str, dest_path: str, desc: str = ""):
    """Download a single file.

    Attempts to use gdown for Google Drive links, huggingface_hub for HF repos,
    and falls back to wget for direct URLs.

    Args:
        url: URL or Google Drive file ID.
        dest_path: Local path to save the file.
        desc: Human-readable description.
    """
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)

    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
        print(f"  [Skip] {desc} already exists: {dest_path}")
        return

    print(f"  [Download] {desc} -> {dest_path}")

    # Try gdown (Google Drive)
    if 'drive.google.com' in url or len(url) > 30:
        try:
            import gdown
            gdown.download(url, dest_path, quiet=False)
            return
        except ImportError:
            print("    gdown not available, trying wget...")
        except Exception as e:
            print(f"    gdown failed: {e}, trying wget...")

    # Fallback: wget
    try:
        import wget
        wget.download(url, dest_path)
        print()
    except ImportError:
        print(f"    Please install gdown or wget, or manually download from: {url}")
        print(f"    to: {dest_path}")


def download_from_hf(repo_id: str, filename: str, dest_path: str, desc: str = ""):
    """Download a file from HuggingFace Hub.

    Args:
        repo_id: HuggingFace repository ID (e.g. 'username/repo-name').
        filename: File path within the repository.
        dest_path: Local path to save the file.
        desc: Human-readable description.
    """
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)

    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
        print(f"  [Skip] {desc} already exists: {dest_path}")
        return

    print(f"  [Download] {desc} from HF:{repo_id} -> {dest_path}")
    try:
        from huggingface_hub import hf_hub_download
        downloaded = hf_hub_download(repo_id=repo_id, filename=filename)
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        if downloaded != dest_path:
            import shutil
            shutil.copy(downloaded, dest_path)
    except ImportError:
        print(f"    huggingface_hub not installed. Please install it or manually download.")
    except Exception as e:
        print(f"    Download failed: {e}")


def run_download(args=None):
    """Download model weights.

    Args:
        args: argparse.Namespace or dict with 'output_dir', 'models'.
    """
    if args is None:
        return

    output_dir = args.output_dir
    models_filter = args.models

    selected_sets: List[str] = []
    if models_filter == "all":
        selected_sets = list(MODEL_MANIFEST.keys())
    else:
        selected_sets = [models_filter]

    if not MODEL_MANIFEST:
        print("Model manifest is empty. Please add download URLs before release.")
        print("You can manually copy model files from your training directory to:")
        print(f"  {output_dir}")
        return

    for model_set in selected_sets:
        if model_set not in MODEL_MANIFEST:
            print(f"[Warn] Unknown model set: {model_set}")
            continue

        print(f"\n--- Downloading {model_set} models ---")
        for entry in MODEL_MANIFEST[model_set]:
            if len(entry) == 3:
                rel_path, url, desc = entry
            else:
                continue

            dest_path = os.path.join(output_dir, rel_path)

            if url.startswith("hf://"):
                repo_and_file = url[5:]
                parts = repo_and_file.split("/", 1)
                if len(parts) == 2:
                    repo_id, filename = parts
                    download_from_hf(repo_id, filename, dest_path, desc)
            else:
                download_file(url, dest_path, desc)

    print(f"\nAll downloads complete. Models saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Download DuraMark pre-trained models")
    parser.add_argument("--output_dir", default="./models", help="Output directory (default: ./models)")
    parser.add_argument("--models", choices=["tts", "detector", "all"], default="all", help="Model sets to download (default: all)")
    args = parser.parse_args()
    run_download(args)


if __name__ == "__main__":
    main()
