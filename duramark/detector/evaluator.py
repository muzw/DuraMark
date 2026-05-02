"""Unified watermark detection evaluator.

Supports two detection modes:
  - 'text': Uses ground-truth text, simple soft voting + unfolded scoring.
  - 'asr':  Uses ASR transcription, anchored DP alignment + blind extraction.
"""

import os
import re
import numpy as np
import torch
from tqdm import tqdm
from hyperpyyaml import load_hyperpyyaml

from duramark.common.text import extract_text_token
from duramark.common.audio import extract_speech_feat
from duramark.common.watermark import (
    soft_voting_extraction,
    calculate_unfolded_detection_score,
)
from duramark.common.file_io import extract_id
from duramark.detector.dp import (
    robust_blind_extraction_with_scores,
    robust_alignment_and_extraction_asymmetric,
    calculate_accuracy,
)


class DurationWatermarkEvaluator:
    """Evaluator for duration-based watermark detection.

    Loads a DurationDetector model and provides methods to evaluate
    watermark detection performance on folders of audio files.

    Args:
        model_dir: Path to the duration detector model directory
                   (containing duration_detector.yaml and duration_detector.pt).
        device: torch device string or object.
    """

    def __init__(self, model_dir: str, device):
        self.device = device
        self.model_dir = model_dir

        with open(os.path.join(model_dir, 'duration_detector.yaml'), 'r') as f:
            self.configs = load_hyperpyyaml(f)

        self.detector = self.configs['detector']
        self.detector.load_state_dict(
            torch.load(
                os.path.join(model_dir, 'duration_detector.pt'),
                map_location=device,
            ),
            strict=False,
        )
        self.detector.to(device).eval()

        self.feat_extractor = self.configs['feat_extractor']
        self.sample_rate = self.configs['sample_rate']

    def infer_soft_durations(self, audio_path: str, text: str):
        """Run the duration detector on a single audio file.

        Args:
            audio_path: Path to the WAV file.
            text: Chinese text corresponding to the audio content.

        Returns:
            List of float soft durations, or empty list on failure.
        """
        if not text or len(text) == 0:
            return []
        try:
            text_token, text_token_len = extract_text_token(text, self.device)
            speech_feat, speech_feat_len = extract_speech_feat(
                audio_path, self.feat_extractor, self.sample_rate, self.device
            )
            with torch.no_grad():
                _, soft_duration = self.detector.infer(
                    text_token,
                    text_token_len,
                    speech_feat,
                    speech_feat_len,
                    self.device,
                )
            if isinstance(soft_duration, torch.Tensor):
                return soft_duration.cpu().numpy().tolist()
            return list(soft_duration)
        except Exception:
            return []

    def _load_precomputed_asr(self, folder_path: str, filename: str = "transcription_results.txt"):
        """Load pre-computed ASR transcription from a results file.

        Args:
            folder_path: Directory containing the transcription file.
            filename: Name of the transcription results file.

        Returns:
            Dict mapping filename (with extension) to ASR text.
        """
        asr_map = {}
        txt_path = os.path.join(folder_path, filename)
        if not os.path.exists(txt_path):
            return asr_map
        with open(txt_path, 'r', encoding='utf-8') as f:
            f.readline()  # skip header
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(',', 1)
                if len(parts) == 2:
                    fname = parts[0].strip()
                    content = parts[1].strip()
                    if "[FAILED/EMPTY]" in content:
                        content = ""
                    else:
                        # Normalize ASR output
                        content = (
                            content.replace('x', '叉')
                            .replace('X', '叉')
                            .replace('80后', '八零后')
                            .replace('Tony', '托尼')
                        )
                        content = content.replace('128', '幺二八').replace('119', '幺幺九')
                        content = content.replace('120打电话', '幺二零打电话')
                        content = re.sub(r'[^一-龥]', '', content)
                    asr_map[fname] = content
        return asr_map

    def evaluate_folder(
        self,
        folder_path: str,
        text_dict: dict,
        id_message_map: dict,
        mode: str = "text",
        output_txt: str = None,
    ):
        """Evaluate all WAV files in a folder for watermark detection.

        Args:
            folder_path: Directory containing WAV files.
            text_dict: Dict mapping utt_id -> ground-truth Chinese text.
            id_message_map: Dict mapping utt_id -> list of target watermark bits.
            mode: 'text' or 'asr'.
            output_txt: Optional path to write per-file results.

        Returns:
            List of result dicts, each containing:
                file, det_score, ext_acc, text, target_bits, extracted_bits.
        """
        results_list = []

        if not os.path.exists(folder_path):
            return results_list

        file_list = [f for f in os.listdir(folder_path) if f.endswith('.wav')]
        if not file_list:
            return results_list

        # For ASR mode, load precomputed transcriptions
        asr_map = {}
        if mode == "asr":
            asr_map = self._load_precomputed_asr(folder_path)

        f_out = None
        if output_txt:
            os.makedirs(os.path.dirname(output_txt), exist_ok=True)
            f_out = open(output_txt, 'w')
            f_out.write("Path, Det_Score, Extraction_Acc\n")

        pbar = tqdm(file_list, desc=f"Eval {os.path.basename(folder_path)}", leave=False)

        for filename in pbar:
            file_path = os.path.join(folder_path, filename)
            file_id = extract_id(filename)

            if file_id not in id_message_map:
                continue

            target_bits = id_message_map[file_id]

            # Determine which text to use
            if mode == "asr":
                text = asr_map.get(filename, "")
            else:
                text = text_dict.get(file_id, "")

            soft_durations = self.infer_soft_durations(file_path, text)

            if soft_durations:
                if mode == "asr":
                    # Blind extraction via DP brute-force search
                    extracted_bits, _ = robust_blind_extraction_with_scores(
                        soft_durations,
                        watermark_len=len(target_bits),
                        anchor_len=4,
                        anchor_scale=2.0,
                        penalty_del=-6.0,
                        penalty_ins=-6.0,
                    )
                    acc = calculate_accuracy(extracted_bits, target_bits)

                    # Detection score with known bits
                    det_score, _ = robust_alignment_and_extraction_asymmetric(
                        soft_durations,
                        target_bits,
                        anchor_len=4,
                        anchor_scale=2.0,
                        penalty_del=-6.0,
                        penalty_ins=-6.0,
                    )
                else:
                    # Text mode: simple soft voting
                    det_score = calculate_unfolded_detection_score(
                        soft_durations, target_bits
                    )
                    extracted_bits = soft_voting_extraction(
                        soft_durations, watermark_len=len(target_bits)
                    )
                    match_count = sum(
                        1 for x, y in zip(extracted_bits, target_bits) if x == y
                    )
                    acc = match_count / len(target_bits)

                results_list.append({
                    'file': filename,
                    'det_score': det_score,
                    'ext_acc': acc,
                    'text': text,
                    'target_bits': target_bits,
                    'extracted_bits': extracted_bits,
                })

                if f_out:
                    f_out.write(f"{filename}, {det_score:.4f}, {acc:.4f}\n")
            else:
                results_list.append({
                    'file': filename,
                    'det_score': -999.0,
                    'ext_acc': 0.0,
                    'text': text,
                    'target_bits': target_bits,
                    'extracted_bits': [0] * len(target_bits),
                })
                if f_out:
                    f_out.write(f"{filename}, -999.0, 0.0\n")

        if f_out:
            f_out.close()

        return results_list

    @staticmethod
    def extract_id_from_filename(filename: str) -> str:
        """Extract utterance ID from a filename.

        Args:
            filename: Filename (with or without path).

        Returns:
            Extracted utterance ID.
        """
        return extract_id(filename)

    @staticmethod
    def calculate_metrics(clean_scores, wm_scores, target_fpr=0.01):
        """Compute TPR, FPR, AUC, and optimal threshold.

        Args:
            clean_scores: List of detection scores for unwatermarked audio.
            wm_scores: List of detection scores for watermarked audio.
            target_fpr: Target false positive rate for threshold selection.

        Returns:
            Tuple of (tpr, fpr, threshold, auc).
        """
        if not clean_scores or not wm_scores:
            return 0.0, 0.0, 0.0, 0.5

        clean_arr = np.array(clean_scores)
        wm_arr = np.array(wm_scores)

        y_true = [0] * len(clean_arr) + [1] * len(wm_arr)
        y_scores = np.concatenate([clean_arr, wm_arr])

        from sklearn.metrics import roc_auc_score
        try:
            auc = roc_auc_score(y_true, y_scores)
        except Exception:
            auc = 0.5

        clean_sorted = np.sort(clean_arr)[::-1]
        n_clean = len(clean_arr)

        cutoff_index = int(n_clean * target_fpr)
        if cutoff_index >= n_clean:
            cutoff_index = n_clean - 1

        threshold = clean_sorted[cutoff_index]
        tpr = np.sum(wm_arr > threshold) / len(wm_arr)
        real_fpr = np.sum(clean_arr > threshold) / n_clean

        return tpr, real_fpr, threshold, auc

    @staticmethod
    def get_attack_folder_name(pert_type: str, param) -> str:
        """Generate a folder name for a given perturbation type and parameter.

        Args:
            pert_type: Type of perturbation (e.g. 'gaussian_noise', 'opus', 'rerecord').
            param: Perturbation parameter value.

        Returns:
            Folder name string.
        """
        if pert_type == "time_stretch":
            return f'time_stretch_speed_{param}'
        elif pert_type in ["gaussian_noise", "background_noise"]:
            return f'{pert_type}_snr_{param}'
        elif pert_type == "opus":
            return f'opus_bitrate_{param}k'
        elif pert_type == "quantization":
            return f'quantization_bit_{param}'
        elif pert_type in ["highpass", "lowpass"]:
            return f'{pert_type}_ratio_{param}'
        elif pert_type == "smooth":
            return f'smooth_window_{param}'
        elif pert_type == "echo":
            return f'echo_decay_{param}'
        elif pert_type == "mp3":
            return f'mp3_bitrate_{param}'
        elif pert_type == "rerecord":
            return f'rerecord_{param}'
        elif pert_type == "dac":
            return f'dac_bitrate_{param}k'
        elif pert_type == "encodec":
            return f'encodec_bitrate_{param}'
        elif pert_type == "facodec":
            return f'facodec_bitrate_{param}k'
        elif pert_type == "speechtokenizer":
            return f'speechtokenizer_bitrate_{param}k'
        elif pert_type in ["bigvgan", "vocos", "hifigan", "demucs", "frcrn"]:
            return str(param)
        else:
            return f'{pert_type}_{param}'
