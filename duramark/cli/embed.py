"""Watermark embedding: multi-GPU parallel TTS synthesis with watermark bits.

Adapted from infer_watermark.py.
"""

import os
import sys
import time
import multiprocessing
import queue as queue_lib
from multiprocessing import Pool, Manager

import regex
import numpy as np
import torch
import torchaudio
from torchaudio.transforms import Resample
import tqdm
import traceback
import gc


def keep_chinese(text):
    return regex.sub(r'\P{Han}+', '', text)


def split_list(lst, n):
    if n <= 0:
        return [lst]
    if not lst:
        return []
    avg = len(lst) // n
    remainder = len(lst) % n
    out = []
    start = 0
    for i in range(n):
        length = avg + 1 if i < remainder else avg
        if length > 0:
            out.append(lst[start : start + length])
            start += length
    return out


def generate_random_watermark_bits(length):
    random_bits = torch.randint(0, 2, (length,)).tolist()
    return random_bits


def check_existing_files(
    text_lines, ref_text_dict, ref_wav_dict, out_dir, watermark_lengths, enable_watermark
):
    """Build task list, skipping already-generated files and missing dependencies."""
    tasks = []
    existing_records = set()
    missing_dependency_count = 0

    if os.path.exists(out_dir):
        for f in os.listdir(out_dir):
            if not f.endswith(".wav"):
                continue
            name_part = os.path.splitext(f)[0]
            if not enable_watermark:
                existing_records.add(name_part)
            else:
                if '_' in name_part:
                    try:
                        key_part, wm_str = name_part.rsplit('_', 1)
                        existing_records.add((key_part, len(wm_str)))
                    except ValueError:
                        continue

    if len(existing_records) > 0:
        print(f"   [Check] Ignored {len(existing_records)} existing files.")

    def try_create_task(key_in, text_in, target_len_in):
        ref_audio_in = ref_wav_dict.get(key_in)
        ref_text_raw_in = ref_text_dict.get(key_in)
        if ref_audio_in is None or ref_text_raw_in is None:
            return None
        ref_word_interval_path = (
            ref_audio_in.replace("audio", "interval").replace(".wav", ".npy")
        )
        if not os.path.exists(ref_word_interval_path):
            return "missing_dep"
        return (
            key_in,
            text_in,
            keep_chinese(ref_text_raw_in),
            ref_audio_in,
            target_len_in,
            ref_word_interval_path,
        )

    if not enable_watermark:
        for line in text_lines:
            parts = line.strip().split(" ", maxsplit=1)
            if len(parts) != 2:
                continue
            key = parts[0]
            text = keep_chinese(parts[1])
            if key in existing_records:
                continue
            res = try_create_task(key, text, None)
            if res == "missing_dep":
                missing_dependency_count += 1
            elif res:
                tasks.append(res)
    else:
        for target_len in watermark_lengths:
            for line in text_lines:
                parts = line.strip().split(" ", maxsplit=1)
                if len(parts) != 2:
                    continue
                key = parts[0]
                text = keep_chinese(parts[1])
                if (key, target_len) in existing_records:
                    continue
                res = try_create_task(key, text, target_len)
                if res == "missing_dep":
                    missing_dependency_count += 1
                elif res:
                    tasks.append(res)

    if missing_dependency_count > 0:
        print(f"   [Check] Skipped {missing_dependency_count} tasks (missing .npy interval).")
    return tasks


def gen_audio(task_chunk, model_name, out_dir, process_id, gpu_id, queue):
    """Worker function: loads model and generates watermarked audio for a task chunk."""
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    success_count = 0
    fail_count = 0

    try:
        print(f"[Worker {process_id}] Start Init on GPU {gpu_id}...")

        # Inject Matcha-TTS path (needed before importing duramark.tts)
        _matcha_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            '..', 'third_party', 'Matcha-TTS'
        )
        if os.path.isdir(_matcha_path) and _matcha_path not in sys.path:
            sys.path.insert(0, _matcha_path)

        from duramark.tts.cosyvoice import CosyVoice
        from duramark.tts.utils.file_utils import load_wav

        resampler = Resample(orig_freq=22050, new_freq=16000)

        cosyvoice = CosyVoice(model_name, load_jit=False, load_onnx=False)
        print(f"[Worker {process_id}] Model Loaded Successfully.")

        with torch.inference_mode():
            for i, (key, text, ref_text, ref_audio, target_len, ref_word_interval_path) in enumerate(task_chunk):
                if target_len is None:
                    out_path = os.path.join(out_dir, f"{key}.wav")
                    watermark_bits = None
                else:
                    watermark_bits = generate_random_watermark_bits(length=target_len)
                    wm_str = "".join(map(str, watermark_bits))
                    out_path = os.path.join(out_dir, f"{key}_{wm_str}.wav")

                try:
                    ref_word_interval = np.load(ref_word_interval_path)
                    prompt_speech_16k = load_wav(ref_audio, 16000)

                    output_gen = cosyvoice.inference_zero_shot(
                        text,
                        ref_text,
                        ref_word_interval,
                        prompt_speech_16k,
                        stream=False,
                        watermark_bits=watermark_bits,
                    )

                    for j in output_gen:
                        resample_wav = resampler(j['tts_speech'])
                        torchaudio.save(out_path, resample_wav, 16000)

                    success_count += 1
                    queue.put(1)

                    del output_gen, prompt_speech_16k, ref_word_interval

                except Exception:
                    print(f"\n[Worker {process_id}] ERROR Processing {key}!")
                    traceback.print_exc()
                    fail_count += 1
                    queue.put(0)

                if (i + 1) % 5 == 0:
                    gc.collect()
                    torch.cuda.empty_cache()

    except Exception:
        print(f"\n[Worker {process_id}] CRITICAL WORKER CRASH!")
        traceback.print_exc()
        remaining = len(task_chunk) - (success_count + fail_count)
        for _ in range(remaining):
            queue.put(0)

    return success_count, fail_count


def run_embed(args):
    """Main entry point for the embed subcommand."""
    NUM_GPUS = args.num_gpus
    WORKERS_PER_GPU = args.workers_per_gpu

    model_name = args.model_dir
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    # Read input files
    text_lines = open(args.input).read().splitlines()

    ref_text_lines = open(args.ref_text).read().splitlines()
    ref_text_dict = {
        l.split()[0]: l.split(' ', 1)[1]
        for l in ref_text_lines
        if len(l.split()) >= 2
    }

    ref_wav_lines = open(args.ref_wav).read().splitlines()
    ref_wav_dict = {
        l.split()[0]: l.split(' ', 1)[1]
        for l in ref_wav_lines
        if len(l.split()) >= 2
    }

    enable_wm = True
    wm_len = args.watermark_length
    current_wm_lengths = [wm_len] if enable_wm else []

    tasks = check_existing_files(
        text_lines, ref_text_dict, ref_wav_dict,
        out_dir, current_wm_lengths, enable_wm,
    )

    if not tasks:
        print("[Skip] No new tasks.")
        return

    total_workers = NUM_GPUS * WORKERS_PER_GPU
    actual_workers = min(total_workers, len(tasks))
    task_chunks = split_list(tasks, actual_workers)

    manager = Manager()
    queue = manager.Queue()

    args_list = []
    for i in range(actual_workers):
        assigned_gpu = (i // WORKERS_PER_GPU) % NUM_GPUS
        if task_chunks[i]:
            args_list.append((
                task_chunks[i], model_name, out_dir, i, assigned_gpu, queue,
            ))

    total_tasks = len(tasks)
    print(f"[Run] {total_tasks} tasks | {actual_workers} workers | Per GPU: {WORKERS_PER_GPU}")

    with Pool(processes=actual_workers) as pool:
        result_async = pool.starmap_async(
            gen_audio,
            args_list,
            error_callback=lambda e: print(f"Pool Error: {e}"),
        )

        pbar = tqdm.tqdm(total=total_tasks, unit="it", desc="[Progress]")
        finished_count = 0
        last_update_time = time.time()

        while finished_count < total_tasks:
            try:
                _ = queue.get(timeout=60)
                pbar.update(1)
                finished_count += 1
                last_update_time = time.time()
            except queue_lib.Empty:
                time_since_last = time.time() - last_update_time
                if result_async.ready():
                    break
                if time_since_last > 300:
                    print("\n[Error] Process FROZEN for 300s. Terminating pool...")
                    pool.terminate()
                    break
            except KeyboardInterrupt:
                print("\n[Stop] User interrupted.")
                pool.terminate()
                pool.join()
                sys.exit(1)

        pbar.close()

        try:
            if result_async.ready():
                final_results = result_async.get(timeout=1)
                total_success = sum(r[0] for r in final_results)
                total_fail = sum(r[1] for r in final_results)
            else:
                print("[Warn] Pool execution was interrupted/incomplete.")
                total_success = finished_count
                total_fail = total_tasks - finished_count
        except Exception:
            total_success = finished_count
            total_fail = 0

    if total_fail > 0:
        print(f"[Done] Success: {total_success}, Failed: {total_fail}")
    else:
        print(f"[Done] All {total_success} finished.")
