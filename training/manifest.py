"""Manifest generation for APD training data.

Separated from dataset.py to avoid importing torch in worker processes,
which would cause OOM when spawning multiple workers on Windows.
"""

import json
import multiprocessing as mp
import os
import random
from pathlib import Path

import numpy as np
import soundfile as sf

from .augmentation import AudioDegrader, load_audio, random_crop
from .config import AudioConfig, APDLabelConfig
from .pseudo_label import compute_apd_label


def _worker_process_chunk(args):
    """Worker function for parallel manifest generation.

    Each worker processes a chunk of sample indices and writes results
    to a temporary file to avoid accumulating entries in memory.
    Returns the path to the temp file.
    """
    (
        worker_id,
        indices,
        clean_files,
        noise_files,
        speaker_files,
        degraded_dir,
        tmp_dir,
        audio_config,
        degradation_config,
        label_config,
        seed,
    ) = args

    # Per-worker RNG seeding for reproducibility
    random.seed(seed)
    np.random.seed(seed)

    degrader = AudioDegrader(noise_files, speaker_files, degradation_config, audio_config)
    sr = audio_config.sample_rate
    window = audio_config.window_samples

    tmp_path = Path(tmp_dir) / f"worker_{worker_id:03d}.jsonl"
    with open(tmp_path, "w") as tf:
        for count, i in enumerate(indices):
            clean_path = random.choice(clean_files)
            clean = load_audio(clean_path, sr)
            clean = random_crop(clean, window)

            params = degrader.sample_params()
            degraded = degrader.degrade(clean, params)

            min_len = min(len(clean), len(degraded))
            clean_crop = clean[:min_len]
            degraded_crop = degraded[:min_len]

            apd_score, metadata = compute_apd_label(
                clean_crop, degraded_crop, params, label_config, sr,
            )

            degraded_path = Path(degraded_dir) / f"{i:07d}.wav"
            sf.write(str(degraded_path), degraded_crop, sr)

            entry = {
                "idx": i,
                "clean_path": str(clean_path),
                "degraded_path": str(degraded_path),
                "apd_score": round(apd_score, 4),
                "stoi": round(metadata["stoi"], 4),
                "pesq": round(metadata["pesq"], 4),
                "snr": params.snr,
                "masker_type": params.masker_type,
                "rt60": params.rt60,
                "speech_rate": params.speech_rate,
                "sir": params.sir,
                "n_babble_speakers": params.n_babble_speakers,
            }
            tf.write(json.dumps(entry) + "\n")

            if (count + 1) % 1000 == 0:
                print(f"  [Worker {worker_id}] {count+1}/{len(indices)} generated")

    return str(tmp_path)


def generate_manifest(
    clean_files: list[str],
    noise_files: list[str],
    speaker_files: list[str],
    output_dir,
    manifest_name: str,
    n_samples: int,
    audio_config: AudioConfig = AudioConfig(),
    degradation_config=None,
    label_config: APDLabelConfig = APDLabelConfig(),
    seed: int = 42,
    n_workers: int = 0,
):
    """Generate degraded audio samples and compute pseudo-labels.

    Uses multiprocessing to parallelize across CPU cores.

    Args:
        clean_files: List of paths to clean speech files
        noise_files: List of paths to noise files
        speaker_files: List of paths to speaker files (for competing/babble)
        output_dir: Directory to save degraded audio and manifest
        manifest_name: Name of the manifest file (e.g., "train.jsonl")
        n_samples: Number of samples to generate
        audio_config: Audio configuration
        degradation_config: Degradation parameter ranges
        label_config: APD label configuration
        seed: Random seed
        n_workers: Number of parallel workers (0 = all CPU cores, max 4)
    """
    from .config import DegradationConfig

    if degradation_config is None:
        degradation_config = DegradationConfig()

    output_dir = Path(output_dir)
    degraded_dir = output_dir / "degraded"
    degraded_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / manifest_name

    if n_workers <= 0:
        n_workers = min(os.cpu_count() or 1, 4)

    # Split indices into chunks, one per worker
    all_indices = list(range(n_samples))
    chunks = [all_indices[i::n_workers] for i in range(n_workers)]

    tmp_dir = output_dir / "_tmp_workers"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Each worker gets a different seed derived from the base seed
    worker_args = [
        (
            worker_id,
            chunk,
            clean_files,
            noise_files,
            speaker_files,
            str(degraded_dir),
            str(tmp_dir),
            audio_config,
            degradation_config,
            label_config,
            seed + worker_id,
        )
        for worker_id, chunk in enumerate(chunks)
    ]

    print(f"Processing {n_samples} samples with {n_workers} workers...")

    with mp.Pool(n_workers) as pool:
        tmp_paths = pool.map(_worker_process_chunk, worker_args)

    # Merge temp files: read entries, sort by original index, write manifest
    print("Merging worker outputs...")
    all_entries = []
    for tmp_path in tmp_paths:
        with open(tmp_path) as f:
            for line in f:
                entry = json.loads(line)
                idx = entry.pop("idx")
                all_entries.append((idx, entry))
        os.remove(tmp_path)
    tmp_dir.rmdir()

    all_entries.sort(key=lambda x: x[0])

    with open(manifest_path, "w") as mf:
        for _, entry in all_entries:
            mf.write(json.dumps(entry) + "\n")

    print(f"Manifest saved to {manifest_path} ({n_samples} samples)")
    return manifest_path
