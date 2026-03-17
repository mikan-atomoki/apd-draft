"""
Dataset classes for APD Intelligibility Estimator training.

Two-phase approach:
  1. Offline preprocessing: Generate degraded audio + pseudo-labels → manifest (JSONLines)
  2. Online training: Load from manifest + light augmentation
"""

import json
import multiprocessing as mp
import os
import random
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

from .augmentation import (
    AudioDegrader,
    DegradationParams,
    apply_gain,
    apply_mixup,
    apply_shift,
    load_audio,
    random_crop,
)
from .config import AugmentationConfig, AudioConfig
from .pseudo_label import compute_apd_label, APDLabelConfig


# =========================================================================
# Manifest entry format (one line of JSONLines)
# =========================================================================
# {
#   "clean_path": "...",
#   "degraded_path": "...",
#   "apd_score": 0.42,
#   "stoi": 0.75,
#   "pesq": 0.61,
#   "snr": 5.0,
#   "masker_type": "babble_multi",
#   "rt60": 0.8,
#   "speech_rate": 1.0,
#   "sir": null,
#   "n_babble_speakers": 6
# }


class APDManifestDataset(Dataset):
    """Load precomputed samples from manifest file.

    Each sample is a (degraded_audio, apd_score) pair.
    Optional online augmentation (gain, shift, mixup).
    """

    def __init__(
        self,
        manifest_path: str | Path,
        audio_config: AudioConfig = AudioConfig(),
        augmentation: Optional[AugmentationConfig] = None,
        return_metadata: bool = False,
    ):
        self.audio_config = audio_config
        self.augmentation = augmentation
        self.return_metadata = return_metadata

        # Load manifest
        self.entries = []
        with open(manifest_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.entries.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> dict:
        entry = self.entries[idx]

        # Load degraded audio
        audio = load_audio(entry["degraded_path"], self.audio_config.sample_rate)
        audio = random_crop(audio, self.audio_config.window_samples)
        audio = torch.from_numpy(audio).float()
        label = float(entry["apd_score"])

        # Online augmentation
        if self.augmentation is not None:
            audio = apply_gain(audio, self.augmentation.gain_db_range)
            audio = apply_shift(audio, self.augmentation.shift_ms_range,
                                self.audio_config.sample_rate)

        # Shape: (1, window_samples) for model input
        audio = audio.unsqueeze(0)

        result = {"audio": audio, "label": torch.tensor(label, dtype=torch.float32)}

        if self.return_metadata:
            result["metadata"] = {
                "stoi": entry.get("stoi"),
                "pesq": entry.get("pesq"),
                "snr": entry.get("snr"),
                "masker_type": entry.get("masker_type"),
                "rt60": entry.get("rt60"),
                "speech_rate": entry.get("speech_rate"),
            }

        return result


def collate_fn(batch: list[dict]) -> dict:
    """Custom collate for APD dataset."""
    audios = torch.stack([b["audio"] for b in batch])
    labels = torch.stack([b["label"] for b in batch])
    result = {"audio": audios, "label": labels}

    if "metadata" in batch[0]:
        result["metadata"] = [b["metadata"] for b in batch]

    return result


def collate_with_mixup(batch: list[dict], alpha: float = 0.2,
                       prob: float = 0.3) -> dict:
    """Collate with mixup augmentation applied to random pairs."""
    audios = []
    labels = []

    items = list(batch)
    for i, b in enumerate(items):
        if random.random() < prob and len(items) > 1:
            j = random.choice([k for k in range(len(items)) if k != i])
            mixed_audio, mixed_label = apply_mixup(
                b["audio"], b["label"].item(),
                items[j]["audio"], items[j]["label"].item(),
                alpha=alpha,
            )
            audios.append(mixed_audio)
            labels.append(torch.tensor(mixed_label, dtype=torch.float32))
        else:
            audios.append(b["audio"])
            labels.append(b["label"])

    return {
        "audio": torch.stack(audios),
        "label": torch.stack(labels),
    }


# =========================================================================
# Offline preprocessing: Generate manifest from raw data
# =========================================================================

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

    Uses multiprocessing to parallelize across all CPU cores.

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
        n_workers: Number of parallel workers (0 = all CPU cores)
    """
    from .config import DegradationConfig

    if degradation_config is None:
        degradation_config = DegradationConfig()

    output_dir = Path(output_dir)
    degraded_dir = output_dir / "degraded"
    degraded_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / manifest_name

    if n_workers <= 0:
        n_workers = os.cpu_count() or 1

    # Split indices into contiguous chunks, one per worker
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
