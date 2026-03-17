"""
Dataset classes for APD Intelligibility Estimator training.

Two-phase approach:
  1. Offline preprocessing: Generate degraded audio + pseudo-labels → manifest (JSONLines)
  2. Online training: Load from manifest + light augmentation
"""

import json
import random
from pathlib import Path
from typing import Optional

import numpy as np

from .augmentation import (
    apply_gain,
    apply_mixup,
    apply_shift,
    load_audio,
    random_crop,
)
from .config import AugmentationConfig, AudioConfig


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


import torch
from torch.utils.data import Dataset

from .manifest import generate_manifest  # noqa: F401 — re-export


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
