"""
Audio degradation and augmentation for APD training data generation.

Degradation pipeline:
  clean audio → add noise (various masker types) → add reverb → adjust speed
  → compute STOI/PESQ → compute APD pseudo-label

Online augmentation (during training):
  gain jitter, time shift, mixup
"""

import random
from dataclasses import dataclass
from typing import Optional

import math

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from .config import AudioConfig, DegradationConfig


@dataclass
class DegradationParams:
    """Parameters describing how audio was degraded."""
    snr: Optional[float] = None         # dB
    masker_type: str = "none"           # none/stationary/modulated/babble_multi/competing_1_2
    rt60: Optional[float] = None        # seconds
    speech_rate: Optional[float] = None # multiplier (1.0 = normal)
    sir: Optional[float] = None         # dB (for competing speakers)
    n_babble_speakers: Optional[int] = None


def load_audio(path: str, sr: int = 16000) -> np.ndarray:
    """Load audio file and resample to target sr. Returns float32 mono."""
    audio, orig_sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if orig_sr != sr:
        gcd = math.gcd(sr, orig_sr)
        audio = resample_poly(audio, sr // gcd, orig_sr // gcd).astype(np.float32)
    return audio


def random_crop(audio: np.ndarray, length: int) -> np.ndarray:
    """Random crop to exact length. Pad with zeros if too short."""
    if len(audio) >= length:
        start = random.randint(0, len(audio) - length)
        return audio[start:start + length]
    else:
        padded = np.zeros(length, dtype=np.float32)
        padded[:len(audio)] = audio
        return padded


def mix_at_snr(signal: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """Mix signal and noise at specified SNR."""
    sig_power = np.mean(signal ** 2) + 1e-10
    noise_power = np.mean(noise ** 2) + 1e-10
    target_noise_power = sig_power / (10 ** (snr_db / 10))
    scale = np.sqrt(target_noise_power / noise_power)
    return signal + noise * scale


def apply_rir(audio: np.ndarray, rir: np.ndarray) -> np.ndarray:
    """Convolve audio with room impulse response."""
    # Normalize RIR so direct sound has unit gain
    rir = rir / (np.abs(rir).max() + 1e-10)
    convolved = np.convolve(audio, rir, mode="full")[:len(audio)]
    return convolved.astype(np.float32)


def generate_rir(rt60: float, room_dim: Optional[list] = None,
                 sr: int = 16000) -> np.ndarray:
    """Generate synthetic RIR using pyroomacoustics."""
    import pyroomacoustics as pra

    # Clamp RT60 to physically reasonable range
    rt60 = max(rt60, 0.1)

    for _attempt in range(10):
        if room_dim is None:
            # Random room: 3-10m each dimension
            dims = [
                random.uniform(3.0, 10.0),
                random.uniform(3.0, 8.0),
                random.uniform(2.5, 4.0),
            ]
        else:
            dims = list(room_dim)

        try:
            e_absorption, max_order = pra.inverse_sabine(rt60, dims)
        except ValueError:
            # RT60 too small for this room size — try a smaller room
            room_dim = None  # retry with new random room
            rt60 = max(rt60, 0.15)
            continue

        if max_order > 50:
            max_order = 50

        room = pra.ShoeBox(
            dims, fs=sr,
            materials=pra.Material(e_absorption),
            max_order=max_order,
        )

        def rand_pos(d):
            margin = 0.5
            return [random.uniform(margin, di - margin) for di in d]

        room.add_source(rand_pos(dims))
        room.add_microphone(rand_pos(dims))
        room.compute_rir()

        return room.rir[0][0].astype(np.float32)

    # All retries failed — return a simple impulse (no reverb)
    rir = np.zeros(sr // 10, dtype=np.float32)
    rir[0] = 1.0
    return rir


def change_speed(audio: np.ndarray, rate: float, sr: int = 16000) -> np.ndarray:
    """Change speech rate using resampling (simple approach)."""
    if abs(rate - 1.0) < 0.01:
        return audio
    # Resample to change speed: rate > 1 = faster
    new_sr = int(sr * rate)
    gcd = math.gcd(new_sr, sr)
    return resample_poly(audio, new_sr // gcd, sr // gcd).astype(np.float32)


class AudioDegrader:
    """Applies degradation to clean audio for pseudo-label generation."""

    def __init__(
        self,
        noise_files: list[str],
        speaker_files: list[str],
        config: DegradationConfig,
        audio_config: AudioConfig,
    ):
        self.noise_files = noise_files
        self.speaker_files = speaker_files
        self.config = config
        self.audio_config = audio_config
        self.sr = audio_config.sample_rate
        self.window = audio_config.window_samples

    def sample_params(self) -> DegradationParams:
        """Sample random degradation parameters."""
        cfg = self.config
        params = DegradationParams()

        # Masker type
        types = list(cfg.masker_weights.keys())
        weights = [cfg.masker_weights[t] for t in types]
        params.masker_type = random.choices(types, weights=weights, k=1)[0]

        # SNR (None for 'none' masker)
        if params.masker_type != "none":
            params.snr = random.uniform(*cfg.snr_range)

        # SIR for competing speakers
        if params.masker_type == "competing_1_2":
            params.sir = random.uniform(*cfg.sir_range)
            params.snr = params.sir  # use SIR as effective SNR

        # Babble speakers count
        if params.masker_type == "babble_multi":
            params.n_babble_speakers = random.randint(*cfg.babble_speakers_range)

        # Reverb (50% chance of adding reverb)
        if random.random() < 0.5:
            params.rt60 = random.uniform(*cfg.rt60_range)

        # Speech rate (30% chance of modifying)
        if random.random() < 0.3:
            params.speech_rate = random.uniform(*cfg.speech_rate_range)

        return params

    def degrade(self, clean: np.ndarray, params: DegradationParams) -> np.ndarray:
        """Apply degradation to clean audio according to params."""
        audio = clean.copy()

        # 1. Speed change (before mixing, affects clean signal)
        if params.speech_rate is not None:
            audio = change_speed(audio, params.speech_rate, self.sr)

        # Ensure correct length after speed change
        audio = random_crop(audio, self.window)

        # 2. Add reverb
        if params.rt60 is not None and params.rt60 > 0.05:
            rir = generate_rir(params.rt60, sr=self.sr)
            audio = apply_rir(audio, rir)

        # 3. Add noise/masker
        if params.masker_type == "stationary" or params.masker_type == "modulated":
            noise = self._load_random_noise()
            audio = mix_at_snr(audio, noise, params.snr)

        elif params.masker_type == "competing_1_2":
            n_speakers = random.choice([1, 2])
            interference = np.zeros(self.window, dtype=np.float32)
            for _ in range(n_speakers):
                spk = self._load_random_speaker()
                interference += spk / n_speakers
            audio = mix_at_snr(audio, interference, params.sir)

        elif params.masker_type == "babble_multi":
            n = params.n_babble_speakers or 4
            babble = np.zeros(self.window, dtype=np.float32)
            for _ in range(n):
                spk = self._load_random_speaker()
                babble += spk / n
            audio = mix_at_snr(audio, babble, params.snr)

        # Normalize to prevent clipping
        peak = np.abs(audio).max()
        if peak > 0.99:
            audio = audio * 0.99 / peak

        return audio

    def _load_random_noise(self) -> np.ndarray:
        path = random.choice(self.noise_files)
        noise = load_audio(path, self.sr)
        return random_crop(noise, self.window)

    def _load_random_speaker(self) -> np.ndarray:
        path = random.choice(self.speaker_files)
        spk = load_audio(path, self.sr)
        return random_crop(spk, self.window)


# =========================================================================
# Online augmentation (applied during training, after loading from cache)
# torch is imported lazily — these functions are only called during
# training, not in preprocessing workers.
# =========================================================================


def apply_gain(audio, db_range: tuple[float, float]):
    """Random gain in dB."""
    gain_db = random.uniform(*db_range)
    return audio * (10 ** (gain_db / 20))


def apply_shift(audio, ms_range: tuple[float, float], sr: int = 16000):
    """Random circular time shift."""
    import torch
    max_shift = int(abs(ms_range[1]) * sr / 1000)
    shift = random.randint(-max_shift, max_shift)
    return torch.roll(audio, shift, dims=-1)


def apply_mixup(audio1, label1: float, audio2, label2: float,
                alpha: float = 0.2):
    """Mixup augmentation."""
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    audio = lam * audio1 + (1 - lam) * audio2
    label = lam * label1 + (1 - lam) * label2
    return audio, label
