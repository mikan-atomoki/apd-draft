"""
APD pseudo-label generation.

Computes intelligibility scores reflecting APD (Auditory Processing Disorder)
perception characteristics. Since no ground-truth APD intelligibility data
exists, we use a hybrid approach:

  1. STOI/PESQ as baseline (normal hearing intelligibility)
  2. APD-specific penalty factors (multiplicative model)

The multiplicative model reflects research findings that reverb × noise
interact synergistically for APD listeners (not additively).
"""

import numpy as np

from .augmentation import DegradationParams
from .config import APDLabelConfig


def compute_stoi_score(clean: np.ndarray, degraded: np.ndarray,
                       sr: int = 16000) -> float:
    """Compute STOI (Short-Time Objective Intelligibility)."""
    from pystoi import stoi
    return float(stoi(clean, degraded, sr, extended=False))


def compute_pesq_score(clean: np.ndarray, degraded: np.ndarray,
                       sr: int = 16000) -> float:
    """Compute PESQ and normalize to 0-1 range."""
    from pesq import pesq
    # PESQ range: -0.5 to 4.5 (wideband mode for 16kHz)
    raw = float(pesq(sr, clean, degraded, "wb"))
    # Normalize: [-0.5, 4.5] → [0.0, 1.0]
    return float(np.clip((raw + 0.5) / 5.0, 0.0, 1.0))


def sigmoid_map(snr: float, center: float, slope: float) -> float:
    """Map SNR to a penalty factor via sigmoid.

    Returns ~1.0 for high SNR, ~0.3 for low SNR.
    The sigmoid is shifted by APD SNR loss.
    """
    # Effective SNR for APD listener (shifted down)
    x = slope * (snr - center)
    factor = 1.0 / (1.0 + np.exp(-x))
    # Scale to [0.3, 1.0] range (APD can still partially understand in noise)
    return float(0.3 + 0.7 * factor)


def compute_reverb_factor(rt60: float, cfg: APDLabelConfig) -> float:
    """Compute reverb penalty factor for APD.

    - RT60 <= 0.4s: optimal, no penalty
    - RT60 0.4-0.7s: mild degradation
    - RT60 > 0.7s: rapid degradation (temporal smearing × APD temporal deficit)
    """
    if rt60 is None or rt60 <= cfg.rt60_optimal:
        return 1.0
    elif rt60 <= cfg.rt60_moderate:
        return 0.95 - 0.1 * (rt60 - cfg.rt60_optimal) / (cfg.rt60_moderate - cfg.rt60_optimal)
    else:
        return 0.85 - 0.35 * min((rt60 - cfg.rt60_moderate) / 1.3, 1.0)


def compute_rate_factor(speech_rate: float) -> float:
    """Compute speech rate penalty for APD.

    Fast speech is particularly problematic due to temporal resolution deficits.
    - <= 1.0x: no penalty
    - 1.0-1.2x: mild
    - > 1.2x: rapid degradation
    """
    if speech_rate is None or speech_rate <= 1.0:
        return 1.0
    elif speech_rate <= 1.2:
        return 1.0 - 0.15 * (speech_rate - 1.0) / 0.2
    else:
        return 0.85 - 0.35 * min((speech_rate - 1.2) / 0.3, 1.0)


def compute_apd_label(
    clean: np.ndarray,
    degraded: np.ndarray,
    params: DegradationParams,
    cfg: APDLabelConfig = APDLabelConfig(),
    sr: int = 16000,
    precomputed_stoi: float | None = None,
    precomputed_pesq: float | None = None,
) -> tuple[float, dict]:
    """Compute APD pseudo-label for a degraded audio sample.

    Args:
        clean: Clean reference audio (float32, mono)
        degraded: Degraded audio (float32, mono)
        params: Degradation parameters used to create degraded audio
        cfg: APD label configuration
        sr: Sample rate
        precomputed_stoi: Skip STOI computation if provided
        precomputed_pesq: Skip PESQ computation if provided

    Returns:
        (apd_score, metadata_dict)
        apd_score: float in [0, 1], APD-perspective intelligibility
        metadata_dict: intermediate values for debugging/analysis
    """
    # Phase 1: Baseline intelligibility (normal hearing)
    stoi_val = precomputed_stoi if precomputed_stoi is not None else compute_stoi_score(clean, degraded, sr)
    pesq_val = precomputed_pesq if precomputed_pesq is not None else compute_pesq_score(clean, degraded, sr)
    base_score = cfg.stoi_weight * stoi_val + cfg.pesq_weight * pesq_val

    # Phase 2: APD SNR penalty
    if params.snr is not None:
        snr_penalty = sigmoid_map(params.snr, cfg.snr_sigmoid_center, cfg.snr_sigmoid_slope)
    else:
        snr_penalty = 1.0

    # Phase 3: Masker type penalty
    masker_penalty = cfg.masker_penalties.get(params.masker_type, 1.0)

    # Phase 4: Reverb penalty
    reverb_factor = compute_reverb_factor(params.rt60, cfg)

    # Phase 5: Speech rate penalty
    rate_factor = compute_rate_factor(params.speech_rate)

    # Phase 6: Multiplicative integration
    apd_score = base_score * snr_penalty * masker_penalty * reverb_factor * rate_factor
    apd_score = float(np.clip(apd_score, 0.0, 1.0))

    # Label smoothing
    apd_score += float(np.random.normal(0, cfg.label_noise_sigma))
    apd_score = float(np.clip(apd_score, 0.0, 1.0))

    metadata = {
        "stoi": stoi_val,
        "pesq": pesq_val,
        "base_score": base_score,
        "snr_penalty": snr_penalty,
        "masker_penalty": masker_penalty,
        "reverb_factor": reverb_factor,
        "rate_factor": rate_factor,
        "apd_score_pre_noise": float(np.clip(
            base_score * snr_penalty * masker_penalty * reverb_factor * rate_factor,
            0.0, 1.0,
        )),
    }

    return apd_score, metadata
