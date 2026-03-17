"""
Offline data preprocessing: Generate degraded audio + pseudo-labels.

This is the CPU-intensive step (~6-10 hours for 500k samples).
Run once, then train from the generated manifests.

Usage:
    python -m training.preprocess \
        --librispeech_root data/LibriSpeech \
        --demand_root data/DEMAND \
        --output_dir data/manifests \
        --n_train 500000 --n_val 50000 --n_test 50000
"""

import argparse
from pathlib import Path

from training.config import AudioConfig, APDLabelConfig, DegradationConfig
from training.dataset import generate_manifest


def collect_audio_files(root: Path, extensions=(".flac", ".wav", ".mp3")) -> list[str]:
    """Recursively collect audio files from a directory."""
    files = []
    for ext in extensions:
        files.extend(str(p) for p in root.rglob(f"*{ext}"))
    return sorted(files)


def main():
    parser = argparse.ArgumentParser(description="Preprocess data for APD training")
    parser.add_argument("--librispeech_root", type=str, required=True,
                        help="Path to LibriSpeech directory (train-clean-100/360)")
    parser.add_argument("--demand_root", type=str, default=None,
                        help="Path to DEMAND noise dataset")
    parser.add_argument("--dns_noise_root", type=str, default=None,
                        help="Path to DNS Challenge noise set")
    parser.add_argument("--output_dir", type=str, default="data/manifests")
    parser.add_argument("--n_train", type=int, default=500_000)
    parser.add_argument("--n_val", type=int, default=50_000)
    parser.add_argument("--n_test", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect files
    print("Collecting audio files...")
    ls_root = Path(args.librispeech_root)
    clean_files = collect_audio_files(ls_root)
    print(f"  Clean speech files: {len(clean_files)}")

    # Noise files
    noise_files = []
    if args.demand_root:
        noise_files.extend(collect_audio_files(Path(args.demand_root)))
    if args.dns_noise_root:
        noise_files.extend(collect_audio_files(Path(args.dns_noise_root)))
    print(f"  Noise files: {len(noise_files)}")

    # Speaker files for competing/babble (use a subset of LibriSpeech)
    speaker_files = clean_files  # reuse clean speech as interference
    print(f"  Speaker files (for interference): {len(speaker_files)}")

    if not clean_files:
        print("ERROR: No clean speech files found!")
        return

    if not noise_files:
        print("WARNING: No noise files found. Only 'none' and speaker-based maskers will be used.")
        # Use clean files as noise fallback for testing
        noise_files = clean_files[:100]

    audio_config = AudioConfig()
    degradation_config = DegradationConfig()
    label_config = APDLabelConfig()

    # Generate splits
    for split, n_samples, seed_offset in [
        ("train.jsonl", args.n_train, 0),
        ("val.jsonl", args.n_val, 1000000),
        ("test.jsonl", args.n_test, 2000000),
    ]:
        print(f"\n{'='*60}")
        print(f"Generating {split} ({n_samples} samples)...")
        print(f"{'='*60}")

        generate_manifest(
            clean_files=clean_files,
            noise_files=noise_files,
            speaker_files=speaker_files,
            output_dir=output_dir,
            manifest_name=split,
            n_samples=n_samples,
            audio_config=audio_config,
            degradation_config=degradation_config,
            label_config=label_config,
            seed=args.seed + seed_offset,
        )

    print("\nDone! Manifests saved to:", output_dir)


if __name__ == "__main__":
    main()
