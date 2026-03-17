"""
Download datasets for APD training.

Usage:
    # Download all datasets
    python download_data.py

    # Download specific datasets
    python download_data.py --only librispeech
    python download_data.py --only demand
    python download_data.py --only dns

    # Smaller LibriSpeech subset (for testing)
    python download_data.py --small
"""

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path
from urllib.request import urlretrieve


DATA_DIR = Path("data")

DATASETS = {
    "librispeech-100": {
        "url": "https://www.openslr.org/resources/12/train-clean-100.tar.gz",
        "dest": DATA_DIR / "LibriSpeech",
        "description": "LibriSpeech train-clean-100 (~6.3GB)",
    },
    "librispeech-360": {
        "url": "https://www.openslr.org/resources/12/train-clean-360.tar.gz",
        "dest": DATA_DIR / "LibriSpeech",
        "description": "LibriSpeech train-clean-360 (~23GB)",
    },
    "demand": {
        "url": "https://zenodo.org/records/1227121/files/DEMAND.zip",
        "dest": DATA_DIR / "DEMAND",
        "description": "DEMAND noise dataset (~2.4GB)",
    },
}


class ProgressReporter:
    def __init__(self, desc: str):
        self.desc = desc
        self.last_pct = -1

    def __call__(self, block_num, block_size, total_size):
        if total_size <= 0:
            return
        pct = int(block_num * block_size * 100 / total_size)
        pct = min(pct, 100)
        if pct != self.last_pct:
            bar = "#" * (pct // 2) + "-" * (50 - pct // 2)
            print(f"\r  [{bar}] {pct}%  {self.desc}", end="", flush=True)
            self.last_pct = pct
            if pct == 100:
                print()


def download_file(url: str, dest_path: Path, desc: str) -> Path:
    """Download a file with progress bar."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists():
        print(f"  Already downloaded: {dest_path}")
        return dest_path

    print(f"  Downloading: {url}")
    urlretrieve(url, str(dest_path), ProgressReporter(desc))
    return dest_path


def extract_tar_gz(archive: Path, dest: Path):
    """Extract .tar.gz archive."""
    print(f"  Extracting {archive.name}...")
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(path=dest.parent)
    print(f"  Extracted to {dest}")


def extract_zip(archive: Path, dest: Path):
    """Extract .zip archive."""
    print(f"  Extracting {archive.name}...")
    with zipfile.ZipFile(archive, "r") as z:
        z.extractall(path=dest.parent)
    print(f"  Extracted to {dest}")


def count_audio_files(path: Path) -> int:
    """Count audio files in directory."""
    count = 0
    for ext in (".flac", ".wav", ".mp3", ".ogg"):
        count += len(list(path.rglob(f"*{ext}")))
    return count


def download_librispeech(small: bool = False):
    """Download LibriSpeech datasets."""
    cache_dir = DATA_DIR / "_downloads"
    cache_dir.mkdir(parents=True, exist_ok=True)

    subsets = ["librispeech-100"]
    if not small:
        subsets.append("librispeech-360")

    for key in subsets:
        info = DATASETS[key]
        fname = info["url"].split("/")[-1]
        archive = cache_dir / fname

        print(f"\n--- {info['description']} ---")

        # Check if already extracted
        expected_dir = info["dest"]
        if expected_dir.exists() and count_audio_files(expected_dir) > 100:
            print(f"  Already exists: {expected_dir} ({count_audio_files(expected_dir)} audio files)")
            continue

        download_file(info["url"], archive, info["description"])
        extract_tar_gz(archive, expected_dir)

    dest = DATASETS["librispeech-100"]["dest"]
    n = count_audio_files(dest)
    print(f"\nLibriSpeech ready: {dest} ({n} audio files)")


def download_demand():
    """Download DEMAND noise dataset."""
    info = DATASETS["demand"]
    cache_dir = DATA_DIR / "_downloads"
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n--- {info['description']} ---")

    dest = info["dest"]
    if dest.exists() and count_audio_files(dest) > 10:
        print(f"  Already exists: {dest} ({count_audio_files(dest)} audio files)")
        return

    archive = cache_dir / "DEMAND.zip"
    download_file(info["url"], archive, info["description"])
    extract_zip(archive, dest)

    n = count_audio_files(dest)
    print(f"\nDEMAND ready: {dest} ({n} audio files)")


def download_dns():
    """Download DNS Challenge noise clips.

    The DNS Challenge dataset is large and hosted on Azure Blob Storage.
    We download just the noise clips subset via their provided script.
    """
    dest = DATA_DIR / "dns_noise"

    print(f"\n--- DNS Challenge noise set ---")

    if dest.exists() and count_audio_files(dest) > 100:
        print(f"  Already exists: {dest} ({count_audio_files(dest)} audio files)")
        return

    dest.mkdir(parents=True, exist_ok=True)

    # DNS Challenge provides data via Azure Blob + azcopy
    # Check if azcopy is available
    has_azcopy = shutil.which("azcopy") is not None

    if not has_azcopy:
        print("  WARNING: azcopy not found. DNS Challenge data requires azcopy to download.")
        print("  Install: https://learn.microsoft.com/en-us/azure/storage/common/storage-use-azcopy-v10")
        print()
        print("  Alternatively, manually download noise clips from:")
        print("    https://github.com/microsoft/DNS-Challenge")
        print(f"  and place them in: {dest}")
        print()
        print("  Skipping DNS noise download. Pipeline will still work with")
        print("  LibriSpeech + DEMAND (just fewer noise categories).")
        return

    # DNS-Challenge 5 noise clips blob URL
    blob_url = "https://dnschallengepublic.blob.core.windows.net/dns5archive/V5_noise.tar.bz2"

    cache_dir = DATA_DIR / "_downloads"
    archive = cache_dir / "V5_noise.tar.bz2"

    print(f"  Downloading DNS noise clips (~15GB)...")
    try:
        subprocess.run(
            ["azcopy", "copy", blob_url, str(archive)],
            check=True,
        )
    except subprocess.CalledProcessError:
        print("  azcopy download failed. Try manual download from:")
        print(f"    {blob_url}")
        return

    print(f"  Extracting...")
    subprocess.run(
        ["tar", "xjf", str(archive), "-C", str(dest)],
        check=True,
    )

    n = count_audio_files(dest)
    print(f"\nDNS noise ready: {dest} ({n} audio files)")


def print_summary():
    """Print summary of available data."""
    print(f"\n{'='*50}")
    print("  Data Summary")
    print(f"{'='*50}")

    for name, path in [
        ("LibriSpeech", DATA_DIR / "LibriSpeech"),
        ("DEMAND", DATA_DIR / "DEMAND"),
        ("DNS noise", DATA_DIR / "dns_noise"),
    ]:
        if path.exists():
            n = count_audio_files(path)
            size_mb = sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1024 / 1024
            print(f"  {name:15s} {n:>6} files  {size_mb:>8.0f} MB  {path}")
        else:
            print(f"  {name:15s}    (not found)  {path}")

    print()
    print("Next step:")
    print("  python run_pipeline.py --librispeech_root data/LibriSpeech --demand_root data/DEMAND")


def main():
    parser = argparse.ArgumentParser(description="Download datasets for APD training")
    parser.add_argument("--only", type=str, choices=["librispeech", "demand", "dns"],
                        help="Download only this dataset")
    parser.add_argument("--small", action="store_true",
                        help="Download smaller subset (train-clean-100 only, skip 360)")
    parser.add_argument("--data_dir", type=str, default="data",
                        help="Base directory for data")
    args = parser.parse_args()

    global DATA_DIR
    DATA_DIR = Path(args.data_dir)
    # Update DATASETS paths
    for key in DATASETS:
        DATASETS[key]["dest"] = DATA_DIR / DATASETS[key]["dest"].name

    print("APD Training Data Downloader")
    print(f"Data directory: {DATA_DIR.resolve()}")

    if args.only == "librispeech":
        download_librispeech(small=args.small)
    elif args.only == "demand":
        download_demand()
    elif args.only == "dns":
        download_dns()
    else:
        download_librispeech(small=args.small)
        download_demand()
        download_dns()

    print_summary()


if __name__ == "__main__":
    main()
