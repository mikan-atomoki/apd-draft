"""
APD Intelligibility Estimator - Full Training Pipeline

Usage:
    # Full pipeline (download -> preprocess -> train -> prune -> export)
    python run_pipeline.py

    # Full pipeline with small dataset (for testing)
    python run_pipeline.py --small

    # Skip download (data already exists)
    python run_pipeline.py --skip_download

    # Skip download + preprocessing (manifests already exist)
    python run_pipeline.py --skip_download --skip_preprocess

    # From pruning onward
    python run_pipeline.py --skip_download --skip_preprocess --skip_train --checkpoint checkpoints/best_model.pt

    # Export only
    python run_pipeline.py --export_only --checkpoint checkpoints/pruned_model.pt
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def check_dependencies():
    """Check all required packages are installed, auto-install if missing."""
    required = [
        ("torch", "torch"),
        ("torchaudio", "torchaudio"),
        ("numpy", "numpy"),
        ("scipy", "scipy"),
        ("soundfile", "soundfile"),
        ("pystoi", "pystoi"),
        ("pyroomacoustics", "pyroomacoustics"),
        # pesq is optional (requires C compiler, often fails on Windows)
    ]
    missing = []
    for import_name, pip_name in required:
        try:
            __import__(import_name)
        except ImportError:
            missing.append(pip_name)

    if missing:
        print(f"Missing packages: {', '.join(missing)}")
        print(f"Installing from requirements.txt...")
        req_file = Path(__file__).resolve().parent / "requirements.txt"
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", str(req_file),
        ])
        print()

    # Optional: pesq
    try:
        __import__("pesq")
    except ImportError:
        print("NOTE: pesq not installed (needs C compiler). Using STOI-only fallback for labels.")
        print("      To install: pip install pesq")
        print()


def fmt_elapsed(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    elif m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


def step_banner(step: int, total: int, title: str):
    print(f"\n{'='*60}")
    print(f"  Step {step}/{total}: {title}")
    print(f"{'='*60}\n")


def run_download(args):
    from download_data import download_librispeech, download_demand, download_dns, print_summary
    small = getattr(args, "small", False)
    download_librispeech(small=small)
    download_demand()
    download_dns()
    print_summary()

    # Auto-set data paths if not explicitly provided
    if args.librispeech_root == "data/LibriSpeech":
        ls = Path("data/LibriSpeech")
        if ls.exists():
            args.librispeech_root = str(ls)
    if args.demand_root is None:
        demand = Path("data/DEMAND")
        if demand.exists():
            args.demand_root = str(demand)
    if args.dns_noise_root is None:
        dns = Path("data/dns_noise")
        if dns.exists():
            args.dns_noise_root = str(dns)


def run_preprocess(args):
    from training.config import AudioConfig, APDLabelConfig, DegradationConfig
    from training.manifest import generate_manifest
    from training.preprocess import collect_audio_files

    # Auto-detect data paths if not set (e.g. when --skip_download)
    if args.demand_root is None:
        demand = Path("data/DEMAND")
        if demand.exists():
            args.demand_root = str(demand)
    if args.dns_noise_root is None:
        dns = Path("data/dns_noise")
        if dns.exists():
            args.dns_noise_root = str(dns)

    output_dir = Path(args.manifest_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Collecting audio files...")
    ls_root = Path(args.librispeech_root)
    clean_files = collect_audio_files(ls_root)
    print(f"  Clean speech: {len(clean_files)} files")

    noise_files = []
    if args.demand_root:
        noise_files.extend(collect_audio_files(Path(args.demand_root)))
    if args.dns_noise_root:
        noise_files.extend(collect_audio_files(Path(args.dns_noise_root)))
    print(f"  Noise: {len(noise_files)} files")

    if not clean_files:
        print("ERROR: No clean speech files found!")
        sys.exit(1)

    if not noise_files:
        print("WARNING: No noise files. Using clean speech as noise fallback.")
        noise_files = clean_files[:100]

    speaker_files = clean_files

    audio_config = AudioConfig()
    degradation_config = DegradationConfig()
    label_config = APDLabelConfig()

    for split, n, seed_offset in [
        ("train.jsonl", args.n_train, 0),
        ("val.jsonl", args.n_val, 1000000),
        ("test.jsonl", args.n_test, 2000000),
    ]:
        print(f"\nGenerating {split} ({n} samples)...")
        t0 = time.time()
        generate_manifest(
            clean_files=clean_files,
            noise_files=noise_files,
            speaker_files=speaker_files,
            output_dir=output_dir,
            manifest_name=split,
            n_samples=n,
            audio_config=audio_config,
            degradation_config=degradation_config,
            label_config=label_config,
            seed=args.seed + seed_offset,
        )
        print(f"  Done in {fmt_elapsed(time.time() - t0)}")


def run_train(args):
    from training.config import Config
    from training.train import train

    cfg = Config()
    cfg.data.manifest_dir = Path(args.manifest_dir)
    cfg.train.checkpoint_dir = Path(args.checkpoint_dir)
    cfg.train.batch_size = args.batch_size
    cfg.train.epochs = args.epochs
    cfg.train.lr = args.lr
    cfg.train.device = args.device
    cfg.train.seed = args.seed

    model, history = train(cfg)
    return model


def run_prune(args):
    import torch
    from functools import partial
    from torch.utils.data import DataLoader

    from model.model_definition import create_model
    from training.config import Config
    from training.dataset import APDManifestDataset, collate_fn
    from training.pruning import iterative_prune

    cfg = Config()

    # Load trained model
    ckpt_path = args.checkpoint or str(Path(args.checkpoint_dir) / "best_model.pt")
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=args.device, weights_only=False)

    model = create_model(overparameterized=True)
    model.load_state_dict(ckpt["model_state_dict"])

    n_before = sum(p.numel() for p in model.parameters())
    print(f"Pre-pruning params: {n_before:,}")

    # Data loaders
    train_dataset = APDManifestDataset(
        Path(args.manifest_dir) / "train.jsonl", audio_config=cfg.audio,
    )
    val_dataset = APDManifestDataset(
        Path(args.manifest_dir) / "val.jsonl", audio_config=cfg.audio,
    )
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=4, collate_fn=collate_fn, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size * 2, shuffle=False,
        num_workers=4, collate_fn=collate_fn, pin_memory=True,
    )

    # Prune
    pruned = iterative_prune(
        model, train_loader, val_loader,
        device=args.device,
        n_rounds=args.prune_rounds,
        prune_ratio=args.prune_ratio,
        finetune_epochs=args.prune_finetune_epochs,
    )

    n_after = sum(p.numel() for p in pruned.parameters())
    print(f"Post-pruning params: {n_after:,} ({100*(1 - n_after/n_before):.1f}% reduction)")

    # Save
    output_path = Path(args.checkpoint_dir) / "pruned_model.pt"
    torch.save({
        "model_state_dict": pruned.state_dict(),
        "model_config": {
            "encoder_dim": pruned.encoder.conv.out_channels,
            "bottleneck_dim": pruned.bottleneck.weight.shape[0],
            "tcn_channels": pruned.tcn_input.weight.shape[0],
        },
    }, output_path)
    print(f"Saved to {output_path}")

    return pruned


def run_export(args):
    import torch
    from model.model_definition import APDIntelligibilityEstimator, create_model
    from training.export_apd import export_apd, validate_export

    # Load checkpoint
    ckpt_path = args.checkpoint or str(Path(args.checkpoint_dir) / "pruned_model.pt")
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)

    # Detect model architecture from state dict
    model_config = ckpt.get("model_config", None)
    if model_config:
        enc_dim = model_config["encoder_dim"]
        bn_dim = model_config["bottleneck_dim"]
        tcn_dim = model_config["tcn_channels"]
    else:
        enc_dim = state_dict["encoder.conv.weight"].shape[0]
        bn_dim = state_dict["bottleneck.weight"].shape[0]
        tcn_dim = state_dict["tcn_input.weight"].shape[0]

    n_repeats = 0
    while f"tcn_blocks.{n_repeats}.layers.0.depthwise.weight" in state_dict:
        n_repeats += 1
    n_layers = 0
    if n_repeats > 0:
        while f"tcn_blocks.0.layers.{n_layers}.depthwise.weight" in state_dict:
            n_layers += 1

    use_bitnet = "head.fc_out.scale" in state_dict

    print(f"  Architecture: enc={enc_dim}, bn={bn_dim}, tcn={tcn_dim}, "
          f"repeats={n_repeats}, layers={n_layers}, bitnet_out={use_bitnet}")

    model = APDIntelligibilityEstimator(
        encoder_dim=enc_dim,
        bottleneck_dim=bn_dim,
        tcn_channels=tcn_dim,
        n_repeats=n_repeats,
        n_layers=n_layers,
        use_bitnet_output=use_bitnet,
    )
    model.load_state_dict(state_dict)
    model.eval()

    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Export
    output_path = Path(args.output_apd)
    export_apd(model, output_path)

    # Validate
    if args.validate:
        success = validate_export(model, output_path, n_inputs=args.n_validation)
        if not success:
            print("VALIDATION FAILED")
            sys.exit(1)

    print(f"\nExported: {output_path}")


def main():
    check_dependencies()

    parser = argparse.ArgumentParser(
        description="APD Intelligibility Estimator - Full Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline (download + preprocess + train + prune + export)
  python run_pipeline.py

  # Quick test with small dataset
  python run_pipeline.py --small

  # Data already downloaded
  python run_pipeline.py --skip_download

  # Resume from pruning
  python run_pipeline.py --skip_download --skip_preprocess --skip_train --checkpoint checkpoints/best_model.pt

  # Export only
  python run_pipeline.py --export_only --checkpoint checkpoints/pruned_model.pt
        """,
    )

    # Pipeline control
    pipe = parser.add_argument_group("Pipeline control")
    pipe.add_argument("--skip_download", action="store_true",
                      help="Skip data download")
    pipe.add_argument("--skip_preprocess", action="store_true",
                      help="Skip data preprocessing")
    pipe.add_argument("--small", action="store_true",
                      help="Use small dataset (LibriSpeech train-clean-100 only)")
    pipe.add_argument("--skip_train", action="store_true",
                      help="Skip training")
    pipe.add_argument("--skip_prune", action="store_true",
                      help="Skip pruning")
    pipe.add_argument("--export_only", action="store_true",
                      help="Only run export step")

    # Data
    data = parser.add_argument_group("Data")
    data.add_argument("--librispeech_root", type=str, default="data/LibriSpeech")
    data.add_argument("--demand_root", type=str, default=None)
    data.add_argument("--dns_noise_root", type=str, default=None)
    data.add_argument("--manifest_dir", type=str, default="data/manifests")
    data.add_argument("--n_train", type=int, default=500_000)
    data.add_argument("--n_val", type=int, default=50_000)
    data.add_argument("--n_test", type=int, default=50_000)

    # Training
    train = parser.add_argument_group("Training")
    train.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    train.add_argument("--batch_size", type=int, default=32)
    train.add_argument("--epochs", type=int, default=50)
    train.add_argument("--lr", type=float, default=3e-4)
    train.add_argument("--device", type=str, default="cuda")
    train.add_argument("--seed", type=int, default=42)

    # Pruning
    prune = parser.add_argument_group("Pruning")
    prune.add_argument("--prune_rounds", type=int, default=4)
    prune.add_argument("--prune_ratio", type=float, default=0.15)
    prune.add_argument("--prune_finetune_epochs", type=int, default=10)

    # Export
    export = parser.add_argument_group("Export")
    export.add_argument("--output_apd", type=str, default="model.apd")
    export.add_argument("--validate", action="store_true", default=True)
    export.add_argument("--no_validate", dest="validate", action="store_false")
    export.add_argument("--n_validation", type=int, default=100)

    # Shared
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Explicit checkpoint path (overrides auto-detection)")

    args = parser.parse_args()

    if args.export_only:
        args.skip_download = True
        args.skip_preprocess = True
        args.skip_train = True
        args.skip_prune = True

    # Determine steps
    steps = []
    if not getattr(args, "skip_download", False):
        steps.append(("Download", run_download))
    if not args.skip_preprocess:
        steps.append(("Preprocess", run_preprocess))
    if not args.skip_train:
        steps.append(("Train", run_train))
    if not args.skip_prune:
        steps.append(("Prune", run_prune))
    steps.append(("Export", run_export))

    total = len(steps)
    pipeline_start = time.time()

    print(f"APD Pipeline: {' -> '.join(name for name, _ in steps)}")
    print(f"Device: {args.device}")
    print()

    for i, (name, fn) in enumerate(steps, 1):
        step_banner(i, total, name)
        t0 = time.time()
        fn(args)
        print(f"\n  [{name}] completed in {fmt_elapsed(time.time() - t0)}")

    print(f"\n{'='*60}")
    print(f"  Pipeline complete! Total time: {fmt_elapsed(time.time() - pipeline_start)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
