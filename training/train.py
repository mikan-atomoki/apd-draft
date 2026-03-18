"""
Training loop for APD Intelligibility Estimator.

Usage:
    python -m training.train --manifest_dir data/manifests --checkpoint_dir checkpoints
"""

import argparse
import json
import math
import random
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.model_definition import create_model
from training.config import AugmentationConfig, AudioConfig, Config, TrainConfig
from training.dataset import APDManifestDataset, collate_fn, collate_with_mixup
from training.loss import APDLoss


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_cosine_schedule_with_warmup(optimizer, warmup_steps: int,
                                     total_steps: int, min_lr: float = 1e-6):
    """Cosine decay with linear warmup."""
    base_lrs = [pg["lr"] for pg in optimizer.param_groups]

    def lr_lambda(step, base_lr):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(min_lr / base_lr, cosine)

    schedulers = []
    for i, pg in enumerate(optimizer.param_groups):
        schedulers.append(
            torch.optim.lr_scheduler.LambdaLR(
                optimizer, partial(lr_lambda, base_lr=base_lrs[i])
            )
        )
    # Return the first one (all param groups share same schedule)
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer, partial(lr_lambda, base_lr=base_lrs[0])
    )


@torch.no_grad()
def evaluate(model: nn.Module, dataloader: DataLoader, criterion: APDLoss,
             device: str) -> dict:
    """Evaluate model on validation set."""
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0
    n_batches = 0

    for batch in dataloader:
        audio = batch["audio"].to(device)
        labels = batch["label"].to(device)

        preds = model(audio).squeeze(-1)
        loss, _ = criterion(preds, labels)

        total_loss += loss.item()
        n_batches += 1

        all_preds.extend(preds.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    spearman_corr, _ = spearmanr(all_preds, all_labels)
    mae = np.mean(np.abs(all_preds - all_labels))

    return {
        "loss": total_loss / max(n_batches, 1),
        "spearman": spearman_corr,
        "mae": mae,
        "pred_mean": float(all_preds.mean()),
        "pred_std": float(all_preds.std()),
        "label_mean": float(all_labels.mean()),
        "label_std": float(all_labels.std()),
    }


def log_bitnet_stats(model: nn.Module) -> dict:
    """Log BitNet layer statistics for monitoring STE training stability."""
    stats = {}
    for name, module in model.named_modules():
        if hasattr(module, "scale") and hasattr(module, "binarize"):
            w = module.weight.data
            stats[name] = {
                "w_scale": float(module.scale.data.item()),
                "w_abs_mean": float(w.abs().mean().item()),
                "w_std": float(w.std().item()),
            }
    return stats


def train(cfg: Config):
    """Main training loop."""
    set_seed(cfg.train.seed)
    device = cfg.train.device

    # Create model
    model = create_model(overparameterized=True, use_bitnet_output=True)
    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Datasets
    train_manifest = Path(cfg.data.manifest_dir) / cfg.data.train_manifest
    val_manifest = Path(cfg.data.manifest_dir) / cfg.data.val_manifest

    train_dataset = APDManifestDataset(
        train_manifest,
        audio_config=cfg.audio,
        augmentation=cfg.augmentation,
    )
    val_dataset = APDManifestDataset(
        val_manifest,
        audio_config=cfg.audio,
        augmentation=None,  # no augmentation for validation
    )

    mixup_collate = partial(
        collate_with_mixup,
        alpha=cfg.augmentation.mixup_alpha,
        prob=cfg.augmentation.mixup_prob,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        prefetch_factor=cfg.data.prefetch_factor,
        collate_fn=mixup_collate,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.train.batch_size * 2,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        prefetch_factor=cfg.data.prefetch_factor,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # Loss
    criterion = APDLoss(
        ranking_weight=cfg.train.ranking_loss_weight,
        boundary_weight=cfg.train.boundary_loss_weight,
        boundary_thresholds=cfg.train.boundary_thresholds,
        boundary_sigma=cfg.train.boundary_sigma,
    ).to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
    )

    # Scheduler
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * cfg.train.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, cfg.train.warmup_steps, total_steps, cfg.train.min_lr,
    )

    # Checkpointing
    ckpt_dir = Path(cfg.train.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Training state
    best_spearman = -1.0
    patience_counter = 0
    global_step = 0
    history = []

    print(f"Training for {cfg.train.epochs} epochs, {steps_per_epoch} steps/epoch")
    print(f"Total steps: {total_steps}, Warmup: {cfg.train.warmup_steps}")

    for epoch in range(cfg.train.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_components = {"mse": 0.0, "ranking": 0.0, "boundary": 0.0}
        n_batches = 0
        t0 = time.time()

        for batch in train_loader:
            audio = batch["audio"].to(device)
            labels = batch["label"].to(device)

            # Forward
            preds = model(audio).squeeze(-1)
            loss, components = criterion(preds, labels)

            # Backward
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping (important for BitNet STE stability)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg.train.grad_clip_max_norm,
            )

            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            for k in epoch_components:
                epoch_components[k] += components[k]
            n_batches += 1
            global_step += 1

            # Logging
            if global_step % cfg.train.log_every_steps == 0:
                lr = optimizer.param_groups[0]["lr"]
                avg_loss = epoch_loss / n_batches
                print(
                    f"  step {global_step} | loss={avg_loss:.4f} "
                    f"(mse={epoch_components['mse']/n_batches:.4f} "
                    f"rank={epoch_components['ranking']/n_batches:.4f} "
                    f"bound={epoch_components['boundary']/n_batches:.4f}) "
                    f"| lr={lr:.2e}"
                )

        epoch_time = time.time() - t0
        avg_epoch_loss = epoch_loss / max(n_batches, 1)

        # Validation
        val_metrics = evaluate(model, val_loader, criterion, device)
        bitnet_stats = log_bitnet_stats(model)

        print(
            f"Epoch {epoch+1}/{cfg.train.epochs} "
            f"| train_loss={avg_epoch_loss:.4f} "
            f"| val_loss={val_metrics['loss']:.4f} "
            f"| spearman={val_metrics['spearman']:.4f} "
            f"| mae={val_metrics['mae']:.4f} "
            f"| time={epoch_time:.0f}s"
        )

        # History
        epoch_record = {
            "epoch": epoch + 1,
            "train_loss": avg_epoch_loss,
            "val_loss": val_metrics["loss"],
            "spearman": val_metrics["spearman"],
            "mae": val_metrics["mae"],
            "lr": optimizer.param_groups[0]["lr"],
            "bitnet_stats": bitnet_stats,
        }
        history.append(epoch_record)

        # Save history
        with open(ckpt_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

        # Checkpointing
        is_best = val_metrics["spearman"] > best_spearman
        if is_best:
            best_spearman = val_metrics["spearman"]
            patience_counter = 0
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_spearman": best_spearman,
                "config": cfg,
            }, ckpt_dir / "best_model.pt")
            print(f"  -> New best model (spearman={best_spearman:.4f})")
        else:
            patience_counter += 1

        if (epoch + 1) % cfg.train.save_every_epochs == 0:
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_spearman": best_spearman,
            }, ckpt_dir / f"checkpoint_epoch{epoch+1}.pt")

        # Early stopping
        if patience_counter >= cfg.train.patience:
            print(f"Early stopping at epoch {epoch+1} (patience={cfg.train.patience})")
            break

    print(f"\nTraining complete. Best Spearman: {best_spearman:.4f}")
    return model, history


def main():
    parser = argparse.ArgumentParser(description="Train APD Intelligibility Estimator")
    parser.add_argument("--manifest_dir", type=str, default="data/manifests")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    args = parser.parse_args()

    cfg = Config()
    cfg.data.manifest_dir = Path(args.manifest_dir)
    cfg.train.checkpoint_dir = Path(args.checkpoint_dir)
    cfg.train.batch_size = args.batch_size
    cfg.train.epochs = args.epochs
    cfg.train.lr = args.lr
    cfg.train.device = args.device
    cfg.train.seed = args.seed

    if args.resume:
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=args.device)
        model = create_model(overparameterized=True)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"  Loaded epoch {ckpt['epoch']}, best_spearman={ckpt.get('best_spearman', 'N/A')}")

    train(cfg)


if __name__ == "__main__":
    main()
