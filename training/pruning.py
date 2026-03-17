"""
Structured channel pruning for APD Intelligibility Estimator.

Strategy: Iterative magnitude-based channel pruning
  4 rounds × 15% channel removal → 10 epochs fine-tuning per round
  Then full retraining (25 epochs) with knowledge distillation.

Key challenge: Pruning a channel requires coordinated removal across
all dependent layers (Conv weights, GroupNorm, PReLU parameters).
"""

import copy
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.model_definition import (
    APDIntelligibilityEstimator,
    AudioEncoder,
    BitConv1d,
    DepthwiseSeparableConv,
    IntelligibilityHead,
    TCNBlock,
    BitLinear,
)


# =========================================================================
# Channel importance scoring
# =========================================================================

def compute_channel_importance(module: nn.Module) -> torch.Tensor:
    """Compute L1-norm based importance score per output channel.

    For Conv layers: importance[c] = ||W[c, :, :]||_1
    """
    if isinstance(module, (nn.Conv1d, BitConv1d)):
        # weight shape: (out_channels, in_channels/groups, kernel_size)
        return module.weight.data.abs().sum(dim=(1, 2))
    elif isinstance(module, (nn.Linear, BitLinear)):
        # weight shape: (out_features, in_features)
        return module.weight.data.abs().sum(dim=1)
    else:
        raise ValueError(f"Unsupported module type: {type(module)}")


def get_channels_to_prune(importance: torch.Tensor,
                          ratio: float) -> list[int]:
    """Get indices of channels to prune (lowest importance)."""
    n_prune = max(1, int(len(importance) * ratio))
    # Keep at least 25% of channels
    n_keep = max(len(importance) // 4, len(importance) - n_prune)
    n_prune = len(importance) - n_keep

    _, indices = importance.sort()
    return sorted(indices[:n_prune].tolist())


def get_channels_to_keep(importance: torch.Tensor,
                         ratio: float) -> list[int]:
    """Get indices of channels to keep."""
    prune_idx = set(get_channels_to_prune(importance, ratio))
    return [i for i in range(len(importance)) if i not in prune_idx]


# =========================================================================
# Layer-wise pruning operations
# =========================================================================

def prune_conv1d_output(conv: nn.Conv1d, keep_idx: list[int]) -> nn.Conv1d:
    """Prune output channels of a Conv1d layer."""
    new_out = len(keep_idx)
    new_conv = nn.Conv1d(
        conv.in_channels, new_out, conv.kernel_size[0],
        stride=conv.stride[0], padding=conv.padding[0],
        dilation=conv.dilation[0], groups=conv.groups if conv.groups == 1 else new_out,
        bias=conv.bias is not None,
    )
    new_conv.weight.data = conv.weight.data[keep_idx]
    if conv.bias is not None:
        new_conv.bias.data = conv.bias.data[keep_idx]
    return new_conv


def prune_conv1d_input(conv: nn.Conv1d, keep_idx: list[int]) -> nn.Conv1d:
    """Prune input channels of a Conv1d (non-grouped)."""
    new_in = len(keep_idx)
    new_conv = nn.Conv1d(
        new_in, conv.out_channels, conv.kernel_size[0],
        stride=conv.stride[0], padding=conv.padding[0],
        dilation=conv.dilation[0], groups=conv.groups,
        bias=conv.bias is not None,
    )
    new_conv.weight.data = conv.weight.data[:, keep_idx]
    if conv.bias is not None:
        new_conv.bias.data = conv.bias.data.clone()
    return new_conv


def prune_bitconv1d_output(conv: BitConv1d, keep_idx: list[int]) -> BitConv1d:
    """Prune output channels of BitConv1d."""
    new_out = len(keep_idx)
    in_ch = conv.weight.shape[1] * conv.groups
    new_conv = BitConv1d(
        in_ch, new_out, conv.weight.shape[2],
        stride=conv.stride, padding=conv.padding,
        dilation=conv.dilation,
        groups=1 if conv.groups == 1 else new_out,
        bias=conv.bias is not None,
    )
    new_conv.weight.data = conv.weight.data[keep_idx]
    new_conv.scale.data = conv.scale.data.clone()
    if conv.bias is not None:
        new_conv.bias.data = conv.bias.data[keep_idx]
    return new_conv


def prune_bitconv1d_input(conv: BitConv1d, keep_idx: list[int]) -> BitConv1d:
    """Prune input channels of BitConv1d (non-grouped)."""
    new_in = len(keep_idx)
    new_conv = BitConv1d(
        new_in, conv.weight.shape[0], conv.weight.shape[2],
        stride=conv.stride, padding=conv.padding,
        dilation=conv.dilation, groups=conv.groups,
        bias=conv.bias is not None,
    )
    new_conv.weight.data = conv.weight.data[:, keep_idx]
    new_conv.scale.data = conv.scale.data.clone()
    if conv.bias is not None:
        new_conv.bias.data = conv.bias.data.clone()
    return new_conv


def prune_groupnorm(norm: nn.GroupNorm, keep_idx: list[int]) -> nn.GroupNorm:
    """Prune GroupNorm channels."""
    new_ch = len(keep_idx)
    # Adjust num_groups: keep it 1 (LayerNorm style) or find valid divisor
    num_groups = 1  # safest choice
    new_norm = nn.GroupNorm(num_groups, new_ch, eps=norm.eps)
    if norm.weight is not None:
        new_norm.weight.data = norm.weight.data[keep_idx]
    if norm.bias is not None:
        new_norm.bias.data = norm.bias.data[keep_idx]
    return new_norm


def prune_prelu(prelu: nn.PReLU, keep_idx: list[int]) -> nn.PReLU:
    """Prune PReLU per-channel parameters."""
    if prelu.num_parameters == 1:
        return copy.deepcopy(prelu)
    new_prelu = nn.PReLU(num_parameters=len(keep_idx))
    new_prelu.weight.data = prelu.weight.data[keep_idx]
    return new_prelu


def prune_bitlinear_input(linear: BitLinear, keep_idx: list[int]) -> BitLinear:
    """Prune input features of BitLinear."""
    new_in = len(keep_idx)
    new_linear = BitLinear(new_in, linear.out_features, bias=linear.bias is not None)
    new_linear.weight.data = linear.weight.data[:, keep_idx]
    new_linear.scale.data = linear.scale.data.clone()
    if linear.bias is not None:
        new_linear.bias.data = linear.bias.data.clone()
    return new_linear


# =========================================================================
# Structured pruning of the full model
# =========================================================================

def prune_encoder(encoder: AudioEncoder, keep_idx: list[int]) -> AudioEncoder:
    """Prune encoder output channels."""
    new_dim = len(keep_idx)
    new_encoder = AudioEncoder(
        in_channels=1, encoder_dim=new_dim,
        kernel_size=encoder.conv.kernel_size[0],
        stride=encoder.conv.stride[0],
    )
    new_encoder.conv = prune_conv1d_output(encoder.conv, keep_idx)
    new_encoder.norm = prune_groupnorm(encoder.norm, keep_idx)
    new_encoder.activation = prune_prelu(encoder.activation, keep_idx)
    return new_encoder


def prune_dsc(dsc: DepthwiseSeparableConv, keep_idx: list[int]) -> DepthwiseSeparableConv:
    """Prune DepthwiseSeparableConv channels (depthwise + pointwise + norm + prelu)."""
    new_ch = len(keep_idx)
    new_dsc = DepthwiseSeparableConv.__new__(DepthwiseSeparableConv)
    nn.Module.__init__(new_dsc)

    # Depthwise: both in and out channels are pruned (groups=channels)
    dw = dsc.depthwise
    new_dw = nn.Conv1d(
        new_ch, new_ch, dw.kernel_size[0],
        padding=dw.padding[0], dilation=dw.dilation[0],
        groups=new_ch, bias=dw.bias is not None,
    )
    new_dw.weight.data = dw.weight.data[keep_idx]
    if dw.bias is not None:
        new_dw.bias.data = dw.bias.data[keep_idx]
    new_dsc.depthwise = new_dw

    new_dsc.norm = prune_groupnorm(dsc.norm, keep_idx)
    new_dsc.activation = prune_prelu(dsc.activation, keep_idx)

    # Pointwise: prune both input and output
    new_dsc.pointwise = prune_bitconv1d_output(
        prune_bitconv1d_input(dsc.pointwise, keep_idx), keep_idx
    )

    return new_dsc


def prune_tcn_block(block: TCNBlock, keep_idx: list[int]) -> TCNBlock:
    """Prune all DSC layers in a TCN block."""
    new_block = TCNBlock.__new__(TCNBlock)
    nn.Module.__init__(new_block)
    new_block.layers = nn.ModuleList([
        prune_dsc(layer, keep_idx) for layer in block.layers
    ])
    return new_block


def prune_model(
    model: APDIntelligibilityEstimator,
    prune_ratio: float = 0.15,
) -> APDIntelligibilityEstimator:
    """Prune the full model by removing channels.

    Pruning is coordinated across all layers that share the same channel dimension.
    Three independent channel groups:
      1. encoder_dim: encoder output → bottleneck input
      2. bottleneck_dim: bottleneck output → tcn_input input
      3. tcn_channels: tcn_input output → TCN blocks → head input
    """
    # --- Group 1: encoder_dim ---
    enc_importance = compute_channel_importance(model.encoder.conv)
    enc_keep = get_channels_to_keep(enc_importance, prune_ratio)

    # --- Group 2: bottleneck_dim ---
    bn_importance = compute_channel_importance(model.bottleneck)
    bn_keep = get_channels_to_keep(bn_importance, prune_ratio)

    # --- Group 3: tcn_channels ---
    # Aggregate importance across all pointwise convs in TCN
    tcn_ch = model.tcn_input.weight.shape[0]
    tcn_importance = torch.zeros(tcn_ch)
    tcn_importance += compute_channel_importance(model.tcn_input)
    for block in model.tcn_blocks:
        for layer in block.layers:
            tcn_importance += compute_channel_importance(layer.pointwise)
    tcn_keep = get_channels_to_keep(tcn_importance, prune_ratio)

    # --- Build pruned model ---
    new_model = APDIntelligibilityEstimator.__new__(APDIntelligibilityEstimator)
    nn.Module.__init__(new_model)

    # Encoder
    new_model.encoder = prune_encoder(model.encoder, enc_keep)

    # Bottleneck: input=encoder_dim, output=bottleneck_dim
    pruned_bn = prune_bitconv1d_input(model.bottleneck, enc_keep)
    new_model.bottleneck = prune_bitconv1d_output(pruned_bn, bn_keep)

    # TCN input: input=bottleneck_dim, output=tcn_channels
    pruned_tcn_in = prune_bitconv1d_input(model.tcn_input, bn_keep)
    new_model.tcn_input = prune_bitconv1d_output(pruned_tcn_in, tcn_keep)

    # TCN blocks
    new_model.tcn_blocks = nn.ModuleList([
        prune_tcn_block(block, tcn_keep) for block in model.tcn_blocks
    ])

    # Head: input=tcn_channels
    new_head = IntelligibilityHead.__new__(IntelligibilityHead)
    nn.Module.__init__(new_head)
    new_head.fc1 = prune_bitlinear_input(model.head.fc1, tcn_keep)
    new_head.activation = copy.deepcopy(model.head.activation)
    new_head.dropout = copy.deepcopy(model.head.dropout)
    new_head.fc_out = copy.deepcopy(model.head.fc_out)
    new_head.sigmoid = copy.deepcopy(model.head.sigmoid)
    new_model.head = new_head

    return new_model


# =========================================================================
# Knowledge distillation loss
# =========================================================================

def distillation_loss(
    student_out: torch.Tensor,
    teacher_out: torch.Tensor,
    targets: torch.Tensor,
    temperature: float = 4.0,
    alpha: float = 0.7,
) -> torch.Tensor:
    """Combined distillation + task loss.

    For regression (not classification), we use MSE between
    softened outputs as distillation signal.
    """
    # Task loss: MSE with ground truth
    task_loss = nn.functional.mse_loss(student_out, targets)

    # Distillation loss: MSE with teacher (no temperature scaling needed for regression)
    distill_loss = nn.functional.mse_loss(student_out, teacher_out.detach())

    return alpha * distill_loss + (1 - alpha) * task_loss


# =========================================================================
# Iterative pruning pipeline
# =========================================================================

def iterative_prune(
    model: APDIntelligibilityEstimator,
    train_loader,
    val_loader,
    device: str = "cuda",
    n_rounds: int = 4,
    prune_ratio: float = 0.15,
    finetune_epochs: int = 10,
    lr: float = 1e-4,
    grad_clip: float = 5.0,
) -> APDIntelligibilityEstimator:
    """Iterative pruning with fine-tuning between rounds.

    Args:
        model: Trained (unpruned) model
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Device to use
        n_rounds: Number of pruning rounds
        prune_ratio: Fraction of channels to prune per round
        finetune_epochs: Epochs of fine-tuning after each prune
        lr: Learning rate for fine-tuning
        grad_clip: Gradient clipping max norm

    Returns:
        Pruned and fine-tuned model
    """
    from training.loss import APDLoss

    teacher = copy.deepcopy(model).to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    current = model.to(device)
    criterion = APDLoss()

    for round_idx in range(n_rounds):
        # Count params before
        n_before = sum(p.numel() for p in current.parameters())

        # Prune
        current = prune_model(current, prune_ratio).to(device)

        n_after = sum(p.numel() for p in current.parameters())
        print(
            f"Prune round {round_idx+1}/{n_rounds}: "
            f"{n_before:,} -> {n_after:,} params "
            f"({100*(1 - n_after/n_before):.1f}% reduction)"
        )

        # Fine-tune with distillation
        optimizer = torch.optim.AdamW(current.parameters(), lr=lr)

        for epoch in range(finetune_epochs):
            current.train()
            total_loss = 0.0
            n = 0

            for batch in train_loader:
                audio = batch["audio"].to(device)
                labels = batch["label"].to(device)

                student_out = current(audio).squeeze(-1)
                with torch.no_grad():
                    teacher_out = teacher(audio).squeeze(-1)

                loss = distillation_loss(student_out, teacher_out, labels)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(current.parameters(), grad_clip)
                optimizer.step()

                total_loss += loss.item()
                n += 1

            avg_loss = total_loss / max(n, 1)
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"  Fine-tune epoch {epoch+1}/{finetune_epochs}: loss={avg_loss:.4f}")

    print(f"\nPruning complete. Final params: {sum(p.numel() for p in current.parameters()):,}")
    return current


def main():
    """Standalone pruning script."""
    import argparse
    from functools import partial
    from torch.utils.data import DataLoader
    from training.config import Config
    from training.dataset import APDManifestDataset, collate_fn

    parser = argparse.ArgumentParser(description="Prune APD model")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--manifest_dir", type=str, default="data/manifests")
    parser.add_argument("--output", type=str, default="checkpoints/pruned_model.pt")
    parser.add_argument("--n_rounds", type=int, default=4)
    parser.add_argument("--prune_ratio", type=float, default=0.15)
    parser.add_argument("--finetune_epochs", type=int, default=10)
    parser.add_argument("--retrain_epochs", type=int, default=25)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    cfg = Config()

    # Load model
    ckpt = torch.load(args.checkpoint, map_location=args.device)
    from model.model_definition import create_model
    model = create_model(overparameterized=True)
    model.load_state_dict(ckpt["model_state_dict"])

    # Data loaders
    train_dataset = APDManifestDataset(
        Path(args.manifest_dir) / "train.jsonl", audio_config=cfg.audio,
    )
    val_dataset = APDManifestDataset(
        Path(args.manifest_dir) / "val.jsonl", audio_config=cfg.audio,
    )
    train_loader = DataLoader(
        train_dataset, batch_size=32, shuffle=True,
        num_workers=4, collate_fn=collate_fn, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=64, shuffle=False,
        num_workers=4, collate_fn=collate_fn, pin_memory=True,
    )

    # Prune
    pruned = iterative_prune(
        model, train_loader, val_loader,
        device=args.device,
        n_rounds=args.n_rounds,
        prune_ratio=args.prune_ratio,
        finetune_epochs=args.finetune_epochs,
    )

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": pruned.state_dict(),
        "model_config": {
            "encoder_dim": pruned.encoder.conv.out_channels,
            "bottleneck_dim": pruned.bottleneck.weight.shape[0],
            "tcn_channels": pruned.tcn_input.weight.shape[0],
        },
    }, output_path)
    print(f"Pruned model saved to {output_path}")


if __name__ == "__main__":
    main()
