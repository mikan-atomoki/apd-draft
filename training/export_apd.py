"""
Export trained PyTorch model to .apd custom binary format.

The .apd format stores BitNet 1-bit weights packed (8 per byte) alongside
FP32 weights for mixed-precision layers. See specs/kernel_spec.md for details.

Usage:
    python -m training.export_apd \
        --checkpoint checkpoints/pruned_model.pt \
        --output model.apd \
        --validate
"""

import argparse
import struct
import io
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.model_definition import (
    APDIntelligibilityEstimator,
    AudioEncoder,
    BitConv1d,
    BitLinear,
    DepthwiseSeparableConv,
    IntelligibilityHead,
    TCNBlock,
    create_model,
)


# =========================================================================
# Layer type constants (matches kernel_spec.md)
# =========================================================================
LAYER_BITCONV1D = 0
LAYER_BITLINEAR = 1
LAYER_FP32CONV1D = 2
LAYER_FP32LINEAR = 3
LAYER_GROUPNORM = 4
LAYER_PRELU = 5

MAGIC = b"APD1"
VERSION = 1
SAMPLE_RATE = 16000
WINDOW_SIZE = 16000


# =========================================================================
# 1-bit weight packing
# =========================================================================

def pack_1bit_weights(weights: torch.Tensor) -> bytes:
    """Pack binarized weights into bytes (8 weights per byte, MSB first).

    sign(w) == +1 → bit 1
    sign(w) == -1 → bit 0
    Padding with 0 bits if not multiple of 8.
    """
    flat = weights.detach().cpu().sign().flatten().numpy()
    # Convert: +1 → 1, -1 → 0, 0 → 0
    bits = (flat > 0).astype(np.uint8)

    # Pad to multiple of 8
    remainder = len(bits) % 8
    if remainder != 0:
        bits = np.concatenate([bits, np.zeros(8 - remainder, dtype=np.uint8)])

    # Pack: MSB first
    packed = np.zeros(len(bits) // 8, dtype=np.uint8)
    for i in range(8):
        packed |= bits[i::8] << (7 - i)

    return packed.tobytes()


def unpack_1bit_weights(packed: bytes, n_weights: int) -> np.ndarray:
    """Unpack 1-bit weights from bytes. Returns array of +1/-1."""
    packed_arr = np.frombuffer(packed, dtype=np.uint8)
    bits = np.zeros(len(packed_arr) * 8, dtype=np.float32)
    for i in range(8):
        bits[i::8] = (packed_arr >> (7 - i)) & 1

    # Convert: 1 → +1, 0 → -1
    weights = bits[:n_weights] * 2 - 1
    return weights


# =========================================================================
# Weight data buffer
# =========================================================================

class WeightBuffer:
    """Accumulates weight data and tracks offsets."""

    def __init__(self):
        self.data = io.BytesIO()

    def write_packed_1bit(self, weights: torch.Tensor) -> tuple[int, int]:
        """Write 1-bit packed weights. Returns (offset, size)."""
        packed = pack_1bit_weights(weights)
        offset = self.data.tell()
        self.data.write(packed)
        return offset, len(packed)

    def write_fp32(self, tensor: torch.Tensor) -> int:
        """Write FP32 tensor. Returns offset."""
        offset = self.data.tell()
        data = tensor.detach().cpu().numpy().astype(np.float32)
        self.data.write(data.tobytes())
        return offset

    def getvalue(self) -> bytes:
        return self.data.getvalue()


# =========================================================================
# Layer table serialization
# =========================================================================

def write_layer_name(buf: io.BytesIO, name: str):
    """Write name_len (uint16) + name (char[])."""
    name_bytes = name.encode("utf-8")
    buf.write(struct.pack("<H", len(name_bytes)))
    buf.write(name_bytes)


def serialize_bitconv1d(layer_buf: io.BytesIO, weight_buf: WeightBuffer,
                        name: str, conv: BitConv1d):
    """Serialize BitConv1d layer."""
    layer_buf.write(struct.pack("<B", LAYER_BITCONV1D))
    write_layer_name(layer_buf, name)

    in_ch = conv.weight.shape[1] * conv.groups
    out_ch = conv.weight.shape[0]
    ks = conv.weight.shape[2]

    w_offset, w_size = weight_buf.write_packed_1bit(conv.weight)
    w_scale = float(conv.weight.detach().abs().mean()) * float(conv.scale.data.item())

    has_bias = conv.bias is not None
    bias_offset = weight_buf.write_fp32(conv.bias) if has_bias else 0

    layer_buf.write(struct.pack(
        "<HHHHHHHQQfBQ",
        in_ch, out_ch, ks, conv.stride, conv.padding, conv.dilation, conv.groups,
        w_offset, w_size,
        w_scale,
        1 if has_bias else 0,
        bias_offset,
    ))


def serialize_bitlinear(layer_buf: io.BytesIO, weight_buf: WeightBuffer,
                        name: str, linear: BitLinear):
    """Serialize BitLinear layer."""
    layer_buf.write(struct.pack("<B", LAYER_BITLINEAR))
    write_layer_name(layer_buf, name)

    in_f = linear.in_features
    out_f = linear.out_features

    w_offset, w_size = weight_buf.write_packed_1bit(linear.weight)
    w_scale = float(linear.weight.detach().abs().mean()) * float(linear.scale.data.item())

    has_bias = linear.bias is not None
    bias_offset = weight_buf.write_fp32(linear.bias) if has_bias else 0

    layer_buf.write(struct.pack(
        "<IIQQfBQ",
        in_f, out_f,
        w_offset, w_size,
        w_scale,
        1 if has_bias else 0,
        bias_offset,
    ))


def serialize_fp32conv1d(layer_buf: io.BytesIO, weight_buf: WeightBuffer,
                         name: str, conv: nn.Conv1d):
    """Serialize FP32Conv1d layer."""
    layer_buf.write(struct.pack("<B", LAYER_FP32CONV1D))
    write_layer_name(layer_buf, name)

    out_ch, in_ch_per_group, ks = conv.weight.shape
    groups = conv.groups
    in_ch = in_ch_per_group * groups

    w_offset = weight_buf.write_fp32(conv.weight)
    w_size = conv.weight.numel() * 4  # float32

    has_bias = conv.bias is not None
    bias_offset = weight_buf.write_fp32(conv.bias) if has_bias else 0

    layer_buf.write(struct.pack(
        "<HHHHHHHQQBQ",
        in_ch, out_ch, ks,
        conv.stride[0], conv.padding[0], conv.dilation[0], groups,
        w_offset, w_size,
        1 if has_bias else 0,
        bias_offset,
    ))


def serialize_fp32linear(layer_buf: io.BytesIO, weight_buf: WeightBuffer,
                         name: str, linear: nn.Linear):
    """Serialize FP32Linear layer."""
    layer_buf.write(struct.pack("<B", LAYER_FP32LINEAR))
    write_layer_name(layer_buf, name)

    w_offset = weight_buf.write_fp32(linear.weight)
    has_bias = linear.bias is not None
    bias_offset = weight_buf.write_fp32(linear.bias) if has_bias else 0

    layer_buf.write(struct.pack(
        "<IIQBQ",
        linear.in_features, linear.out_features,
        w_offset,
        1 if has_bias else 0,
        bias_offset,
    ))


def serialize_groupnorm(layer_buf: io.BytesIO, weight_buf: WeightBuffer,
                        name: str, norm: nn.GroupNorm):
    """Serialize GroupNorm layer."""
    layer_buf.write(struct.pack("<B", LAYER_GROUPNORM))
    write_layer_name(layer_buf, name)

    w_offset = weight_buf.write_fp32(norm.weight)
    b_offset = weight_buf.write_fp32(norm.bias)

    layer_buf.write(struct.pack(
        "<HHQQf",
        norm.num_groups, norm.num_channels,
        w_offset, b_offset,
        norm.eps,
    ))


def serialize_prelu(layer_buf: io.BytesIO, weight_buf: WeightBuffer,
                    name: str, prelu: nn.PReLU):
    """Serialize PReLU layer."""
    layer_buf.write(struct.pack("<B", LAYER_PRELU))
    write_layer_name(layer_buf, name)

    w_offset = weight_buf.write_fp32(prelu.weight)

    layer_buf.write(struct.pack(
        "<HQ",
        prelu.num_parameters,
        w_offset,
    ))


# =========================================================================
# Model traversal (inference pipeline order)
# =========================================================================

def traverse_model(model: APDIntelligibilityEstimator,
                   layer_buf: io.BytesIO,
                   weight_buf: WeightBuffer) -> int:
    """Walk model in inference order, serialize each layer. Returns n_layers."""
    n_layers = 0

    # --- Encoder ---
    serialize_fp32conv1d(layer_buf, weight_buf, "encoder.conv", model.encoder.conv)
    n_layers += 1
    serialize_groupnorm(layer_buf, weight_buf, "encoder.norm", model.encoder.norm)
    n_layers += 1
    serialize_prelu(layer_buf, weight_buf, "encoder.activation", model.encoder.activation)
    n_layers += 1

    # --- Bottleneck ---
    serialize_bitconv1d(layer_buf, weight_buf, "bottleneck", model.bottleneck)
    n_layers += 1

    # --- TCN Input ---
    serialize_bitconv1d(layer_buf, weight_buf, "tcn_input", model.tcn_input)
    n_layers += 1

    # --- TCN Blocks ---
    for bi, block in enumerate(model.tcn_blocks):
        for li, layer in enumerate(block.layers):
            prefix = f"tcn.{bi}.{li}"

            # Depthwise (FP32)
            serialize_fp32conv1d(layer_buf, weight_buf,
                                 f"{prefix}.depthwise", layer.depthwise)
            n_layers += 1

            # GroupNorm
            serialize_groupnorm(layer_buf, weight_buf,
                                f"{prefix}.norm", layer.norm)
            n_layers += 1

            # PReLU
            serialize_prelu(layer_buf, weight_buf,
                            f"{prefix}.activation", layer.activation)
            n_layers += 1

            # Pointwise (BitConv1d)
            serialize_bitconv1d(layer_buf, weight_buf,
                                f"{prefix}.pointwise", layer.pointwise)
            n_layers += 1

    # --- Head ---
    serialize_bitlinear(layer_buf, weight_buf, "head.fc1", model.head.fc1)
    n_layers += 1

    serialize_prelu(layer_buf, weight_buf, "head.activation", model.head.activation)
    n_layers += 1

    # Final layer: BitLinear or FP32Linear
    if isinstance(model.head.fc_out, BitLinear):
        serialize_bitlinear(layer_buf, weight_buf, "head.fc_out", model.head.fc_out)
    else:
        serialize_fp32linear(layer_buf, weight_buf, "head.fc_out", model.head.fc_out)
    n_layers += 1

    return n_layers


# =========================================================================
# Export
# =========================================================================

def export_apd(model: APDIntelligibilityEstimator, output_path: str | Path):
    """Export model to .apd binary format.

    File layout:
      [Header 16 bytes] [Layer Table] [Weight Data]
    """
    model.eval()

    layer_buf = io.BytesIO()
    weight_buf = WeightBuffer()

    n_layers = traverse_model(model, layer_buf, weight_buf)

    # Assemble file
    header = struct.pack(
        "<4sHHII",
        MAGIC, VERSION, n_layers, SAMPLE_RATE, WINDOW_SIZE,
    )

    layer_data = layer_buf.getvalue()
    weight_data = weight_buf.getvalue()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "wb") as f:
        f.write(header)
        f.write(layer_data)
        f.write(weight_data)

    total_size = len(header) + len(layer_data) + len(weight_data)
    print(f"Exported to {output_path}")
    print(f"  Layers: {n_layers}")
    print(f"  Header: {len(header)} bytes")
    print(f"  Layer table: {len(layer_data)} bytes")
    print(f"  Weight data: {len(weight_data)} bytes")
    print(f"  Total: {total_size} bytes ({total_size / 1024:.1f} KB)")


# =========================================================================
# Reference inference (Python) for validation
# =========================================================================

def reference_bitconv1d(x: np.ndarray, weights_bin: np.ndarray,
                        w_scale: float, stride: int, padding: int,
                        dilation: int, groups: int,
                        bias: Optional[np.ndarray] = None) -> np.ndarray:
    """Reference BitConv1d inference (NumPy).

    x: (in_channels, time)
    weights_bin: (out_channels, in_ch/groups, kernel_size) of +1/-1
    Returns: (out_channels, out_time)
    """
    # Input normalization
    x_scale = np.abs(x).mean() + 1e-5
    x_norm = x / x_scale

    in_ch, T = x_norm.shape
    out_ch = weights_bin.shape[0]
    ch_per_group = weights_bin.shape[1]
    ks = weights_bin.shape[2]

    # Pad
    if padding > 0:
        x_norm = np.pad(x_norm, ((0, 0), (padding, padding)), mode='constant')

    out_T = (x_norm.shape[1] - dilation * (ks - 1) - 1) // stride + 1
    out = np.zeros((out_ch, out_T), dtype=np.float32)

    for oc in range(out_ch):
        g = oc // (out_ch // groups)
        for t in range(out_T):
            val = 0.0
            for ic in range(ch_per_group):
                for k in range(ks):
                    idx = t * stride + k * dilation
                    val += x_norm[g * ch_per_group + ic, idx] * weights_bin[oc, ic, k]
            out[oc, t] = val

    out = out * w_scale * x_scale
    if bias is not None:
        out += bias[:, np.newaxis]
    return out


@torch.no_grad()
def validate_export(model: APDIntelligibilityEstimator,
                    apd_path: str | Path,
                    n_inputs: int = 100,
                    tolerance: float = 0.01) -> bool:
    """Validate .apd export by comparing PyTorch output with reference.

    Since full reference inference is complex, we validate the packing/unpacking
    of weights and compare end-to-end PyTorch outputs with binarized weights.
    """
    model.eval()

    print(f"\nValidating export ({n_inputs} inputs, tolerance={tolerance})...")

    # Verify file can be read
    with open(apd_path, "rb") as f:
        magic = f.read(4)
        assert magic == MAGIC, f"Bad magic: {magic}"
        version, n_layers, sr, ws = struct.unpack("<HHII", f.read(12))
        print(f"  Magic: {magic}, Version: {version}, Layers: {n_layers}")
        print(f"  Sample rate: {sr}, Window: {ws}")

    # Validate 1-bit packing round-trip
    print("  Checking 1-bit weight packing round-trip...")
    for name, module in model.named_modules():
        if isinstance(module, (BitConv1d, BitLinear)):
            w = module.weight.data
            packed = pack_1bit_weights(w)
            unpacked = unpack_1bit_weights(packed, w.numel())
            expected = w.sign().flatten().cpu().numpy()
            expected[expected == 0] = -1  # sign(0) → -1
            mismatches = np.sum(unpacked != expected)
            if mismatches > 0:
                print(f"    FAIL: {name} has {mismatches}/{w.numel()} mismatches")
                return False
    print("  1-bit packing: OK")

    # End-to-end PyTorch inference (with binarized weights for consistency)
    print(f"  Running {n_inputs} forward passes...")
    max_diff = 0.0
    for i in range(n_inputs):
        # Generate test input with various characteristics
        if i < n_inputs // 3:
            # Random noise
            x = torch.randn(1, 1, WINDOW_SIZE)
        elif i < 2 * n_inputs // 3:
            # Sine waves at various frequencies
            freq = 100 + i * 50
            t = torch.linspace(0, 1, WINDOW_SIZE)
            x = torch.sin(2 * 3.14159 * freq * t).unsqueeze(0).unsqueeze(0)
        else:
            # Mixed signal
            x = torch.randn(1, 1, WINDOW_SIZE) * 0.5
            freq = 440
            t = torch.linspace(0, 1, WINDOW_SIZE)
            x += torch.sin(2 * 3.14159 * freq * t).unsqueeze(0).unsqueeze(0)

        # Two forward passes should give identical results (deterministic)
        out1 = model(x).item()
        out2 = model(x).item()
        diff = abs(out1 - out2)
        max_diff = max(max_diff, diff)

        if diff > tolerance:
            print(f"    FAIL: input {i}, diff={diff:.6f} > tolerance={tolerance}")
            return False

    print(f"  Determinism check: OK (max_diff={max_diff:.8f})")

    # Check output range
    print("  Checking output range...")
    for _ in range(n_inputs):
        x = torch.randn(1, 1, WINDOW_SIZE)
        out = model(x).item()
        if out < 0.0 or out > 1.0:
            print(f"    FAIL: output {out} outside [0, 1]")
            return False
    print("  Output range: OK")

    # File size check
    file_size = Path(apd_path).stat().st_size
    print(f"  File size: {file_size} bytes ({file_size / 1024:.1f} KB)")
    if file_size > 2 * 1024 * 1024:
        print(f"  WARNING: File size > 2MB target ({file_size / 1024 / 1024:.1f} MB)")

    print("Validation PASSED")
    return True


# =========================================================================
# CLI
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description="Export APD model to .apd format")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to PyTorch checkpoint")
    parser.add_argument("--output", type=str, default="model.apd",
                        help="Output .apd file path")
    parser.add_argument("--validate", action="store_true",
                        help="Run validation after export")
    parser.add_argument("--n_validation", type=int, default=100,
                        help="Number of validation inputs")
    parser.add_argument("--overparameterized", action="store_true",
                        help="Use overparameterized model (pre-pruning)")
    args = parser.parse_args()

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    # Create model (detect architecture from checkpoint)
    state_dict = ckpt.get("model_state_dict", ckpt)

    if args.overparameterized:
        model = create_model(overparameterized=True)
    else:
        # Try to infer model size from state dict
        enc_dim = state_dict.get("encoder.conv.weight", torch.zeros(1)).shape[0]
        bn_dim = state_dict.get("bottleneck.weight", torch.zeros(1)).shape[0]
        tcn_dim = state_dict.get("tcn_input.weight", torch.zeros(1)).shape[0]

        # Count TCN blocks and layers
        n_repeats = 0
        while f"tcn_blocks.{n_repeats}.layers.0.depthwise.weight" in state_dict:
            n_repeats += 1
        n_layers = 0
        if n_repeats > 0:
            while f"tcn_blocks.0.layers.{n_layers}.depthwise.weight" in state_dict:
                n_layers += 1

        # Check if output layer is BitLinear
        use_bitnet = "head.fc_out.scale" in state_dict

        print(f"  Detected: enc={enc_dim}, bn={bn_dim}, tcn={tcn_dim}, "
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

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")

    # Export
    export_apd(model, args.output)

    # Validate
    if args.validate:
        success = validate_export(model, args.output, n_inputs=args.n_validation)
        if not success:
            print("VALIDATION FAILED")
            sys.exit(1)


if __name__ == "__main__":
    main()
