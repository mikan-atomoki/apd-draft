"""
APD Speech Intelligibility Estimator
=====================================
APD当事者にとっての音声了解度をリアルタイム推定するモデル。

Architecture: Conv-TasNet based encoder-estimator
Input: 16kHz mono audio (1ch, 16000 samples = 1sec window)
Output: Speech intelligibility score (float, 0.0 - 1.0)
        0.0 = APD当事者にとって聴き取り不可能
        1.0 = 問題なく聴き取れる

Quantization: BitNet (1-bit weights) on most layers
              Exceptions (FP32):
                - Encoder 1st layer (raw waveform input)
                - Depthwise conv (too few weights per filter)
                - Final projection (optional, configurable)
"""

import torch
import torch.nn as nn


# =============================================================================
# BitNet Linear Layer
# =============================================================================
class BitLinear(nn.Module):
    """1-bit weight linear layer (BitNet).

    学習時: STE (Straight-Through Estimator) でFP勾配を保持
    推論時: 重みは {-1, +1} のみ → 乗算不要、加減算のみ
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.scale = nn.Parameter(torch.ones(1))

    def binarize(self, w: torch.Tensor) -> torch.Tensor:
        """STE binarization: forward は sign, backward は identity."""
        return w.sign().detach() + w - w.detach()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 入力の absmax 正規化
        x_scale = x.abs().mean(dim=-1, keepdim=True).clamp(min=1e-5)
        x_norm = x / x_scale

        # 重みの binarize
        w_bin = self.binarize(self.weight)
        w_scale = self.weight.abs().mean()

        out = nn.functional.linear(x_norm, w_bin, self.bias)
        return out * w_scale * x_scale * self.scale


class BitConv1d(nn.Module):
    """1-bit weight 1D convolution (BitNet variant for conv layers)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
    ):
        super().__init__()
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels // groups, kernel_size)
        )
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.scale = nn.Parameter(torch.ones(1))

    def binarize(self, w: torch.Tensor) -> torch.Tensor:
        return w.sign().detach() + w - w.detach()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_scale = x.abs().mean(dim=(1, 2), keepdim=True).clamp(min=1e-5)
        x_norm = x / x_scale

        w_bin = self.binarize(self.weight)
        w_scale = self.weight.abs().mean()

        out = nn.functional.conv1d(
            x_norm, w_bin, self.bias,
            stride=self.stride, padding=self.padding,
            dilation=self.dilation, groups=self.groups,
        )
        return out * w_scale * x_scale * self.scale


# =============================================================================
# Encoder: Waveform → Latent Representation
# =============================================================================
class AudioEncoder(nn.Module):
    """1D conv encoder: raw waveform → latent features.

    過パラメータ設計: 隠れ層を大きめに取り、プルーニングで削る前提。
    """

    def __init__(
        self,
        in_channels: int = 1,
        encoder_dim: int = 512,      # 過パラメータ (最終的にプルーニングで256程度に)
        kernel_size: int = 40,        # 2.5ms @ 16kHz
        stride: int = 20,            # 1.25ms hop
    ):
        super().__init__()
        # 第1層は生波形を直接受けるためFP32を維持 (BitNetだと精度劣化が大きい)
        self.conv = nn.Conv1d(
            in_channels, encoder_dim, kernel_size,
            stride=stride, padding=kernel_size // 2, bias=False,
        )
        self.norm = nn.GroupNorm(1, encoder_dim)
        self.activation = nn.PReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, 1, time) → (batch, encoder_dim, frames)"""
        return self.activation(self.norm(self.conv(x)))


# =============================================================================
# Separation / Feature Extraction Block (TCN)
# =============================================================================
class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable conv.

    Depthwise: FP32 (kernel_size個しか重みがなく、BitNetだと表現力が消える)
    Pointwise: BitNet (チャネル間混合は重みが多いので1-bit化に耐える)
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
    ):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        # Depthwise: FP32維持 (1フィルタあたりkernel_size個の重みしかない)
        self.depthwise = nn.Conv1d(
            channels, channels, kernel_size,
            padding=padding, dilation=dilation, groups=channels, bias=False,
        )
        # Pointwise: BitNet適用 (channels×channels個の重みがあり量子化耐性あり)
        self.pointwise = BitConv1d(channels, channels, 1)
        self.norm = nn.GroupNorm(1, channels)
        self.activation = nn.PReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.activation(self.norm(self.depthwise(x)))
        out = self.pointwise(out)
        return out + residual


class TCNBlock(nn.Module):
    """Temporal Convolutional Network block.

    Conv-TasNet style: exponentially increasing dilation.
    """

    def __init__(self, channels: int, n_layers: int = 8, kernel_size: int = 3):
        super().__init__()
        self.layers = nn.ModuleList([
            DepthwiseSeparableConv(
                channels, kernel_size, dilation=2**i,
            )
            for i in range(n_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


# =============================================================================
# Intelligibility Estimator Head
# =============================================================================
class IntelligibilityHead(nn.Module):
    """Global pooling → score regression.

    NOTE: 最終出力層はBitNetだと精度劣化リスクがあるため、
          FP32 fallback オプションを用意。
    """

    def __init__(
        self,
        in_features: int,
        hidden_dim: int = 256,
        use_bitnet_output: bool = True,
    ):
        super().__init__()
        self.fc1 = BitLinear(in_features, hidden_dim)
        self.activation = nn.PReLU()
        self.dropout = nn.Dropout(0.1)

        # 最終層: BitNet or FP32
        if use_bitnet_output:
            self.fc_out = BitLinear(hidden_dim, 1)
        else:
            self.fc_out = nn.Linear(hidden_dim, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, channels, frames) → (batch, 1)"""
        # Global average pooling over time
        x = x.mean(dim=-1)  # (batch, channels)
        x = self.dropout(self.activation(self.fc1(x)))
        return self.sigmoid(self.fc_out(x))


# =============================================================================
# Full Model
# =============================================================================
class APDIntelligibilityEstimator(nn.Module):
    """APD音声了解度推定モデル (full pipeline).

    過パラメータ設計 → プルーニング → BitNet の順で圧縮する前提。

    Model size (overparameterized):
        encoder_dim=512, bottleneck=256, n_repeats=3, n_layers=8
        → ~8M params (FP32: ~32MB)
        → After pruning + BitNet: target ~1-2MB

    Inference target: <10ms per 1sec window on mobile CPU
    """

    def __init__(
        self,
        encoder_dim: int = 512,
        bottleneck_dim: int = 256,
        tcn_channels: int = 512,     # 過パラメータ
        n_repeats: int = 3,          # TCNブロック繰り返し
        n_layers: int = 8,           # 各TCNブロック内の層数
        kernel_size: int = 3,
        use_bitnet_output: bool = True,
    ):
        super().__init__()

        # Encoder
        self.encoder = AudioEncoder(
            in_channels=1, encoder_dim=encoder_dim,
        )

        # Bottleneck: encoder_dim → bottleneck_dim
        self.bottleneck = BitConv1d(encoder_dim, bottleneck_dim, 1)

        # Expand to TCN width
        self.tcn_input = BitConv1d(bottleneck_dim, tcn_channels, 1)

        # TCN feature extractor (repeated blocks)
        self.tcn_blocks = nn.ModuleList([
            TCNBlock(tcn_channels, n_layers, kernel_size)
            for _ in range(n_repeats)
        ])

        # Intelligibility estimation head
        self.head = IntelligibilityHead(
            in_features=tcn_channels,
            hidden_dim=256,
            use_bitnet_output=use_bitnet_output,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 1, 16000) - 1sec of 16kHz mono audio

        Returns:
            score: (batch, 1) - intelligibility score [0.0, 1.0]
        """
        # Encode
        features = self.encoder(x)          # (B, encoder_dim, frames)
        features = self.bottleneck(features) # (B, bottleneck_dim, frames)
        features = self.tcn_input(features)  # (B, tcn_channels, frames)

        # TCN feature extraction
        for tcn in self.tcn_blocks:
            features = tcn(features)

        # Estimate intelligibility
        score = self.head(features)          # (B, 1)
        return score


# =============================================================================
# Model Factory
# =============================================================================
def create_model(
    overparameterized: bool = True,
    use_bitnet_output: bool = True,
) -> APDIntelligibilityEstimator:
    """モデル生成ファクトリ.

    Args:
        overparameterized: True=学習用(大), False=プルーニング後想定(小)
        use_bitnet_output: 最終層もBitNetにするか
    """
    if overparameterized:
        return APDIntelligibilityEstimator(
            encoder_dim=512,
            bottleneck_dim=256,
            tcn_channels=512,
            n_repeats=3,
            n_layers=8,
            use_bitnet_output=use_bitnet_output,
        )
    else:
        # プルーニング後の参考サイズ
        return APDIntelligibilityEstimator(
            encoder_dim=256,
            bottleneck_dim=128,
            tcn_channels=256,
            n_repeats=2,
            n_layers=6,
            use_bitnet_output=use_bitnet_output,
        )


if __name__ == "__main__":
    model = create_model(overparameterized=True)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    print(f"FP32 size: {total_params * 4 / 1024 / 1024:.1f} MB")
    print(f"BitNet (1-bit) target size: {total_params / 8 / 1024 / 1024:.1f} MB")

    # Test forward pass
    x = torch.randn(1, 1, 16000)  # 1sec @ 16kHz
    score = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output score: {score.item():.4f}")
