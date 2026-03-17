"""
Loss functions for APD Intelligibility Estimator training.

Combined loss:
  total = MSE + λ_rank * ranking_loss + λ_boundary * boundary_weighted_mse

Design rationale:
  - MSE: basic regression loss
  - Ranking loss: preserves relative order even when pseudo-label absolute
    values are imprecise (critical for pseudo-label based training)
  - Boundary-aware MSE: higher weight near decision thresholds (0.3, 0.5, 0.8)
    where classification accuracy matters most for the APD app UI
"""

import torch
import torch.nn as nn


class RankingLoss(nn.Module):
    """Pairwise ranking loss within a batch.

    For all pairs (i, j) where label_i > label_j,
    penalize if prediction_i <= prediction_j.

    Uses margin-based hinge: max(0, margin - (pred_i - pred_j))
    """

    def __init__(self, margin: float = 0.05):
        super().__init__()
        self.margin = margin

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: (batch,) or (batch, 1)
            targets: (batch,) or (batch, 1)
        """
        pred = predictions.view(-1)
        tgt = targets.view(-1)

        # All pairwise differences
        pred_diff = pred.unsqueeze(1) - pred.unsqueeze(0)  # (B, B)
        tgt_diff = tgt.unsqueeze(1) - tgt.unsqueeze(0)     # (B, B)

        # Only consider pairs where target order is clear (difference > margin)
        valid_mask = tgt_diff > self.margin  # (B, B) bool

        if not valid_mask.any():
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

        # Hinge loss: penalize when pred_diff < margin for pairs where tgt_i > tgt_j
        violations = torch.clamp(self.margin - pred_diff, min=0.0)
        loss = (violations * valid_mask.float()).sum() / valid_mask.float().sum()

        return loss


class BoundaryWeightedMSE(nn.Module):
    """MSE with higher weight near decision boundary thresholds.

    The APD app uses thresholds [0.3, 0.5, 0.8] to color-code severity.
    Accuracy near these boundaries matters more than in the middle of zones.

    Weight function: w(y) = 1 + Σ_t  A * exp(-(y - t)² / (2σ²))
    """

    def __init__(
        self,
        thresholds: list[float] = None,
        sigma: float = 0.05,
        amplitude: float = 2.0,
    ):
        super().__init__()
        if thresholds is None:
            thresholds = [0.3, 0.5, 0.8]
        self.register_buffer(
            "thresholds", torch.tensor(thresholds, dtype=torch.float32)
        )
        self.sigma = sigma
        self.amplitude = amplitude

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        pred = predictions.view(-1)
        tgt = targets.view(-1)

        # Compute weights based on proximity to thresholds
        # tgt: (B,), thresholds: (T,) → distances: (B, T)
        distances = (tgt.unsqueeze(1) - self.thresholds.unsqueeze(0)) ** 2
        gaussian_weights = self.amplitude * torch.exp(-distances / (2 * self.sigma ** 2))
        weights = 1.0 + gaussian_weights.sum(dim=1)  # (B,)

        # Weighted MSE
        mse = (pred - tgt) ** 2
        return (weights * mse).mean()


class APDLoss(nn.Module):
    """Combined loss for APD intelligibility estimation.

    total = MSE + λ_rank * ranking_loss + λ_boundary * boundary_mse
    """

    def __init__(
        self,
        ranking_weight: float = 0.1,
        boundary_weight: float = 0.05,
        boundary_thresholds: list[float] = None,
        boundary_sigma: float = 0.05,
        ranking_margin: float = 0.05,
    ):
        super().__init__()
        self.mse = nn.MSELoss()
        self.ranking = RankingLoss(margin=ranking_margin)
        self.boundary = BoundaryWeightedMSE(
            thresholds=boundary_thresholds, sigma=boundary_sigma,
        )
        self.ranking_weight = ranking_weight
        self.boundary_weight = boundary_weight

    def forward(
        self, predictions: torch.Tensor, targets: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Returns:
            (total_loss, loss_components_dict)
        """
        mse_loss = self.mse(predictions.view(-1), targets.view(-1))
        rank_loss = self.ranking(predictions, targets)
        bound_loss = self.boundary(predictions, targets)

        total = mse_loss + self.ranking_weight * rank_loss + self.boundary_weight * bound_loss

        components = {
            "mse": mse_loss.item(),
            "ranking": rank_loss.item(),
            "boundary": bound_loss.item(),
            "total": total.item(),
        }

        return total, components
