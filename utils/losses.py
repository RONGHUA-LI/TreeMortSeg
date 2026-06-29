import torch
import torch.nn as nn
from torchvision.ops import sigmoid_focal_loss


class DiceLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logits, targets, mask=None):
        """Compute Dice loss from logits and binary targets."""
        probs = torch.sigmoid(logits)
        if mask is not None:
            probs = probs * mask

        p_flat = probs.view(-1)
        t_flat = targets.float().view(-1)

        inter = (p_flat * t_flat).sum()
        union = p_flat.sum() + t_flat.sum()
        dice = (inter + self.eps) / (union + self.eps)
        return 1 - dice


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction="none"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        """Compute sigmoid focal loss with the configured weighting."""
        return sigmoid_focal_loss(
            inputs=logits,
            targets=targets.float(),
            alpha=self.alpha,
            gamma=self.gamma,
            reduction=self.reduction,
        )


class FocalDiceLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, lambda_dice=1.0):
        super().__init__()
        self.focal = FocalLoss(alpha=alpha, gamma=gamma)
        self.dice = DiceLoss()
        self.lambda_dice = lambda_dice

    def forward(self, logits, targets):
        """Compute focal loss plus a weighted Dice-loss term."""
        loss_focal = self.focal(logits, targets)
        loss_dice = self.dice(logits, targets)
        return loss_focal + self.lambda_dice * loss_dice


class EdgeBoundaryLoss(nn.Module):
    def __init__(self, lambda_boundary=1.0, max_dist=10.0):
        super().__init__()
        self.dice = DiceLoss()
        self.lambda_boundary = lambda_boundary
        self.max_dist = max_dist

    def forward(self, logits, gts, dist_maps, mask=None):
        """Penalize edge errors with an additional distance-map boundary term."""
        loss_dice = self.dice(logits, gts, mask)
        probs = torch.sigmoid(logits)

        dist_maps = torch.clamp(dist_maps, 0, self.max_dist) / self.max_dist

        if mask is not None:
            probs = probs * mask
            dist_maps = dist_maps * mask
            loss_boundary = (probs * dist_maps).sum() / (mask.sum() + 1e-6)
        else:
            loss_boundary = (probs * dist_maps).mean()

        return loss_dice + self.lambda_boundary * loss_boundary
