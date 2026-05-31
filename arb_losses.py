import torch
from torch import nn
import torch.nn.functional as F


class BCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bceloss = nn.BCELoss()

    def forward(self, pred, target):
        size = pred.size(0)
        return self.bceloss(pred.view(size, -1), target.view(size, -1))


class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        smooth = 1.0
        size = pred.size(0)
        pred_ = pred.view(size, -1)
        target_ = target.view(size, -1)
        intersection = pred_ * target_
        dice_score = (2 * intersection.sum(1) + smooth) / (pred_.sum(1) + target_.sum(1) + smooth)
        return 1 - dice_score.mean()


class BceDiceLoss(nn.Module):
    def __init__(self, wb=1.0, wd=1.0):
        super().__init__()
        self.bce = BCELoss()
        self.dice = DiceLoss()
        self.wb = wb
        self.wd = wd

    def forward(self, pred, target):
        return self.wb * self.bce(pred, target) + self.wd * self.dice(pred, target)


class SoftBoundaryGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
        sobel_y = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]])
        self.register_buffer("sobel_x", sobel_x.view(1, 1, 3, 3))
        self.register_buffer("sobel_y", sobel_y.view(1, 1, 3, 3))

    def forward(self, mask):
        gx = F.conv2d(mask, self.sobel_x, padding=1)
        gy = F.conv2d(mask, self.sobel_y, padding=1)
        edge = torch.sqrt(gx.pow(2) + gy.pow(2) + 1e-6)
        edge = edge / (edge.amax(dim=(2, 3), keepdim=True) + 1e-6)
        return F.max_pool2d(edge, kernel_size=3, stride=1, padding=1).clamp(0.0, 1.0)


class SurfaceAwareLoss(nn.Module):
    def __init__(self, boundary_weight=1.0):
        super().__init__()
        self.boundary_generator = SoftBoundaryGenerator()
        self.boundary_weight = boundary_weight
        self.compound = BceDiceLoss(1.0, 1.0)

    def forward(self, pred, target):
        target_boundary = self.boundary_generator(target)
        pred_boundary = self.boundary_generator(pred)
        return self.compound(pred, target) + self.boundary_weight * self.compound(pred_boundary, target_boundary)


class ARBCompoundLoss(nn.Module):
    def __init__(self, deep_weights=None, boundary_weights=None, deep_supervision_weight=0.4, boundary_supervision_weight=0.3):
        super().__init__()
        self.seg_loss = BceDiceLoss(1.0, 1.0)
        self.boundary_loss = BceDiceLoss(1.0, 1.0)
        self.surface_loss = SurfaceAwareLoss(boundary_weight=0.5)
        self.boundary_generator = SoftBoundaryGenerator()
        self.deep_weights = deep_weights or [0.1, 0.2, 0.3, 0.4, 0.5]
        self.boundary_weights = boundary_weights or [0.1, 0.2, 0.3, 0.4, 0.5]
        self.deep_supervision_weight = deep_supervision_weight
        self.boundary_supervision_weight = boundary_supervision_weight

    def _resize_like(self, x, ref):
        return F.interpolate(x, size=ref.shape[2:], mode="bilinear", align_corners=True)

    def forward(self, outputs, target):
        deep_masks, deep_boundaries, final_mask = outputs
        target = target.float()

        final_loss = self.seg_loss(final_mask, target)
        surface_loss = self.surface_loss(final_mask, target)
        target_boundary = self.boundary_generator(target)

        deep_loss = 0.0
        for idx, pred in enumerate(deep_masks):
            deep_loss = deep_loss + self.deep_weights[idx] * self.seg_loss(self._resize_like(pred, target), target)

        boundary_loss = 0.0
        for idx, pred in enumerate(deep_boundaries):
            boundary_loss = boundary_loss + self.boundary_weights[idx] * self.boundary_loss(self._resize_like(pred, target_boundary), target_boundary)

        return final_loss + self.deep_supervision_weight * deep_loss + self.boundary_supervision_weight * boundary_loss + 0.2 * surface_loss
