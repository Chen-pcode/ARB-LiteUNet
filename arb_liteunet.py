import math

import torch
from torch import nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        if self.data_format == "channels_first":
            mean = x.mean(1, keepdim=True)
            var = (x - mean).pow(2).mean(1, keepdim=True)
            x = (x - mean) / torch.sqrt(var + self.eps)
            return self.weight[:, None, None] * x + self.bias[:, None, None]
        raise NotImplementedError


class Down(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=2, stride=2)

    def forward(self, x):
        return self.conv(self.bn(x))


class ConvLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=7, padding=3, stride=1, groups=dim, padding_mode="reflect")
        self.norm1 = nn.BatchNorm2d(dim)
        self.conv2 = nn.Conv2d(dim, 4 * dim, kernel_size=1)
        self.act1 = nn.GELU()
        self.norm2 = nn.BatchNorm2d(dim)
        self.conv3 = nn.Conv2d(4 * dim, dim, kernel_size=1)
        self.act2 = nn.GELU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.conv2(x)
        x = self.act1(x)
        x = self.conv3(x)
        x = self.norm2(x)
        x = self.act2(x)
        return x


class SqueezeExcite(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super().__init__()
        hidden = max(1, in_channels // reduction)
        self.fc1 = nn.Conv2d(in_channels, hidden, kernel_size=1)
        self.fc2 = nn.Conv2d(hidden, in_channels, kernel_size=1)
        self.activation = nn.ReLU()
        self.scale = nn.GELU()

    def forward(self, x):
        scale = torch.mean(x, dim=(2, 3), keepdim=True)
        scale = self.activation(self.fc1(scale))
        scale = self.scale(self.fc2(scale))
        return x * scale


class SCGS(nn.Module):
    def __init__(self, inp, hidden_dim, oup, kernel_size, stride, use_se, use_hs):
        super().__init__()
        assert stride in [1, 2]
        padding = (kernel_size - 1) // 2
        self.use_res_connect = stride == 1 and inp == oup

        self.expand = nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False) if inp != hidden_dim else None
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.act1 = nn.GELU() if use_hs else nn.ReLU()

        self.depthwise = nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, padding, groups=hidden_dim, bias=False)
        self.bn2 = nn.BatchNorm2d(hidden_dim)
        self.act2 = nn.GELU() if use_hs else nn.ReLU()

        self.se = SqueezeExcite(hidden_dim) if use_se else nn.Identity()
        self.project = nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False)
        self.bn3 = nn.BatchNorm2d(oup)

    def forward(self, x):
        out = x
        if self.expand is not None:
            out = self.act1(self.bn1(self.expand(out)))
        out = self.act2(self.bn2(self.depthwise(out)))
        out = self.se(out)
        out = self.bn3(self.project(out))
        if self.use_res_connect:
            return x + out
        return out


class HGAS(nn.Module):
    def __init__(self, dim_in, dim_out, x=8, y=8):
        super().__init__()
        self.num_groups = 4
        assert dim_in % self.num_groups == 0, "HGAS expects dim_in divisible by 4."
        self.c_dim_in = dim_in // self.num_groups

        self.params_xy = nn.Parameter(torch.ones(1, self.c_dim_in, x, y))
        self.conv_xy = nn.Sequential(
            nn.Conv2d(self.c_dim_in, self.c_dim_in, kernel_size=3, padding=1, groups=self.c_dim_in),
            nn.GELU(),
            nn.Conv2d(self.c_dim_in, self.c_dim_in, 1),
        )

        self.params_zx = nn.Parameter(torch.ones(1, self.c_dim_in, x, 1))
        self.conv_zx = nn.Sequential(
            nn.Conv2d(self.c_dim_in, self.c_dim_in, kernel_size=3, padding=1, groups=self.c_dim_in),
            nn.GELU(),
            nn.Conv2d(self.c_dim_in, self.c_dim_in, 1),
        )

        self.params_zy = nn.Parameter(torch.ones(1, self.c_dim_in, 1, y))
        self.conv_zy = nn.Sequential(
            nn.Conv2d(self.c_dim_in, self.c_dim_in, kernel_size=3, padding=1, groups=self.c_dim_in),
            nn.GELU(),
            nn.Conv2d(self.c_dim_in, self.c_dim_in, 1),
        )

        self.dw = nn.Sequential(
            nn.Conv2d(self.c_dim_in, self.c_dim_in, 1),
            nn.GELU(),
            nn.Conv2d(self.c_dim_in, self.c_dim_in, kernel_size=3, padding=1, groups=self.c_dim_in),
        )

        self.norm1 = LayerNorm(dim_in, data_format="channels_first")
        self.norm2 = LayerNorm(dim_in, data_format="channels_first")
        self.ldw = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, kernel_size=3, padding=1, groups=dim_in),
            nn.GELU(),
            nn.Conv2d(dim_in, dim_out, 1),
        )

    def forward(self, x):
        x = self.norm1(x)
        x1, x2, x3, x4 = torch.chunk(x, 4, dim=1)

        x1 = x1 * self.conv_xy(F.interpolate(self.params_xy, size=x1.shape[2:], mode="bilinear", align_corners=True))

        zx_params = F.interpolate(self.params_zx, size=x2.shape[2:], mode="bilinear", align_corners=True)
        x2 = x2 * self.conv_zx(zx_params)

        zy_params = F.interpolate(self.params_zy, size=x3.shape[2:], mode="bilinear", align_corners=True)
        x3 = x3 * self.conv_zy(zy_params)

        x4 = self.dw(x4)
        x = torch.cat([x2, x4, x1, x3], dim=1)
        x = self.norm2(x)
        return self.ldw(x)


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
        edge = F.max_pool2d(edge, kernel_size=3, stride=1, padding=1)
        return edge.clamp(0.0, 1.0)


class BoundaryPredictionHead(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        mid = max(1, in_channels // 2)
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, mid, kernel_size=3, padding=1, groups=max(1, min(mid, in_channels))),
            nn.GELU(),
            nn.Conv2d(mid, 1, kernel_size=1),
        )

    def forward(self, x):
        return torch.sigmoid(self.head(x))


class MaskPredictionHead(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        mid = max(1, in_channels // 2)
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, mid, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(mid, 1, kernel_size=1),
        )

    def forward(self, x):
        return torch.sigmoid(self.head(x))


class ArtifactRobustContextGate(nn.Module):
    def __init__(self, skip_channels, decoder_channels):
        super().__init__()
        self.skip_proj = nn.Conv2d(skip_channels, decoder_channels, kernel_size=1)
        self.dec_proj = nn.Conv2d(decoder_channels, decoder_channels, kernel_size=1)
        self.boundary_proj = nn.Conv2d(1, decoder_channels, kernel_size=1)
        self.mix = nn.Sequential(
            nn.Conv2d(decoder_channels * 3, decoder_channels, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(decoder_channels, decoder_channels, kernel_size=3, padding=1, groups=decoder_channels, bias=False),
            nn.GELU(),
        )
        self.artifact_head = nn.Sequential(
            nn.Conv2d(decoder_channels, decoder_channels, kernel_size=3, padding=1, groups=decoder_channels),
            nn.GELU(),
            nn.Conv2d(decoder_channels, 1, kernel_size=1),
        )
        self.refine = nn.Sequential(
            nn.Conv2d(decoder_channels, decoder_channels, kernel_size=3, padding=1, groups=decoder_channels),
            nn.GELU(),
            nn.Conv2d(decoder_channels, decoder_channels, kernel_size=1),
        )

    def forward(self, skip, decoder, boundary_prior=None):
        skip = self.skip_proj(skip)
        decoder = self.dec_proj(decoder)
        if boundary_prior is None:
            boundary_prior = torch.zeros_like(decoder[:, :1])
        boundary_prior = self.boundary_proj(boundary_prior)
        mixed = self.mix(torch.cat([skip, decoder, boundary_prior], dim=1))
        artifact_map = torch.sigmoid(self.artifact_head(skip + decoder))
        enhanced_skip = skip * (1.0 - artifact_map)
        enhanced_skip = enhanced_skip + self.refine(mixed)
        return enhanced_skip, artifact_map


class BoundaryRegionFusion(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 4, channels, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1),
        )
        self.boundary_gate = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.mask_gate = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.out = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1),
        )

    def forward(self, decoder, skip, mask_prior=None, boundary_prior=None):
        if mask_prior is None:
            mask_prior = torch.zeros_like(decoder[:, :1])
        if boundary_prior is None:
            boundary_prior = torch.zeros_like(decoder[:, :1])
        fusion = torch.cat([decoder, skip, mask_prior, boundary_prior], dim=1)
        gate = torch.sigmoid(self.gate(fusion))
        decoder = decoder + gate * skip
        decoder = decoder + self.mask_gate(mask_prior) * skip * 0.5
        decoder = decoder + self.boundary_gate(boundary_prior) * skip * 0.5
        return self.out(decoder)


class ARBLiteUNet(nn.Module):
    def __init__(self, num_classes=1, input_channels=3, c_list=None, use_arcg=True, use_brf=True):
        super().__init__()
        if c_list is None:
            c_list = [8, 12, 16, 32, 48, 64]
        self.use_arcg = use_arcg
        self.use_brf = use_brf

        self.encoder1 = SCGS(input_channels, c_list[0] * 2, c_list[0], 3, stride=1, use_se=True, use_hs=True)
        self.encoder2 = SCGS(c_list[0], c_list[1] * 2, c_list[1], 3, stride=1, use_se=True, use_hs=True)
        self.encoder3 = nn.Sequential(
            ConvLayer(c_list[1]),
            SCGS(c_list[1], c_list[2] * 2, c_list[2], 3, stride=1, use_se=True, use_hs=True),
        )
        self.encoder4 = HGAS(c_list[2], c_list[3])
        self.encoder5 = HGAS(c_list[3], c_list[4])
        self.encoder6 = HGAS(c_list[4], c_list[5])

        self.down1 = Down(c_list[0])
        self.down2 = Down(c_list[1])
        self.down3 = Down(c_list[2])

        self.up5 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.up4 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.up3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        self.decoder1 = HGAS(c_list[5], c_list[4])
        self.decoder2 = HGAS(c_list[4], c_list[3])
        self.decoder3 = HGAS(c_list[3], c_list[2])
        self.decoder4 = SCGS(c_list[2], c_list[2] * 2, c_list[1], 3, stride=1, use_se=True, use_hs=True)
        self.decoder5 = SCGS(c_list[1], c_list[1] * 2, c_list[0], 3, stride=1, use_se=True, use_hs=True)

        self.arcg5 = ArtifactRobustContextGate(c_list[4], c_list[4])
        self.arcg4 = ArtifactRobustContextGate(c_list[3], c_list[3])
        self.arcg3 = ArtifactRobustContextGate(c_list[2], c_list[2])
        self.arcg2 = ArtifactRobustContextGate(c_list[1], c_list[1])
        self.arcg1 = ArtifactRobustContextGate(c_list[0], c_list[0])

        self.brf5 = BoundaryRegionFusion(c_list[4])
        self.brf4 = BoundaryRegionFusion(c_list[3])
        self.brf3 = BoundaryRegionFusion(c_list[2])
        self.brf2 = BoundaryRegionFusion(c_list[1])
        self.brf1 = BoundaryRegionFusion(c_list[0])

        self.mask5 = MaskPredictionHead(c_list[4])
        self.mask4 = MaskPredictionHead(c_list[3])
        self.mask3 = MaskPredictionHead(c_list[2])
        self.mask2 = MaskPredictionHead(c_list[1])
        self.mask1 = MaskPredictionHead(c_list[0])

        self.bound5 = BoundaryPredictionHead(c_list[4])
        self.bound4 = BoundaryPredictionHead(c_list[3])
        self.bound3 = BoundaryPredictionHead(c_list[2])
        self.bound2 = BoundaryPredictionHead(c_list[1])
        self.bound1 = BoundaryPredictionHead(c_list[0])

        self.final = nn.Conv2d(c_list[0], num_classes, kernel_size=1)
        self.boundary_generator = SoftBoundaryGenerator()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Conv2d)):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def _resize(self, x, ref):
        return F.interpolate(x, size=ref.shape[2:], mode="bilinear", align_corners=True)

    def forward(self, x):
        x = self.encoder1(x)
        x = F.gelu(self.down1(x))
        t1 = x

        x = self.encoder2(x)
        x = F.gelu(self.down2(x))
        t2 = x

        x = self.encoder3(x)
        x = F.gelu(self.down3(x))
        t3 = x

        x = self.encoder4(x)
        x = F.gelu(F.max_pool2d(x, 2))
        t4 = x

        x = self.encoder5(x)
        x = F.gelu(F.max_pool2d(x, 2))
        t5 = x

        x = F.gelu(self.encoder6(x))

        deep_masks = []
        deep_boundaries = []

        x = self.decoder1(x)
        x = F.gelu(x)
        mask5 = self.mask5(x)
        bound5 = self.bound5(x)
        if self.use_arcg:
            skip5, _ = self.arcg5(t5, x, bound5)
        else:
            skip5 = t5
        x = self.brf5(x, skip5, mask5, bound5) if self.use_brf else x + skip5
        deep_masks.append(mask5)
        deep_boundaries.append(bound5)

        x = self.up5(x)
        x = self.decoder2(x)
        x = F.gelu(x)
        mask4 = self.mask4(x)
        bound4 = self.bound4(x)
        if self.use_arcg:
            skip4, _ = self.arcg4(t4, x, bound4)
        else:
            skip4 = t4
        x = self.brf4(x, skip4, mask4, bound4) if self.use_brf else x + skip4
        deep_masks.append(mask4)
        deep_boundaries.append(bound4)

        x = self.up4(x)
        x = self.decoder3(x)
        x = F.gelu(x)
        mask3 = self.mask3(x)
        bound3 = self.bound3(x)
        if self.use_arcg:
            skip3, _ = self.arcg3(t3, x, bound3)
        else:
            skip3 = t3
        x = self.brf3(x, skip3, mask3, bound3) if self.use_brf else x + skip3
        deep_masks.append(mask3)
        deep_boundaries.append(bound3)

        x = self.up3(x)
        x = self.decoder4(x)
        x = F.gelu(x)
        mask2 = self.mask2(x)
        bound2 = self.bound2(x)
        if self.use_arcg:
            skip2, _ = self.arcg2(t2, x, bound2)
        else:
            skip2 = t2
        x = self.brf2(x, skip2, mask2, bound2) if self.use_brf else x + skip2
        deep_masks.append(mask2)
        deep_boundaries.append(bound2)

        x = self.up2(x)
        x = self.decoder5(x)
        x = F.gelu(x)
        mask1 = self.mask1(x)
        bound1 = self.bound1(x)
        if self.use_arcg:
            skip1, _ = self.arcg1(t1, x, bound1)
        else:
            skip1 = t1
        x = self.brf1(x, skip1, mask1, bound1) if self.use_brf else x + skip1
        deep_masks.append(mask1)
        deep_boundaries.append(bound1)

        final_mask = torch.sigmoid(self.final(self.up1(x)))
        deep_masks = tuple(F.interpolate(m, size=final_mask.shape[2:], mode="bilinear", align_corners=True) for m in deep_masks)
        deep_boundaries = tuple(F.interpolate(b, size=final_mask.shape[2:], mode="bilinear", align_corners=True) for b in deep_boundaries)

        return deep_masks, deep_boundaries, final_mask
