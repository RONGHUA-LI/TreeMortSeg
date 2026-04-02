import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import convnext_small, ConvNeXt_Small_Weights
from torchvision.ops import Conv2dNormActivation


class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.ca = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sa = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.ca(torch.mean(x, dim=(2, 3), keepdim=True))
        max_out = self.ca(torch.max(torch.max(x, dim=2, keepdim=True)[0], dim=3, keepdim=True)[0])
        x = x * self.sigmoid(avg_out + max_out)

        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial = torch.cat([avg_out, max_out], dim=1)
        x = x * self.sa(spatial)

        return x

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        res = torch.cat([avg_out, max_out], dim=1)
        res = self.conv(res)
        return x * self.sigmoid(res)

class ConvNeXtAdapter(nn.Module):
    def __init__(self, in_channels=4, pretrained=True):
        super().__init__()
        weights = ConvNeXt_Small_Weights.IMAGENET1K_V1 if pretrained else None
        base_model = convnext_small(weights=weights)

        if in_channels != 3:
            old_stem = base_model.features[0][0]
            new_stem = nn.Conv2d(in_channels, old_stem.out_channels,
                                 kernel_size=old_stem.kernel_size,
                                 stride=old_stem.stride,
                                 padding=old_stem.padding)
            if pretrained:
                with torch.no_grad():
                    new_stem.weight[:, :3] = old_stem.weight
                    new_stem.weight[:, 3:] = old_stem.weight.mean(dim=1, keepdim=True).repeat(1, in_channels - 3, 1, 1)
                    new_stem.bias = old_stem.bias
            base_model.features[0][0] = new_stem

        self.features = base_model.features
        self.stage_indices = [1, 3, 5, 7]

    def forward(self, x):
        outs = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in self.stage_indices:
                outs.append(x)
        return outs


class PPM(nn.Module):
    def __init__(self, in_dim, reduction_dim, bins=(1, 2, 3, 6)):
        super().__init__()
        self.features = []
        for bin in bins:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(bin),
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(reduction_dim),
                nn.ReLU(inplace=True)
            ))
        self.features = nn.ModuleList(self.features)

    def forward(self, x):
        x_size = x.size()
        out = [x]
        for f in self.features:
            out.append(F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=False))
        return torch.cat(out, 1)


class UPerNetNeck(nn.Module):
    def __init__(self, in_channels_list, out_channels=128):
        super().__init__()
        self.ppm = PPM(in_channels_list[3], int(in_channels_list[3] / 4))
        self.ppm_conv = Conv2dNormActivation(in_channels_list[3] * 2, out_channels, kernel_size=3)

        self.fpn_laterals = nn.ModuleList([
            Conv2dNormActivation(in_channels_list[0], out_channels, kernel_size=1),
            Conv2dNormActivation(in_channels_list[1], out_channels, kernel_size=1),
            Conv2dNormActivation(in_channels_list[2], out_channels, kernel_size=1),
        ])
        self.fpn_outputs = nn.ModuleList([
            Conv2dNormActivation(out_channels, out_channels, kernel_size=3) for _ in range(3)
        ])

    def forward(self, inputs):
        c1, c2, c3, c4 = inputs
        p4 = self.ppm_conv(self.ppm(c4))

        p3 = self.fpn_laterals[2](c3) + F.interpolate(p4, size=c3.shape[2:], mode='bilinear', align_corners=False)
        p3_out = self.fpn_outputs[2](p3)
        p2 = self.fpn_laterals[1](c2) + F.interpolate(p3, size=c2.shape[2:], mode='bilinear', align_corners=False)
        p2_out = self.fpn_outputs[1](p2)
        p1 = self.fpn_laterals[0](c1) + F.interpolate(p2, size=c1.shape[2:], mode='bilinear', align_corners=False)
        p1_out = self.fpn_outputs[0](p1)

        f_p4 = F.interpolate(p4, size=p1.shape[2:], mode='bilinear', align_corners=False)
        f_p3 = F.interpolate(p3_out, size=p1.shape[2:], mode='bilinear', align_corners=False)
        f_p2 = F.interpolate(p2_out, size=p1.shape[2:], mode='bilinear', align_corners=False)

        fused_feat = torch.cat([p1_out, f_p2, f_p3, f_p4], dim=1)
        return {"P1": p1_out, "P2": p2_out, "P3": p3_out, "P4": p4, "Fused": fused_feat}


class DistDecoder(nn.Module):
    def __init__(self, p3_dim=128, p4_dim=128):
        super().__init__()
        self.fusion = Conv2dNormActivation(p3_dim + p4_dim, 128, kernel_size=3)

        self.up_to_1_4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            Conv2dNormActivation(128, 64, kernel_size=3),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            Conv2dNormActivation(64, 64, kernel_size=3)
        )

        self.regress_head = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, p3, p4):
        p4_up = F.interpolate(p4, size=p3.shape[2:], mode='bilinear', align_corners=False)
        feat = self.fusion(torch.cat([p3, p4_up], dim=1))
        feat_1_4 = self.up_to_1_4(feat)
        return self.regress_head(feat_1_4), feat_1_4

class EdgeDecoder(nn.Module):
    def __init__(self, c1_dim, p2_dim=128):
        super().__init__()
        self.conv_1_4 = Conv2dNormActivation(c1_dim + p2_dim, 128, kernel_size=3)
        self.up_to_1_2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv_1_2 = Conv2dNormActivation(128, 32, kernel_size=3)
        self.up_to_1_1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv_1_1 = Conv2dNormActivation(32, 32, kernel_size=3)
        self.head = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, c1, p2):
        p2_up = F.interpolate(p2, size=c1.shape[2:], mode='bilinear', align_corners=False)
        feat = self.conv_1_4(torch.cat([c1, p2_up], dim=1))
        feat = self.conv_1_2(self.up_to_1_2(feat))
        feat_orig = self.conv_1_1(self.up_to_1_1(feat))
        logits = self.head(feat_orig)
        return logits, feat_orig


class MaskDecoder(nn.Module):
    def __init__(self, fused_dim=512, dist_dim=64, edge_dim=32, num_classes=1):
        super().__init__()

        self.attention_1_4 = CBAM(fused_dim)
        self.pre_conv_dist = Conv2dNormActivation(fused_dim + dist_dim, 128, kernel_size=3)
        self.spatial_attention_1_4 = SpatialAttention(kernel_size=7)
        self.up_to_1_2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.refine_1_2 = Conv2dNormActivation(128, 32, kernel_size=3)
        self.up_to_1_1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.refine_1_1 = Conv2dNormActivation(32, 32, kernel_size=3)
        self.final_fusion = Conv2dNormActivation(32 + edge_dim, 32, kernel_size=3)
        self.spatial_attention_orig = SpatialAttention(kernel_size=7)

        self.final_conv = nn.Sequential(
            nn.Dropout2d(0.1),
            nn.Conv2d(32, num_classes, kernel_size=1)
        )

    def forward(self, fused_feat, dist_feat=None, edge_feat_orig=None):
        fused_feat = self.attention_1_4(fused_feat)
        x = self.pre_conv_dist(torch.cat([fused_feat, dist_feat], dim=1))
        x = self.spatial_attention_1_4(x)
        x = self.refine_1_2(self.up_to_1_2(x))
        x_orig = self.up_to_1_1(x)
        x_orig = self.refine_1_1(x_orig)
        x_orig = torch.cat([x_orig, edge_feat_orig], dim=1)
        x_orig = self.final_fusion(x_orig)
        x_orig = self.spatial_attention_orig(x_orig)

        return self.final_conv(x_orig)


class TreeMortSeg(nn.Module):
    def __init__(self, in_channels=4, num_classes=1, pretrained=True):
        super().__init__()

        self.backbone = ConvNeXtAdapter(in_channels=in_channels, pretrained=pretrained)
        base_dims = [96, 192, 384, 768]
        neck_dim = 128

        self.neck = UPerNetNeck(in_channels_list=base_dims, out_channels=neck_dim)

        self.edge_decoder = EdgeDecoder(c1_dim=base_dims[0], p2_dim=neck_dim)
        self.dist_decoder = DistDecoder(p3_dim=neck_dim, p4_dim=neck_dim)

        self.mask_decoder = MaskDecoder(
            fused_dim=neck_dim * 4,  # 512
            dist_dim=64,
            edge_dim=32,
            num_classes=num_classes
        )

    def forward(self, x):
        img_size = x.shape[2:]
        c_feats = self.backbone(x)
        n_feats = self.neck(c_feats)

        edge_logit, edge_feat = self.edge_decoder(c_feats[0], n_feats['P2'])
        dist_logit, dist_feat = self.dist_decoder(n_feats['P3'], n_feats['P4'])
        mask_logit = self.mask_decoder(n_feats['Fused'], dist_feat, edge_feat)

        edge_logit = F.interpolate(edge_logit, size=img_size, mode='bilinear',
                              align_corners=False) if edge_logit is not None else None
        dist_logit = F.interpolate(dist_logit, size=img_size, mode='bilinear',
                              align_corners=False) if dist_logit is not None else None
        mask_logit = F.interpolate(mask_logit, size=img_size, mode='bilinear',
                              align_corners=False) if mask_logit is not None else None
        return mask_logit, edge_logit, dist_logit


def build_treemortseg(cfg: dict) -> nn.Module:
    return TreeMortSeg(
        in_channels=int(cfg.get('in_channels', 4)),
        num_classes=int(cfg.get('num_classes', 1)),
        pretrained=bool(cfg.get('convnext_pretrained', True)),
    )

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dummy_input = torch.randn(1, 4, 224, 224).to(device)

    model = TreeMortSeg(in_channels=4, num_classes=1).to(device)
    model.eval()

    with torch.no_grad():
        m, e, d = model(dummy_input)

    try:
        from thop import profile

        macs, params = profile(model, inputs=(dummy_input,), verbose=False)
        print(f"Parameters: {params / 1e6:.2f} M")
        print(f"FLOPs: {macs / 1e9:.2f} G")
    except ImportError:
        print("thop not installed, skipping profile.")