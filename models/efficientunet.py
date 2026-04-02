import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.feature_extraction import create_feature_extractor


# 1. 定义标准的双层卷积块 (保持 Decoder 的稳健性)
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


# 2. 定义包含双卷积的解码单元
class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.conv = DoubleConv(in_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        # 上采样到 skip 的尺寸
        x = nn.functional.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
        # 拼接特征
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# --- EfficientUNet B0 版本 ---
class EfficientUNetB0(nn.Module):
    def __init__(self, in_channels=4, num_classes=1):
        super().__init__()
        base_model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        backbone = base_model.features
        # 修改首层
        backbone[0][0] = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False)

        # B0 特征节点
        return_nodes = {'1': 's1', '2': 's2', '3': 's3', '5': 's4', '8': 'bottleneck'}
        self.encoder = create_feature_extractor(backbone, return_nodes=return_nodes)

        # 解码器通道适配: B0 的 skip 通道分别为 [16, 24, 40, 112, 1280]
        self.up1 = DecoderBlock(1280, 112, 256)
        self.up2 = DecoderBlock(256, 40, 128)
        self.up3 = DecoderBlock(128, 24, 64)
        self.up4 = DecoderBlock(64, 16, 32)
        self.final_head = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(32, num_classes, kernel_size=1)
        )

    def forward(self, x):
        feats = self.encoder(x)
        x = self.up1(feats['bottleneck'], feats['s4'])
        x = self.up2(x, feats['s3'])
        x = self.up3(x, feats['s2'])
        x = self.up4(x, feats['s1'])
        return self.final_head(x)


# --- EfficientUNet B7 版本 ---
class EfficientUNetB7(nn.Module):
    def __init__(self, in_channels=4, num_classes=1):
        super().__init__()
        base_model = models.efficientnet_b7(weights=models.EfficientNet_B7_Weights.DEFAULT)
        backbone = base_model.features
        # 修改首层 (B7 首层输出是 64)
        backbone[0][0] = nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1, bias=False)

        # B7 特征节点索引与 B0 不同
        return_nodes = {'1': 's1', '2': 's2', '3': 's3', '5': 's4', '8': 'bottleneck'}
        self.encoder = create_feature_extractor(backbone, return_nodes=return_nodes)

        # 解码器通道适配: B7 的 skip 通道分别为 [32, 48, 80, 224, 2560]
        self.up1 = DecoderBlock(2560, 224, 512)
        self.up2 = DecoderBlock(512, 80, 256)
        self.up3 = DecoderBlock(256, 48, 128)
        self.up4 = DecoderBlock(128, 32, 64)
        self.final_head = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, num_classes, kernel_size=1)
        )

    def forward(self, x):
        feats = self.encoder(x)
        x = self.up1(feats['bottleneck'], feats['s4'])
        x = self.up2(x, feats['s3'])
        x = self.up3(x, feats['s2'])
        x = self.up4(x, feats['s1'])
        return self.final_head(x)


def build_efficient_unet_b7(cfg):
    return EfficientUNetB7(
        in_channels=cfg.get('in_channels', 4),
        num_classes=cfg.get('num_classes', 1)
    )

if __name__ == "__main__":
    from thop import profile

    img = torch.randn(2, 4, 224, 224)

    model_b0 = EfficientUNetB0()
    m0, p0 = profile(model_b0, inputs=(img,), verbose=False)
    print(f"[B0] Params: {p0 / 1e6:.2f}M | FLOPs: {m0 * 2 / 1e9:.2f}G")

    model_b7 = EfficientUNetB7()
    m7, p7 = profile(model_b7, inputs=(img,), verbose=False)
    print(f"[B7] Params: {p7 / 1e6:.2f}M | FLOPs: {m7 * 2 / 1e9:.2f}G")