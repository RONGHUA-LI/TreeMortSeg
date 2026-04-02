import torch
import torch.nn as nn
from transformers import SegformerConfig, SegformerModel


class SegFormer(nn.Module):
    def __init__(self, in_channels, num_classes, model_type="nvidia/mit-b0"):
        super().__init__()

        config = SegformerConfig.from_pretrained(model_type)
        config.num_channels = in_channels

        self.backbone = SegformerModel.from_pretrained(
            model_type,
            config=config,
            ignore_mismatched_sizes=True
        )

        embedding_dim = config.hidden_sizes
        self.linear_layers = nn.ModuleList([
            nn.Conv2d(dim, 256, kernel_size=1) for dim in embedding_dim
        ])

        self.fuse = nn.Sequential(
            nn.Conv2d(256 * 4, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.classifier = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x):
        outputs = self.backbone(x, output_hidden_states=True)
        features = outputs.hidden_states

        target_size = features[0].shape[2:]
        all_features = []

        for i, f in enumerate(features):

            out = self.linear_layers[i](f)
            out = nn.functional.interpolate(out, size=target_size, mode='bilinear', align_corners=False)
            all_features.append(out)

        out = self.fuse(torch.cat(all_features, dim=1))
        out = self.classifier(out)

        out = nn.functional.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=False)
        return out

def build_segformer(cfg: dict):
    num_classes = int(cfg.get('num_classes', 1))
    in_channels = int(cfg.get('in_channels', 4))
    backbone    = cfg['segformer_pretrained'].get("backbone", "nvidia/mit-b2")
    return SegFormer(in_channels=in_channels, num_classes=num_classes, model_type=backbone)