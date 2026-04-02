import types
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights

def create_deeplabv3plus(in_channels, num_classes, backbone, pretrained):
    """
    Create a DeepLabV3+ model with custom input channels and number of classes.
    Args:
      in_channels: number of input bands (e.g. 5 for RGB+NIR+NDVI)
      num_classes: number of output segmentation classes
      backbone:    name of the ResNet backbone (only "resnet50" supported currently)
      pretrained:  whether to load ImageNet-pretrained weights for the backbone
    """
    
    backbone = backbone.lower()
    if backbone != "resnet50":
        raise ValueError(f"Unsupported backbone: {backbone}")

    # Load base model
    weights = DeepLabV3_ResNet50_Weights.DEFAULT if pretrained else None
    model = deeplabv3_resnet50(weights=weights, progress=True)

    # Replace first conv to accept in_channels
    old_conv = model.backbone.conv1
    new_conv = nn.Conv2d(
        in_channels,
        old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=(old_conv.bias is not None)
    )
    if pretrained:
        with torch.no_grad():
            # copy the RGB weights
            new_conv.weight[:, :3, :, :].copy_(old_conv.weight)
            # for extra channels, replicate the mean of RGB weights
            if in_channels > 3:
                mean_weight = old_conv.weight.mean(dim=1, keepdim=True)
                new_conv.weight[:, 3:, :, :].copy_(
                    mean_weight.repeat(1, in_channels - 3, 1, 1)
                )
    model.backbone.conv1 = new_conv

    # Replace main classifier head
    old_cls = model.classifier[-1]
    model.classifier[-1] = nn.Conv2d(
        old_cls.in_channels,
        num_classes,
        kernel_size=old_cls.kernel_size,
        stride=1
    )
    
    # 5) Monkey‐patch forward: call original then grab 'out'
    orig_forward = model.forward
    def forward_out(self, x):
        return orig_forward(x)['out']
    model.forward = types.MethodType(forward_out, model)

    return model


def build_deeplabv3(cfg: dict) -> nn.Module:
    in_ch       = int(cfg.get('in_channels', 5))
    num_classes = int(cfg.get("num_classes", 1))
    backbone    = cfg['deeplab_pretrained'].get("backbone", "resnet50")
    pretrained  = bool(cfg['deeplab_pretrained'].get("pretrained", True))
    return create_deeplabv3plus(
        in_channels=in_ch,
        num_classes=num_classes,
        backbone=backbone,
        pretrained=pretrained
    )