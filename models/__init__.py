import torch.nn as nn

def get_model(model_cfg: dict) -> nn.Module:
    name = model_cfg.get('name', 'unet')
    
    if name.lower() == 'unet':
        from .unet import build_unet
        return build_unet(model_cfg)

    if name.lower() == 'treemortseg':
        from .treemortseg import build_treemortseg
        return build_treemortseg(model_cfg)

    elif name in ('deeplabv3','deeplabv3+'):
        from .deeplab import build_deeplabv3
        return build_deeplabv3(model_cfg)

    elif name in ('segformer'):
        from .segformer import build_segformer
        return build_segformer(model_cfg)

    elif name in ('mask2former'):
        from .mask2former import build_mask2former
        return build_mask2former(model_cfg)

    elif name.lower() == 'efficient_unet_b7':
        from .efficientunet import build_efficient_unet_b7
        return build_efficient_unet_b7(model_cfg)


    raise ValueError(f"Unsupported model: {name}")