import argparse
import logging
import os

import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from PIL import Image
from matplotlib import pyplot as plt
from osgeo import gdal
from torchvision import transforms

from utils.tools import (
    load_config
)
import rasterio




from models import get_model


def load_naip_tile(path: str, normalize: bool = True):
    """
    NAIP tile
    return:
      img_naip: [4, H, W]  (RGBN)
      no_data_mask: [H, W]
    """
    # Read all bands
    arr = rasterio.open(path).read()
    # Separate image
    img = arr[:4, ...].astype(np.float32)  # RGBNIR
    # Create no-data mask
    no_data_mask = (img[0] == 0) & (img[1] == 0) & (img[2] == 0) & (img[3] == 0)

    if normalize:
        # Normalize to [0, 1]
        img = img / 255.0

    return img, no_data_mask

def predict_img(net, naip, num_classes, device, tile_size=224, overlap=112):
    """
    Auto-detect small images (≤224) -> direct inference
    Large images -> tile-based inference
    """
    net.eval()
    H, W = naip.shape[1], naip.shape[2]
    naip_tensor = torch.from_numpy(naip).unsqueeze(0).to(device=device, dtype=torch.float32)

    if H <= tile_size and W <= tile_size:
        with torch.no_grad():
            outputs = net(naip_tensor)
            if isinstance(outputs, (tuple, list)):
                out_masks = outputs[0]
            else:
                out_masks = outputs
        return out_masks.cpu()

    full_logits = torch.zeros((num_classes, H, W), device="cpu")
    count_map = torch.zeros((H, W), device="cpu")
    stride = tile_size - overlap
    with torch.no_grad():
        for y in range(0, H, stride):
            for x in range(0, W, stride):

                y2 = min(y + tile_size, H)
                x2 = min(x + tile_size, W)

                tile = naip_tensor[:, :, y:y2, x:x2]

                pad_bottom = tile_size - (y2 - y)
                pad_right = tile_size - (x2 - x)
                tile = F.pad(tile, (0, pad_right, 0, pad_bottom))

                outputs = net(tile)
                if isinstance(outputs, (tuple, list)):
                    tile_pred = outputs[0]
                else:
                    tile_pred = outputs
                tile_pred = tile_pred.cpu()
                tile_pred = tile_pred[0, :, :y2-y, :x2-x]

                full_logits[:, y:y2, x:x2] += tile_pred
                count_map[y:y2, x:x2] += 1

    full_logits /= count_map
    return full_logits

def get_output_filenames(args):
    def _generate_name(input_path):
        input_dir = os.path.dirname(input_path)
        target_dir = os.path.join("predicts", input_dir)
        os.makedirs(target_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(input_path))[0]
        return os.path.join(target_dir, f"{args.model_name}_{base}")

    if args.output:
        return [os.path.join(args.out_dir, os.path.basename(o)) for o in args.output]
    else:
        return list(map(_generate_name, args.input))

def mask_to_image(mask: np.ndarray, mask_values):
    if isinstance(mask_values[0], list):
        out = np.zeros((mask.shape[-2], mask.shape[-1], len(mask_values[0])), dtype=np.uint8)
    elif mask_values == [0, 1]:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=bool)
    else:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=np.uint8)

    if mask.ndim == 3:
        mask = np.argmax(mask, axis=0)

    for i, v in enumerate(mask_values):
        out[mask == i] = v

    return Image.fromarray(out)

def plot_img_and_mask(img, mask):
    # classes = mask.max() + 1
    classes = 1
    fig, ax = plt.subplots(1, classes+1)
    ax[0].set_title('Input image')
    ax[0].imshow(img)
    for i in range(classes):
        ax[i + 1].set_title(f'Mask (class {i + 1})')
        ax[i + 1].imshow(mask == i)
    plt.xticks([]), plt.yticks([])
    plt.show()


def mask_to_tif(filename, mask, tif_out):
    ref = gdal.Open(filename)
    geo_transform = ref.GetGeoTransform()
    projection = ref.GetProjection()

    h, w = mask.shape

    driver = gdal.GetDriverByName("GTiff")
    out_ds = driver.Create(
        tif_out,
        w,
        h,
        1,
        gdal.GDT_Byte
    )

    out_ds.SetGeoTransform(geo_transform)
    out_ds.SetProjection(projection)
    out_ds.GetRasterBand(1).WriteArray(mask)
    out_ds.FlushCache()

    del out_ds

def parse_args():
    parser = argparse.ArgumentParser(description="Predict masks from input images")
    parser.add_argument('--model-name', type=str, default='treemortseg', help='Model name for get_model()')
    parser.add_argument('--in-channels', type=int, default=4, help='Input channels')
    parser.add_argument('--num-classes', type=int, default=1, help='Number of output classes')
    parser.add_argument('--normalize', action='store_true', default=True, help='Normalize NAIP')

    parser.add_argument('--model', '-m', default='checkpoints/treemortseg_best_loss.pth', help='Model checkpoint path')
    parser.add_argument('--input', '-i', nargs='+', default=[
        'experiments/naip/WI00.tif',
    ], help='Input NAIP tif files')

    parser.add_argument('--output', '-o', nargs='+', help='Output filenames')
    parser.add_argument('--out-dir', type=str, default='', help='Output directory')
    # ========= 推理相关 =========
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5, help='Threshold for binary segmentation')
    parser.add_argument('--tile-size', type=int, default=224, help='Inference tile size')
    parser.add_argument('--overlap', type=int, default=112, help='Tile overlap')
    # ========= 其他 =========
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--viz', '-v', action='store_true', help='Visualize the images as they are processed')
    return parser.parse_args()


if __name__ == '__main__':
    # Parse command-line args and load config
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    in_files = args.input
    out_files = get_output_filenames(args)

    cfg = {
        "model": {
            "name": args.model_name,
            "num_classes": args.num_classes,
            "in_channels": args.in_channels,
            "deeplab_pretrained": {
                "pretrained": True,
                "backbone": "resnet50"
            },
            "vit_pretrained": {
                "vit_weights": "google/vit-base-patch16-224-in21k",  # Pretrained model for ViT
                "vit_patch_size": 16
            },
            "segformer_pretrained": {
                "segformer_weights": "nvidia/segformer-b2-finetuned-ade-512-512"  # Pretrained model for SegFormer
            },
            "mask2former_pretrained":{
                "mask2former_weights": "facebook/mask2former-swin-tiny-ade-semantic"  # Pretrained model for Mask2Former
            },
            "convnext_pretrained": True
        },
        "data": {
            "normalize": args.normalize
        },
    }

    # Initialize model
    net = get_model(cfg['model'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    net.to(device=device)

    checkpoint = torch.load(args.model, map_location=device)
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
        logging.info('Extracted model dict from complex checkpoint structure.')
    else:
        state_dict = checkpoint
        logging.info('Loaded raw state dict or simple model file.')

    mask_values = state_dict.pop('mask_values', [0, 1])
    net.load_state_dict(state_dict)

    logging.info('Model loaded!')

    for i, naip_path in enumerate(in_files):
        logging.info(f'Predicting image {naip_path} ...')

        naip, no_data_mask = load_naip_tile(naip_path,cfg['data']['normalize'])

        num_classes = cfg['model']['num_classes']

        output = predict_img(net=net, naip=naip, num_classes=num_classes, device=device,tile_size=args.tile_size, overlap=args.overlap)

        if cfg['model']['num_classes'] > 1:
            mask = output.argmax(dim=1)
        else:
            mask = torch.sigmoid(output) > args.mask_threshold
        mask = mask[0].long().squeeze().numpy()
        mask = mask * ~no_data_mask


        if not args.no_save:
            out_filename = out_files[i] + ".png"
            result = mask_to_image(mask, mask_values)
            result.save(out_filename, format="PNG")
            logging.info(f'Mask saved to {out_filename}')
            tif_out = out_filename.replace(".png", ".tif")
            mask_to_tif(naip_path, (mask * 255).astype(np.uint8), tif_out)


        if args.viz:
            logging.info(f'Visualizing results for image {naip_path}, close to continue...')
            plot_img_and_mask(np.transpose(naip[:3,...],[1,2,0]), mask)