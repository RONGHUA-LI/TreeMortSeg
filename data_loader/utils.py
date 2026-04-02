import rasterio
import numpy as np
import random


def load_naip_tile(path: str, no_data_value: int = 255, normalize: bool = True):
    """
    NAIP tile
    return:
      img_naip: [4, H, W]  (RGBN + NDVI)
      label:    [H, W]
      no_data_mask: [H, W]
    """
    # Read all bands: [R, G, B, NIR, Label]
    arr = rasterio.open(path).read()  # [5,H,W]

    # Separate image and raw label
    img = arr[:4, ...].astype(np.float32)  # RGBNIR
    raw_label = arr[4, ...].astype(np.uint8)  # uint8: 0,1,255

    # Create no-data mask from raw_label == 255
    no_data_mask = (raw_label == no_data_value)

    # Clean label: map 255 -> 0 (background) but keep mask for exclusion
    label = raw_label.copy()
    label[no_data_mask] = 0

    if normalize:
        # Normalize to [0, 1]
        img = img / 255.0

    return img, label, no_data_mask


def augment_naip_tile(img: np.ndarray, label: np.ndarray, no_data_mask: np.ndarray, aug_cfg: dict):
    """Apply flips and 90° rotations consistently to image, label, and mask."""

    # Random flips
    if aug_cfg.get('random_flip', False):
        if random.random() < 0.5:
            # horizontal flip (axis W)
            img = img[:, :, ::-1]
            label = label[:, :, ::-1]
            no_data_mask = no_data_mask[:, ::-1]
        if random.random() < 0.5:
            # vertical flip (axis H)
            img = img[:, ::-1, :]
            label = label[:, ::-1, :]
            no_data_mask = no_data_mask[::-1, :]

    # Discrete 90° rotations
    if aug_cfg.get('rotation', {}).get('type') == '90':
        k = random.choice([0, 1, 2, 3])  # number of 90° rotations
        img = np.rot90(img, k, axes=(1, 2))
        label = np.rot90(label, k, axes=(1, 2))
        no_data_mask = np.rot90(no_data_mask, k)

    return img.copy(), label.copy(), no_data_mask.copy()

