import random

import numpy as np
import rasterio


def load_naip_tile(path: str, no_data_value: int = 255, normalize: bool = True):
    """
    Load a NAIP tile and segmentation label.

    Returns:
        img: Four-band image array with shape [4, H, W].
        label: Binary label array with shape [H, W].
        no_data_mask: Boolean no-data mask with shape [H, W].
    """
    arr = rasterio.open(path).read()

    img = arr[:4, ...].astype(np.float32)
    raw_label = arr[4, ...].astype(np.uint8)

    no_data_mask = raw_label == no_data_value

    label = raw_label.copy()
    label[no_data_mask] = 0

    if normalize:
        img = img / 255.0

    return img, label, no_data_mask


def augment_naip_tile(
    img: np.ndarray,
    label: np.ndarray,
    no_data_mask: np.ndarray,
    aug_cfg: dict,
):
    """Apply flips and quarter-turn rotations to image, labels, and mask."""

    if aug_cfg.get("random_flip", False):
        if random.random() < 0.5:
            img = img[:, :, ::-1]
            label = label[:, :, ::-1]
            no_data_mask = no_data_mask[:, ::-1]
        if random.random() < 0.5:
            img = img[:, ::-1, :]
            label = label[:, ::-1, :]
            no_data_mask = no_data_mask[::-1, :]

    if aug_cfg.get("rotation", {}).get("type") == "90":
        k = random.choice([0, 1, 2, 3])
        img = np.rot90(img, k, axes=(1, 2))
        label = np.rot90(label, k, axes=(1, 2))
        no_data_mask = np.rot90(no_data_mask, k)

    return img.copy(), label.copy(), no_data_mask.copy()
