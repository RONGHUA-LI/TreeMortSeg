import glob
import logging
import os

import numpy as np
from scipy.ndimage import label


def generate_gaussian_window(tile_size, sigma_scale=0.5):
    """Generate a 2D Gaussian weight matrix for tile-edge smoothing."""
    x = np.linspace(-1, 1, tile_size)
    sigma = sigma_scale
    gauss_1d = np.exp(-0.5 * (x / sigma) ** 2)
    gauss_2d = np.outer(gauss_1d, gauss_1d)
    return gauss_2d.astype(np.float32)


def get_input_and_output_pairs(input_paths, target_dir):
    """Collect input TIFF files and map them to output paths."""
    input_files = []
    out_files = []

    for path in input_paths:
        if os.path.isdir(path):
            tifs = glob.glob(
                os.path.join(path, "**", "*.tif"),
                recursive=True,
            ) + glob.glob(
                os.path.join(path, "**", "*.tiff"),
                recursive=True,
            )

            for file_path in tifs:
                input_files.append(file_path)
                rel_path = os.path.relpath(file_path, path)
                rel_base = os.path.splitext(rel_path)[0]
                out_path = os.path.join(target_dir, f"{rel_base}.tif")
                out_files.append(out_path)

        elif os.path.isfile(path):
            input_files.append(path)
            base = os.path.splitext(os.path.basename(path))[0]
            out_files.append(os.path.join(target_dir, f"{base}.tif"))
        else:
            logging.warning(f"Path not found or invalid: {path}")

    return input_files, out_files


def count_dead_tree_patches(mask):
    """Count connected dead-tree patches in a binary mask."""
    _, num_features = label(mask > 0)
    return num_features
