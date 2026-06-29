import argparse
import glob
import os
from pathlib import Path

import numpy as np
import rasterio
from scipy import ndimage
from skimage import measure
from tqdm import tqdm
import yaml
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config(config_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    input_dir = config.get("data", {}).get("root_dir")
    if not input_dir:
        raise ValueError("The 'data.root_dir' key is missing in the config file.")

    output_dir = os.path.dirname(os.path.normpath(input_dir))
    return input_dir, output_dir


def generate_edge_mask(binary_label):
    structure = ndimage.generate_binary_structure(2, 2)
    erosion = ndimage.binary_erosion(binary_label, structure=structure)
    edge = binary_label - erosion
    return np.where(edge == 1, 255, 0).astype(np.uint8)


def generate_distance_map(binary_label):
    distance_map = np.zeros_like(binary_label, dtype=np.float32)
    instance_map = measure.label(binary_label, connectivity=2)
    num_instances = instance_map.max()

    for i in range(1, num_instances + 1):
        mask_i = (instance_map == i)
        dist_i = ndimage.distance_transform_edt(mask_i)
        max_dist = dist_i.max()
        if max_dist > 0:
            distance_map[mask_i] = dist_i[mask_i] / max_dist

    return distance_map


def process_single_tile(tif_path, edge_out_dir, dist_out_dir):
    filename = os.path.basename(tif_path)

    with rasterio.open(tif_path) as src:
        profile = src.profile
        data = src.read()
        # Extract the last channel as the raw label map
        label_raw = data[-1, :, :]
        binary_label = np.where(label_raw == 255, 1, 0).astype(np.uint8)

    edge_label = generate_edge_mask(binary_label)
    dist_label = generate_distance_map(binary_label)

    # Configure profiles for single-channel output
    edge_profile = profile.copy()
    dist_profile = profile.copy()
    edge_profile.update(count=1, nodata=None)
    dist_profile.update(count=1, dtype="float32", nodata=None)

    edge_save_path = os.path.join(edge_out_dir, filename.replace(".tif", "_edge.tif"))
    dist_save_path = os.path.join(dist_out_dir, filename.replace(".tif", "_dist.tif"))

    with rasterio.open(edge_save_path, "w", **edge_profile) as dst:
        dst.write(edge_label, 1)
    with rasterio.open(dist_save_path, "w", **dist_profile) as dst:
        dst.write(dist_label, 1)


def batch_generate_labels(input_dir, output_dir):
    edge_out_dir = os.path.join(output_dir, "edge_labels")
    dist_out_dir = os.path.join(output_dir, "dist_labels")
    os.makedirs(edge_out_dir, exist_ok=True)
    os.makedirs(dist_out_dir, exist_ok=True)

    tif_files = glob.glob(os.path.join(input_dir, "*.tif"))
    print(f"Input Directory: {input_dir}")
    print(f"Found {len(tif_files)} files. Starting processing pipeline...")

    if not tif_files:
        print("Warning: No .tif files found in the specified input directory.")
        return

    for tif_path in tqdm(tif_files, desc="Processing Tiles"):
        try:
            process_single_tile(tif_path, edge_out_dir, dist_out_dir)
        except Exception as e:
            print(f"\nSkipped file {tif_path} due to error: {e}")
            continue

    print(f"\nTask completed successfully! Results saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate MTL targets from tree mortality labels.")
    parser.add_argument(
        "-c", "--config",
        type=str,
        default=f"{PROJECT_ROOT}/configs/res_random.yaml",
        help="Path to the YAML configuration file (default: res_random.yaml)"
    )
    args = parser.parse_args()

    input_path, output_path = load_config(args.config)
    batch_generate_labels(input_path, output_path)