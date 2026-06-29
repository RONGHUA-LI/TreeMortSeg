import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import rasterio
import scipy.ndimage
import torch
from rasterio.enums import Resampling
from rasterio.windows import Window
from tqdm import tqdm

from model.treemortseg import build_treemortseg

from utils.inference_utils import (
    count_dead_tree_patches,
    generate_gaussian_window,
    get_input_and_output_pairs,
)

BAND_INDEXES = [1, 2, 3, 4]
MASK_THRESHOLD = 0.5
TARGET_RESOLUTION = 0.6


def setup_logging(output_dir: Path) -> None:
    """Configure file and console logging for inference."""
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "global_inference.log"

    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_formatter = logging.Formatter("%(levelname)s: %(message)s")

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    has_file_handler = any(
        isinstance(h, logging.FileHandler) and h.baseFilename == str(log_file.resolve())
        for h in root_logger.handlers
    )
    if not has_file_handler:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

    has_stream_handler = any(
        isinstance(h, logging.StreamHandler)
        and not isinstance(h, logging.FileHandler)
        and type(h) != logging.Handler
        for h in root_logger.handlers
    )
    if not has_stream_handler:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)

    logging.info(f"Global log initialized at {log_file}")


def _requires_resampling(res_x, res_y):
    return not (
        np.isclose(res_x, TARGET_RESOLUTION, atol=1e-3)
        and np.isclose(res_y, TARGET_RESOLUTION, atol=1e-3)
    )


def _processing_shape(src, need_resample):
    if not need_resample:
        return src.height, src.width

    height = int(round(src.height * src.res[1] / TARGET_RESOLUTION))
    width = int(round(src.width * src.res[0] / TARGET_RESOLUTION))
    return height, width


def _read_tile(src, x, y, width, height, full_width, full_height, need_resample):
    if not need_resample:
        window = Window(x, y, width, height)
        return src.read(BAND_INDEXES, window=window)

    source_window = Window(
        x * (src.width / full_width),
        y * (src.height / full_height),
        width * (src.width / full_width),
        height * (src.height / full_height),
    )
    return src.read(
        BAND_INDEXES,
        window=source_window,
        out_shape=(len(BAND_INDEXES), height, width),
        resampling=Resampling.bilinear,
    )


def _pad_tile(tile_img, tile_size, height, width):
    pad_h = tile_size - height
    pad_w = tile_size - width
    pad_top = pad_h // 2
    pad_left = pad_w // 2

    if pad_h > 0 or pad_w > 0:
        tile_img = np.pad(
            tile_img,
            (
                (0, 0),
                (pad_top, pad_h - pad_top),
                (pad_left, pad_w - pad_left),
            ),
        )

    return tile_img, pad_top, pad_left


def _flush_batch(
    net,
    batch_data,
    batch_coords,
    accum_mask,
    count_mask,
    gaussian_weight,
    device,
    progress_bar,
):
    if not batch_data:
        return [], []

    process_batch(
        net,
        batch_data,
        batch_coords,
        accum_mask,
        count_mask,
        gaussian_weight,
        device,
    )
    progress_bar.update(len(batch_data))
    return [], []


def process_batch(
    net, batch_data, batch_coords, accum_mask, count_mask, gaussian_weight, device
):
    """Run one inference batch and blend predictions into the output mask."""
    inputs = torch.from_numpy(np.stack(batch_data)).to(device, dtype=torch.float32)

    with torch.no_grad():
        outputs = net(inputs)
        if isinstance(outputs, (tuple, list)):
            outputs = outputs[0]

        probs = torch.sigmoid(outputs).cpu().squeeze(1)
        probs_np = probs.numpy()

        for i in range(len(batch_coords)):
            x, y, w, h, pad_top, pad_left = batch_coords[i]

            prob_tile = probs_np[i, pad_top : pad_top + h, pad_left : pad_left + w]
            weight_tile = gaussian_weight[
                pad_top : pad_top + h, pad_left : pad_left + w
            ]

            accum_mask[y : y + h, x : x + w] += prob_tile * weight_tile
            count_mask[y : y + h, x : x + w] += weight_tile


def _blend_probability_mask(accum_mask, count_mask):
    valid_mask = count_mask > 1e-5
    accum_mask[valid_mask] /= count_mask[valid_mask]

    final_mask = np.zeros(accum_mask.shape, dtype=np.uint8)
    final_mask[accum_mask > MASK_THRESHOLD] = 255
    return final_mask


def _read_full_image(src, height, width, need_resample):
    if not need_resample:
        return src.read(BAND_INDEXES)

    return src.read(
        BAND_INDEXES,
        out_shape=(len(BAND_INDEXES), height, width),
        resampling=Resampling.bilinear,
    )


def _restore_original_shape(mask, original_height, original_width):
    if mask.shape == (original_height, original_width):
        return mask

    restored_mask = scipy.ndimage.zoom(
        mask,
        (original_height / mask.shape[0], original_width / mask.shape[1]),
        order=0,
    )

    if (
        restored_mask.shape[0] == original_height
        and restored_mask.shape[1] == original_width
    ):
        return restored_mask

    aligned_mask = np.zeros((original_height, original_width), dtype=np.uint8)
    copy_height = min(original_height, restored_mask.shape[0])
    copy_width = min(original_width, restored_mask.shape[1])
    aligned_mask[:copy_height, :copy_width] = restored_mask[:copy_height, :copy_width]
    return aligned_mask


def _write_output_mask(src, output_tif, mask):
    profile = src.profile.copy()
    profile.update({"dtype": "uint8", "count": 1, "compress": "lzw", "nodata": None})

    os.makedirs(os.path.dirname(output_tif), exist_ok=True)

    with rasterio.open(output_tif, "w", **profile) as dst:
        dst.write(mask, 1)


def inference_tif(
    net,
    input_path,
    output_tif,
    device,
    tile_size,
    overlap,
    batch_size,
    current_idx,
    total_count,
    display_path,
):
    """Run tiled inference for one TIFF file and write the predicted mask."""
    net.eval()
    stride = tile_size - overlap
    gaussian_weight = generate_gaussian_window(tile_size)

    with rasterio.open(input_path) as src:
        orig_res_x, orig_res_y = src.res
        orig_h, orig_w = src.height, src.width

        need_resample = _requires_resampling(orig_res_x, orig_res_y)
        height, width = _processing_shape(src, need_resample)

        if need_resample:
            logging.info(
                f"[{current_idx}/{total_count}] Image resolution is "
                f"{orig_res_x:.1f}m; resampling to {TARGET_RESOLUTION:.1f}m "
                "for processing."
            )

        accum_mask = np.zeros((height, width), dtype=np.float32)
        count_mask = np.zeros((height, width), dtype=np.float32)

        rows = range(0, height, stride)
        cols = range(0, width, stride)

        desc_str = f"[{current_idx}/{total_count}] {display_path}"
        pbar = tqdm(total=len(rows) * len(cols), desc=desc_str, file=sys.stdout)

        batch_data = []
        batch_coords = []

        try:
            for y in rows:
                for x in cols:
                    w = min(tile_size, width - x)
                    h = min(tile_size, height - y)

                    tile_img_raw = _read_tile(
                        src,
                        x,
                        y,
                        w,
                        h,
                        width,
                        height,
                        need_resample,
                    )

                    if np.all(tile_img_raw == 0):
                        count_mask[y : y + h, x : x + w] += gaussian_weight[:h, :w]
                        pbar.update(1)
                        continue

                    tile_img = tile_img_raw.astype(np.float32) / 255.0
                    tile_img, pad_top, pad_left = _pad_tile(tile_img, tile_size, h, w)

                    batch_data.append(tile_img)
                    batch_coords.append((x, y, w, h, pad_top, pad_left))

                    if len(batch_data) >= batch_size:
                        batch_data, batch_coords = _flush_batch(
                            net,
                            batch_data,
                            batch_coords,
                            accum_mask,
                            count_mask,
                            gaussian_weight,
                            device,
                            pbar,
                        )

            _flush_batch(
                net,
                batch_data,
                batch_coords,
                accum_mask,
                count_mask,
                gaussian_weight,
                device,
                pbar,
            )

        finally:
            pbar.close()

        final_mask = _blend_probability_mask(accum_mask, count_mask)
        big_img_raw = _read_full_image(src, height, width, need_resample)
        is_zero_pixel = np.all(big_img_raw[:3] == 0, axis=0)
        final_mask[is_zero_pixel] = 0

        if need_resample:
            final_mask_orig = _restore_original_shape(final_mask, orig_h, orig_w)
        else:
            final_mask_orig = final_mask

        _write_output_mask(src, output_tif, final_mask_orig)

        return count_dead_tree_patches(final_mask_orig)


def run_inference(
    model_path,
    input_paths,
    output_dir,
    in_channels,
    tile_size,
    overlap,
    batch_size,
):
    """Run model inference for every input TIFF path."""
    setup_logging(Path(output_dir))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    in_files, out_files = get_input_and_output_pairs(input_paths, output_dir)

    if not in_files:
        logging.error("No valid .tif files found to process. Exiting.")
        exit(1)

    logging.info(f"Total files to process: {len(in_files)}")

    net = build_treemortseg({"in_channels": in_channels})
    checkpoint = torch.load(model_path, map_location=device)
    net.load_state_dict(checkpoint["model"] if "model" in checkpoint else checkpoint)
    net.to(device)
    logging.info("Model loaded successfully.")

    calculated_overlap = int(tile_size * overlap)
    total_files = len(in_files)

    for i, file in enumerate(in_files):
        rel_display_path = os.path.basename(file)

        num_patches = inference_tif(
            net,
            file,
            out_files[i],
            device,
            tile_size,
            calculated_overlap,
            batch_size,
            current_idx=i + 1,
            total_count=total_files,
            display_path=rel_display_path,
        )

        logging.info(
            f"[{i + 1}/{total_files}] Dead tree patch count: {num_patches} "
            f"| Result saved: {os.path.abspath(out_files[i])}"
        )

    logging.info("All image inference completed.")


def parse_args():
    """Parse inference CLI arguments."""
    parser = argparse.ArgumentParser(description="Dead tree segmentation")

    parser.add_argument("--in-channels", type=int, default=4)
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        required=True,
        help="Path to the model weights file (.pth).",
    )
    parser.add_argument(
        "--input",
        "-i",
        nargs="+",
        required=True,
        help="Input TIFF image paths or folders containing TIFF images.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Output folder for segmentation results.",
    )

    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--overlap", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=16)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    run_inference(
        model_path=args.model,
        input_paths=args.input,
        output_dir=args.output,
        in_channels=args.in_channels,
        tile_size=args.tile_size,
        overlap=args.overlap,
        batch_size=args.batch_size,
    )
