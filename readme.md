# TreeMortSeg: Individual tree mortality segmentation using a multi-task deep learning network and high-resolution aerial imagery

## Overview

TreeMortSeg is a multi-task learning framework designed for dead tree segmentation from high-resolution aerial imagery (e.g., NAIP). 

The repository provides:

- **Pre-trained Model:** Model weights.
- **Training Code:** Scripts to train your own models from scratch.
- **Inference Pipeline:** Tools to run the model on one or more GeoTIFF images using a sliding-window approach, export binary segmentation masks and count dead tree patches.
- **Pre-trained Models:** Ready-to-use model weights to get started immediately.
- **Evaluation Tools:** Scripts to test and evaluate your saved checkpoints.

## Model Architecture

![TreeMortSeg Architecture](https://github.com/RONGHUA-LI/TreeMortSeg/blob/main/treemortseg%20architecture.png)

The TreeMortSeg model utilizes a multi-task learning architecture that incorporates structure-aware and edge-aware auxiliary pathways, each equipped with dedicated decoders to extract fine-grained morphological and boundary features. These auxiliary features are integrated as localized spatial priors through a parallel fusion mechanism to guide the primary segmentation task. This design significantly improves the model's sensitivity and precision when identifying small-scale targets and delineating the complex boundaries of dead trees.

## Project Structure

```text
TreeMortSeg/
├── configs/
|   └── res_random.yaml        # Default experiment configuration
├── data_loader/
|   ├── __init__.py            # DataLoader factory
|   └── random_split.py        # Random split dataset implementation
├── model/
|   └── treemortseg.py         # TreeMortSeg network definition
├── scripts/
|   ├── evaluate.py            # Standalone evaluation entry point
|   └── inference.py           # Sliding-window GeoTIFF inference entry point
├── utils/
|   ├── data_utils.py          # Tile loading and augmentation helpers
|   ├── inference_utils.py     # Inference path and mask utilities
|   ├── losses.py              # Dice, focal, and edge boundary losses
|   ├── metrics.py             # IoU and confusion-matrix metrics
|   ├── tools.py               # CLI, config, logging, seed, and result helpers
|   └── train.py               # Training and validation loops
├── main.py                    # Full train-and-evaluate pipeline
├── readme.md
└── requirements.txt
```

## Installation & Prerequisites

### Requirements

- Python 3.10 or newer is recommended.
- A CUDA-capable GPU is recommended for training and inference.

### Environment Setup

1. Clone this repository to your local machine:

```bash
git clone https://github.com/RONGHUA-LI/TreeMortSeg.git
cd TreeMortSeg
```

2. Install dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Dataset Expectations

The TreeFinder dataset can be downloaded from [GitHub - zhwang0/treefinder](https://github.com/zhwang0/treefinder).<br>

The default dataset loader expects a root directory with the following layout:

```text
dataset_root/
├── tiles224_v3/
│   ├── tile_001.tif           # 5-band raster (RGB + NIR + Label)
│   ├── tile_002.tif
│   └── ...
├── edge_labels/
│   ├── tile_001_edge.tif      # Extracted edge boundary mask
│   ├── tile_002_edge.tif
│   └── ...
├── dist_labels/
│   ├── tile_001_dist.tif      # Calculated distance transform representation
│   ├── tile_002_dist.tif
│   └── ...
└── tile_info224_v3_aux.csv    # Metadata catalog containing tile records
```

The label band uses `0` for background, `1` for dead-tree pixels, and `255` by default for no-data pixels.

The auxiliary `edge_labels` and `dist_labels` can be obtained by the following:

```bash
python utils/aux_label_gen.py --config configs/res_random.yaml
```

## Configuration

The main configuration file is `configs/res_random.yaml`.

Minimal configuration edit before training:

```yaml
data:
  root_dir: "path/to/treefinder/"
```

## Usage

### 1. Training and Validation Pipeline

To train the TreeMortSeg model from scratch and run a evaluation automatically,:

```bash
python main.py --config configs/res_random.yaml
```

Expected outputs:

```text
checkpoints/
└── exp001_treemortseg/
    ├── exp001_treemortseg_best.pth
    └── exp001_treemortseg_last.pth

results/
└── exp001_treemortseg/
    ├── exp001_treemortseg_metrics.yaml
    └── exp001_treemortseg_loss_curve.png

logs/
└── exp001_treemortseg.log
```

Override selected configuration values from the command line:

```bash
python main.py \
  --config configs/res_random.yaml \
  --overwrite_cfg True \
  --exp_id 002 \
  --gpu_id 0 \
  --train_ratio 0.8 \
  --random_seed 2025
```

### 2. Standalone Evaluation

Evaluate the best checkpoint for an experiment:

```bash
python scripts/evaluate.py --config configs/res_random.yaml
```

It writes metrics to:

```text
results/exp001_treemortseg/exp001_treemortseg_metrics.yaml
```

### 3. Sliding-Window GeoTIFF Inference

Run inference on single GeoTIFF file:

```bash
python scripts/inference.py \
  --model checkpoints/exp001_treemortseg/exp001_treemortseg_best.pth \
  --input D:/path/to/input_image.tif \
  --output D:/path/to/inference_results \
  --in-channels 4 \
  --tile-size 256 \
  --overlap 0.2 \
  --batch-size 16
```

Run inference on every `.tif` or `.tiff` file under a directory:

```bash
python scripts/inference.py \
  --model checkpoints/exp001_treemortseg/exp001_treemortseg_best.pth \
  --input D:/path/to/input_folder \
  --output D:/path/to/inference_results \
  --batch-size 16
```

Inference outputs:

| Output | Description |
| --- | --- |
| Binary mask GeoTIFF | Pixels above probability threshold are written as `255`; background is `0` |
| `global_inference.log` | Per-file progress, output path, and dead tree patch count |

## Notes on Reproducibility

- `experiment.seed` controls NumPy, Python, and PyTorch random seeds.
- The random split is generated from the configured metadata CSV and seed.
- CUDA operations can still have hardware-specific nondeterminism depending on the local PyTorch setup.
- The inference script resamples input imagery to 0.6 m when source resolution differs from that target.

## Citation

If you use this repository, please cite:

```bibtex
@article{
  title={Individual tree mortality segmentation using a multi-task deep learning network and high-resolution aerial imagery},
  author={Ronghua Li et al.},
  journal={Journal Name},
  year={2026},
  volume={X},
  number={X},
  pages={XX--XX},
  publisher={Publisher}
}
```

## License
Copyright (C) 2026 Ronghua Li

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.