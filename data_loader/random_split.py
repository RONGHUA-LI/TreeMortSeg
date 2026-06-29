import os

import numpy as np
import pandas as pd
import rasterio
import torch
from scipy.ndimage import distance_transform_edt as edt
from torch.utils.data import Dataset

from utils.data_utils import augment_naip_tile, load_naip_tile


def _split_ratios(cfg: dict):
    split_cfg = cfg["data"]["split"]["random"]
    return (
        split_cfg["train_ratio"],
        split_cfg["test_ratio"],
        split_cfg["train_val_ratio"],
    )


def _metadata_path(cfg: dict) -> str:
    return os.path.join(cfg["data"]["root_dir"], cfg["data"]["dataset_info"])


def _shuffle_column(cfg: dict) -> str:
    if cfg["data"]["split"]["random"].get("shuffle_by_tile", True):
        return "FileName"
    return "ImageRawPath"


def _tile_paths(cfg: dict, metadata: pd.DataFrame):
    dataset_dir = os.path.join(cfg["data"]["root_dir"], cfg["data"]["dataset_dir"])
    return [
        os.path.join(dataset_dir, row["FileName"]) for _, row in metadata.iterrows()
    ]


def _balanced_tiles(trainval_df: pd.DataFrame, cfg: dict, rng):
    split_cfg = cfg["data"]["split"]
    pos_threshold = split_cfg.get("pos_threshold", 0)
    pos_frac = split_cfg.get("pos_frac", 0)

    is_positive = trainval_df["LabelSize"] >= pos_threshold
    positive_tiles = trainval_df[is_positive]["FileName"].unique().tolist()
    negative_tiles = trainval_df[~is_positive]["FileName"].unique().tolist()
    rng.shuffle(positive_tiles)
    rng.shuffle(negative_tiles)

    if pos_frac > 0:
        trainval_count = len(trainval_df)
        positive_count = min(int(pos_frac * trainval_count), len(positive_tiles))
        negative_count = min(
            int(positive_count / pos_frac - positive_count),
            len(negative_tiles),
        )
    else:
        positive_count = len(positive_tiles)
        negative_count = len(negative_tiles)

    selected_tiles = positive_tiles[:positive_count] + negative_tiles[:negative_count]
    rng.shuffle(selected_tiles)
    return selected_tiles


def _split_tile_names(cfg: dict, metadata: pd.DataFrame):
    train_ratio, test_ratio, train_val_ratio = _split_ratios(cfg)

    assert 0 < train_ratio < 1 and 0 <= test_ratio < 1
    assert train_ratio + test_ratio <= 1.0

    shuffle_column = _shuffle_column(cfg)
    shuffled_keys = metadata[shuffle_column].unique().tolist()
    rng = np.random.RandomState(cfg["experiment"]["seed"])
    rng.shuffle(shuffled_keys)

    total_count = len(shuffled_keys)
    test_count = int(test_ratio * total_count)
    test_keys = shuffled_keys[:test_count]
    test_df = metadata[metadata[shuffle_column].isin(test_keys)]
    test_tiles = test_df["FileName"].unique().tolist()

    trainval_count = int(train_ratio * total_count)
    trainval_keys = shuffled_keys[test_count : test_count + trainval_count]
    trainval_df = metadata[metadata[shuffle_column].isin(trainval_keys)]
    balanced_tiles = _balanced_tiles(trainval_df, cfg, rng)

    val_count = int(train_val_ratio * len(balanced_tiles))
    return {
        "train": balanced_tiles[val_count:],
        "val": balanced_tiles[:val_count],
        "test": test_tiles,
    }


def _load_auxiliary_labels(edge_base: str, dist_base: str, filename: str):
    edge_path = os.path.join(edge_base, filename.replace(".tif", "_edge.tif"))
    dist_path = os.path.join(dist_base, filename.replace(".tif", "_dist.tif"))

    with rasterio.open(edge_path) as dataset:
        edge_label = dataset.read(1)
    with rasterio.open(dist_path) as dataset:
        dist_label = dataset.read(1)

    return edge_label, dist_label


class RandomSplitDataset(Dataset):
    def __init__(self, cfg: dict, split: str = "train"):
        self.cfg = cfg
        self.split = split

        info = pd.read_csv(_metadata_path(cfg))
        split_tiles = _split_tile_names(cfg, info)[split]
        df = info[info["FileName"].isin(split_tiles)]
        self.paths = _tile_paths(cfg, df)

        self.edge_base = os.path.join(cfg["data"]["root_dir"], "edge_labels")
        self.dist_base = os.path.join(cfg["data"]["root_dir"], "dist_labels")

        if self.split == "test":
            # Preserve tile-level metadata needed by downstream test metrics.
            self.tree_types = df.set_index("FileName")["TreeTypes"].to_dict()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int):
        naip_path = self.paths[idx]
        filename = os.path.basename(naip_path)

        naip, label, no_data_mask = load_naip_tile(
            naip_path,
            no_data_value=self.cfg["data"]["no_data_value"],
            normalize=self.cfg["data"]["normalize"],
        )

        edge_label, dist_label = _load_auxiliary_labels(
            self.edge_base,
            self.dist_base,
            filename,
        )
        boundary_dist = edt(1 - edge_label)

        if self.split == "train":
            combined_labels = np.stack(
                [label, edge_label, dist_label, boundary_dist],
                axis=0,
            )
            naip, combined_labels, no_data_mask = augment_naip_tile(
                naip,
                combined_labels,
                no_data_mask,
                self.cfg["data"]["augmentation"],
            )
            label, edge_label, dist_label, boundary_dist = combined_labels

        res = {
            "naip": torch.from_numpy(naip).float(),
            "label": torch.from_numpy(label).long(),
            "edge": torch.from_numpy(edge_label).float(),
            "dist": torch.from_numpy(dist_label).float(),
            "boundary_dist": torch.from_numpy(boundary_dist).float(),
            "no_data_mask": torch.from_numpy(no_data_mask).bool(),
        }

        if self.split == "test":
            res["tree_type"] = str(self.tree_types.get(filename, "None"))

        return res
