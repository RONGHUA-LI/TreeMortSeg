from torch.utils.data import DataLoader


def get_dataloader(cfg: dict):
    """Return train, val, test DataLoaders based on split method and scenario."""
    method = cfg["data"]["split"]["method"]
    if method == "random":
        from .random_split import RandomSplitDataset as DatasetClass
    else:
        raise ValueError(f"Unknown split method: {method}")

    train_ds = DatasetClass(cfg, split="train")
    val_ds = DatasetClass(cfg, split="val")
    test_ds = DatasetClass(cfg, split="test")

    batch_size = cfg["training"].get("batch_size", 32)
    num_workers = cfg["experiment"].get("num_workers", 16)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2,
    )
    return train_loader, val_loader, test_loader
