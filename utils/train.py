import os
import time
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast

from utils.losses import DiceLoss, FocalLoss, EdgeBoundaryLoss
from utils.metrics import compute_iou_stats


def _loss_weights(cfg):
    return {
        "dice": cfg["training"].get("w_dice", 0.5),
        "mask": cfg["training"].get("w_mask", 1.0),
        "edge": cfg["training"].get("w_edge", 1.0),
        "dist": cfg["training"].get("w_dist", 5.0),
    }


def _batch_to_device(batch, device):
    return {
        "naips": batch["naip"].to(device),
        "labels": batch["label"].unsqueeze(1).float().to(device),
        "edges": batch["edge"].unsqueeze(1).to(device),
        "boundary_dists": batch["boundary_dist"].unsqueeze(1).to(device),
        "dists": batch["dist"].unsqueeze(1).to(device),
        "valid_mask": (~batch["no_data_mask"].unsqueeze(1)).float().to(device),
    }


def _model_outputs(model, images):
    outputs = model(images)
    if not isinstance(outputs, (tuple, list)):
        outputs = (outputs,)
    return (
        outputs[0],
        outputs[1] if len(outputs) > 1 else None,
        outputs[2] if len(outputs) > 2 else None,
    )


def _zero_loss(device):
    return torch.tensor(0.0, device=device)


def _compute_losses(
    outputs,
    batch,
    criterion,
    dice_loss,
    edge_criterion,
    dist_criterion,
    weights,
    device,
):
    out_masks, out_edges, out_dists = outputs

    raw_loss = criterion(out_masks, batch["labels"])
    bce_loss = (raw_loss * batch["valid_mask"]).sum() / batch["valid_mask"].sum()
    dice_value = (
        dice_loss(out_masks, batch["labels"], batch["valid_mask"])
        if weights["dice"] > 0
        else _zero_loss(device)
    )
    mask_loss = bce_loss + weights["dice"] * dice_value

    edge_loss = (
        edge_criterion(
            out_edges,
            batch["edges"],
            batch["boundary_dists"],
            batch["valid_mask"],
        )
        if out_edges is not None
        else _zero_loss(device)
    )

    if out_dists is not None:
        positive_mask = (batch["dists"] > 0).float()
        dist_loss = (
            dist_criterion(torch.sigmoid(out_dists), batch["dists"]) * positive_mask
        ).sum() / (positive_mask.sum() + 1e-8)
    else:
        dist_loss = _zero_loss(device)

    total_loss = (
        mask_loss * weights["mask"]
        + edge_loss * weights["edge"]
        + dist_loss * weights["dist"]
    )
    return {
        "total": total_loss,
        "mask": mask_loss,
        "edge": edge_loss,
        "dist": dist_loss,
    }


def _add_weighted_losses(metrics, losses, weights):
    metrics["total"] += losses["total"].item()
    metrics["mask"] += losses["mask"].item() * weights["mask"]
    metrics["edge"] += losses["edge"].item() * weights["edge"]
    metrics["dist"] += losses["dist"].item() * weights["dist"]


def _append_nonzero_loss(log_parts, label, value):
    if value != 0:
        log_parts.append(f"{label}: {value:.4f}")


def _build_criterion(training_cfg, device):
    criterion_type = training_cfg["criterion"].get("type", "BCEWithLogitsLoss")

    if criterion_type == "BCEWithLogitsLoss":
        if training_cfg["criterion"].get("w_pos", 0) > 0:
            pos_weight = torch.tensor(training_cfg["criterion"]["w_pos"], device=device)
            return nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")
        return nn.BCEWithLogitsLoss(reduction="none")

    if criterion_type == "FocalLoss":
        return FocalLoss(
            alpha=training_cfg["criterion"].get("alpha", 0.25),
            gamma=training_cfg["criterion"].get("gamma", 2.0),
        )

    raise ValueError(f"Unsupported loss type: {criterion_type}")


def _build_scheduler(training_cfg, optimizer):
    if "scheduler" not in training_cfg:
        return None

    scheduler_cfg = training_cfg["scheduler"]
    scheduler_type = scheduler_cfg["type"]
    scheduler_class = getattr(optim.lr_scheduler, scheduler_type)

    if scheduler_type == "ExponentialLR":
        return scheduler_class(optimizer, gamma=scheduler_cfg.get("gamma", 0.95))
    if scheduler_type == "StepLR":
        return scheduler_class(
            optimizer,
            step_size=scheduler_cfg.get("step_size", 10),
            gamma=scheduler_cfg.get("gamma", 0.1),
        )
    if scheduler_type == "CosineAnnealingWarmRestarts":
        return scheduler_class(
            optimizer,
            T_0=scheduler_cfg.get("T_0", 10),
            T_mult=scheduler_cfg.get("T_mult", 1),
            eta_min=float(scheduler_cfg.get("eta_min", 0)),
        )

    return None


def _save_checkpoint(
    checkpoint_path,
    epoch,
    model,
    optimizer,
    scheduler,
    epochs_no_improve,
    train_losses,
    val_losses,
    best_val,
):
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler else None,
            "epochs_no_improve": epochs_no_improve,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "best_val": best_val,
        },
        checkpoint_path,
    )


def _plot_loss_curve(train_losses, val_losses, output_path):
    plt.figure()
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss")
    plt.plot(range(1, len(val_losses) + 1), val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(output_path)
    plt.close()


def train_one_epoch(
    model,
    train_loader,
    optimizer,
    scaler,
    criterion,
    dice_loss,
    edge_criterion,
    dist_criterion,
    device,
    cfg,
    epoch,
    num_epochs,
    logger,
):
    """Train the model for one epoch and return the average total loss."""
    model.train()

    weights = _loss_weights(cfg)
    log_interval = cfg["logging"].get("log_interval", 100)

    metrics = {"total": 0.0, "mask": 0.0, "edge": 0.0, "dist": 0.0}
    step_times = []

    for step, batch in enumerate(train_loader, 1):
        step_start = time.time()

        batch_on_device = _batch_to_device(batch, device)
        optimizer.zero_grad()

        with autocast(device_type="cuda"):
            outputs = _model_outputs(model, batch_on_device["naips"])
            losses = _compute_losses(
                outputs,
                batch_on_device,
                criterion,
                dice_loss,
                edge_criterion,
                dist_criterion,
                weights,
                device,
            )

        scaler.scale(losses["total"]).backward()
        scaler.step(optimizer)
        scaler.update()

        _add_weighted_losses(metrics, losses, weights)
        step_times.append(time.time() - step_start)

        if step % log_interval == 0:
            log_parts = [
                f"[Epoch {epoch}/{num_epochs}]",
                f"Step {step}/{len(train_loader)}",
                f"Time Spent: {sum(step_times) / 60:.1f}m",
                f"Train Total Loss: {metrics['total'] / step:.4f}",
            ]
            _append_nonzero_loss(log_parts, "Mask Loss", metrics["mask"] / step)
            _append_nonzero_loss(log_parts, "Edge Loss", metrics["edge"] / step)
            _append_nonzero_loss(log_parts, "Dist Loss", metrics["dist"] / step)
            logger.info(" - ".join(log_parts))
            step_times = []

    return metrics["total"] / len(train_loader)


def validate_one_epoch(
    model,
    val_loader,
    criterion,
    dice_loss,
    edge_criterion,
    dist_criterion,
    device,
    cfg,
):
    """Validate the model for one epoch and return aggregate metrics."""
    model.eval()

    weights = _loss_weights(cfg)

    v_metrics = {"total": 0.0, "mask": 0.0, "edge": 0.0, "dist": 0.0}
    total_inter, total_union = 0.0, 0.0

    with torch.inference_mode(), autocast(device_type="cuda"):
        for batch in val_loader:
            batch_on_device = _batch_to_device(batch, device)
            outputs = _model_outputs(model, batch_on_device["naips"])
            losses = _compute_losses(
                outputs,
                batch_on_device,
                criterion,
                dice_loss,
                edge_criterion,
                dist_criterion,
                weights,
                device,
            )
            _add_weighted_losses(v_metrics, losses, weights)

            inter, union = compute_iou_stats(
                outputs[0],
                batch_on_device["labels"],
                batch_on_device["valid_mask"],
            )
            total_inter += inter.item()
            total_union += union.item()

    num_batches = len(val_loader)
    return {
        "total": v_metrics["total"] / num_batches,
        "mask": v_metrics["mask"] / num_batches,
        "edge": v_metrics["edge"] / num_batches,
        "dist": v_metrics["dist"] / num_batches,
        "iou": total_inter / (total_union + 1e-6),
    }


def train_model(model, train_loader, val_loader, cfg, exp_name):
    """Run the configured training loop and return the monitored best metric."""
    logger = logging.getLogger(__name__)
    gpu_id = cfg["experiment"].get("gpu_id", 0)
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logger.info(f"Using device: {device}")

    tr_cfg = cfg["training"]
    criterion = _build_criterion(tr_cfg, device)

    dice_loss = DiceLoss()
    edge_criterion = EdgeBoundaryLoss(lambda_boundary=tr_cfg.get("w_boundary", 0.2))
    dist_criterion = nn.L1Loss(reduction="none")
    scaler = GradScaler()

    optimizer = getattr(optim, tr_cfg["optimizer"]["type"])(
        model.parameters(),
        lr=float(tr_cfg["learning_rate"]),
        weight_decay=tr_cfg["optimizer"].get("weight_decay", 0),
    )

    scheduler = _build_scheduler(tr_cfg, optimizer)

    es_cfg = tr_cfg["early_stopping"]
    es_enabled = es_cfg.get("enabled", False)
    monitor = es_cfg.get("monitor", "val_loss")
    mode = es_cfg.get("mode", "min")
    patience = es_cfg.get("patience", 10)
    best_val = float("inf") if mode == "min" else -float("inf")
    improve = lambda cur, best: cur < best if mode == "min" else cur > best
    epochs_no_improve = 0

    ckpt_dir = Path(cfg["output"]["checkpoint_dir"]) / exp_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path(cfg["output"]["results_dir"]) / exp_name
    results_dir.mkdir(parents=True, exist_ok=True)

    train_losses, val_losses = [], []
    start_epoch = 1

    resume_path = tr_cfg.get("resume", {}).get("checkpoint", None)
    if resume_path and os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if scheduler and ckpt["scheduler"]:
            scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_val = ckpt["best_val"]
        epochs_no_improve = ckpt["epochs_no_improve"]
        train_losses = ckpt["train_losses"]
        val_losses = ckpt["val_losses"]
        logger.info(f"Resume from {resume_path}, start_epoch={start_epoch}")

    num_epochs = tr_cfg["epochs"]
    for epoch in range(start_epoch, num_epochs + 1):
        epoch_start = time.time()

        avg_train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            criterion,
            dice_loss,
            edge_criterion,
            dist_criterion,
            device,
            cfg,
            epoch,
            num_epochs,
            logger,
        )
        train_losses.append(avg_train_loss)

        val_time_start = time.time()
        val_metrics = validate_one_epoch(
            model,
            val_loader,
            criterion,
            dice_loss,
            edge_criterion,
            dist_criterion,
            device,
            cfg,
        )
        val_losses.append(val_metrics["total"])
        val_time = time.time() - val_time_start

        log_parts = [
            f"[Epoch {epoch}/{num_epochs}]",
            f"Time Spent: {val_time / 60:.1f}m",
            f"Val Total Loss: {val_metrics['total']:.4f}",
        ]
        _append_nonzero_loss(log_parts, "Mask Loss", val_metrics["mask"])
        _append_nonzero_loss(log_parts, "Edge Loss", val_metrics["edge"])
        _append_nonzero_loss(log_parts, "Dist Loss", val_metrics["dist"])
        log_parts.append(f"Val IoU: {val_metrics['iou']:.4f}")
        logger.info(" - ".join(log_parts))

        current_val = val_metrics["total"] if monitor == "val_loss" else None
        if current_val is not None and improve(current_val, best_val):
            best_val = current_val
            epochs_no_improve = 0
            torch.save(model.state_dict(), ckpt_dir / f"{exp_name}_best.pth")
            logger.info(
                f"--- New Best Val Loss Model Saved! (Val Loss: {best_val:.4f}) ---"
            )
        else:
            epochs_no_improve += 1
            if es_enabled:
                logger.info(
                    f"No improvement for {epochs_no_improve}/{patience} epochs."
                )
                if epochs_no_improve >= patience:
                    logger.info(f"Early stopping triggered at epoch {epoch}.")
                    break

        if scheduler:
            scheduler.step()

        _save_checkpoint(
            ckpt_dir / f"{exp_name}_last.pth",
            epoch,
            model,
            optimizer,
            scheduler,
            epochs_no_improve,
            train_losses,
            val_losses,
            best_val,
        )

        logger.info(f"Epoch {epoch} done in {(time.time() - epoch_start) / 60:.1f}m")

    try:
        _plot_loss_curve(
            train_losses,
            val_losses,
            results_dir / f"{exp_name}_loss_curve.png",
        )
    except Exception as e:
        logger.warning(f"Failed to plot loss curve: {e}")

    return {f"best_{monitor}": best_val}
